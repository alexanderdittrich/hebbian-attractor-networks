# Training script for the Najarro et al. (2020) ant mutilated experiment.
#
# Evolves Hebbian learning rules across multiple morphological variants of
# the MuJoCo ant. Each candidate rule is evaluated on all training variants
# simultaneously in a single batched rollout. Validation is performed on an
# unseen held-out variant to measure generalisation of the learned plasticity
# rule.
#
# Najarro setup (reproduced here):
#   Train variants : no-damage + front-right leg (FR) damaged
#   Test variant   : front-left leg (FL) damaged  (unseen during training)
#   Leg damage     : ankle geom endpoint shortened 0.4 → 0.1 (75 %)
#   Episode length : 1 000 timesteps
#   Fitness        : average distance walked across training variants
#
# Reference:
#   Najarro & Risi, "Meta-Learning through Hebbian Plasticity in Random
#   Networks", NeurIPS 2020.

import datetime
import os
import pathlib
import time
from functools import partial
from importlib.metadata import version
from typing import Any, Dict, List, Optional

import hydra
import jax
import jax.numpy as jnp
import wandb
from flax.serialization import to_state_dict
from hydra.core.hydra_config import HydraConfig
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from crawler_playground.envs.ant.ant import Run as AntRun
from crawler_playground.envs.ant.ant import default_config
from crawler_playground.envs.ant.randomize import domain_randomize
from kandel.flatten import ParameterReshaper
from kandel.model import networks
from kandel.strategy import strategies
from kandel.utility.checkpoint_manager import CheckpointManager
from kandel.utility.running_statistics import (
    init_state,
    normalize,
    update,
)

# ── GPU / XLA flags ────────────────────────────────────────────────────────────

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

TIMESTAMP_FORMAT = "%y%m%d%H%M%S"
DUMMY_POLICY_SEED = 0


# ── Environment helpers ────────────────────────────────────────────────────────


def create_base_env(env_overrides: Optional[Dict] = None) -> AntRun:
    return AntRun(
        config=default_config(),
        config_overrides=env_overrides or {},
    )


def make_wrapped_env(
    base_env: AntRun,
    variants: List[str],
    batch_size: int,
) -> tuple:
    """Return (jit step_fn, jit reset_fn) via BraxDomainRandomizationVmapWrapper.

    Follows the mujoco_playground domain-randomisation convention, making this
    environment compatible with both Kandel ES and Brax PPO training.
    Variants are assigned round-robin so every policy sees every variant an
    equal number of times per generation.
    """
    rand_fn = partial(
        domain_randomize,
        mj_model=base_env.mj_model,
        variants=variants,
        batch_size=batch_size,
    )
    wrapped = BraxDomainRandomizationVmapWrapper(base_env, rand_fn)
    return jax.jit(wrapped.step), jax.jit(wrapped.reset)


# ── Utility (shared with train_playground_kandel.py) ──────────────────────────


def tile_policy(
    policy_params: Any, parameter_reshaper: ParameterReshaper, repeats: int
):
    flat = parameter_reshaper.flatten(policy_params)
    tiled = jnp.repeat(flat, repeats=repeats, axis=0)
    return parameter_reshaper.reshape(tiled)


def detile_rewards(
    rewards: jax.typing.ArrayLike, repeats: int
) -> jax.typing.ArrayLike:
    return jnp.mean(rewards.ravel().reshape((-1, repeats)), axis=-1)


def make_envs_fn(
    envs_step_fn,
    policy_forward_fcn,
    policy_update_fcn,
    norm_update_fcn,
    obs_normalize_fcn,
    cfg,
    relative_sim_steps,
    relative_update_steps,
    total_steps,
    track_terminations,
):
    """Return a JIT-compiled rollout function.

    Carry: (env_states, policy_params, norm_state, active_envs, accumulated_rewards)

    Rewards are accumulated step-by-step into a running sum, replacing the
    previous (pop × rollout_length) scatter buffers.  This eliminates two
    large intermediary arrays and the indexed-update scatter at every step.

    When relative_sim_steps == relative_update_steps == 1 (the common case for
    this environment), the lax.cond branches are bypassed entirely.
    """
    always_step = relative_sim_steps == 1
    always_update = relative_update_steps == 1

    @jax.jit
    def eval_step(carry, t):
        (
            env_states,
            policy_params,
            norm_state,
            active_envs,
            accumulated_rewards,
        ) = carry

        def step_fn(args):
            (
                env_states,
                policy_params,
                norm_state,
                active_envs,
                accumulated_rewards,
            ) = args
            norm_state = norm_update_fcn(norm_state, env_states.obs)
            obs = obs_normalize_fcn(env_states.obs, norm_state)
            policy_params, actions = policy_forward_fcn(policy_params, obs)
            actions = jnp.clip(
                actions, cfg.action_clip_min, cfg.action_clip_max
            )
            env_states = envs_step_fn(env_states, actions)

            current_done = env_states.done.astype(jnp.int32)
            active_envs = active_envs * (1 - current_done)
            if track_terminations:
                reward = env_states.reward * active_envs.astype(jnp.float32)
            else:
                reward = env_states.reward
            accumulated_rewards = accumulated_rewards + reward
            return (
                env_states,
                policy_params,
                norm_state,
                active_envs,
                accumulated_rewards,
            )

        if always_step:
            carry = step_fn(carry)
        else:
            carry = jax.lax.cond(
                t % relative_sim_steps == 0, step_fn, lambda x: x, carry
            )

        (
            env_states,
            policy_params,
            norm_state,
            active_envs,
            accumulated_rewards,
        ) = carry

        def update_fn(args):
            (policy_params,) = args
            return (policy_update_fcn(policy_params),)

        if always_update:
            policy_params = policy_update_fcn(policy_params)
        else:
            (policy_params,) = jax.lax.cond(
                t % relative_update_steps == 0,
                update_fn,
                lambda x: x,
                (policy_params,),
            )

        carry = (
            env_states,
            policy_params,
            norm_state,
            active_envs,
            accumulated_rewards,
        )
        return carry, None

    @jax.jit
    def run_rollout(carry):
        return jax.lax.scan(eval_step, carry, jnp.arange(total_steps))

    return run_rollout


# ── Logging helpers ────────────────────────────────────────────────────────────


def huzzah(cfg):
    print()
    print(" dP                               dP          dP")
    print(" 88                               88          88")
    print(" 88  .dP  .d8888b. 88d888b. .d888b88 .d8888b. 88")
    print(" 88888'   88'  `88 88'  `88 88'  `88 88ooood8 88")
    print(" 88  `8b. 88.  .88 88    88 88.  .88 88.  ... 88")
    print(" dP   `YP `88888P8 dP    dP `88888P8 `88888P' dP")
    print("ooooooooooooooooooooooooooooooooooooooooooooooooo")
    print()
    print(f"Experiment:      \tNajarro Ant Mutilated")
    print(f"Strategy: \t\t{cfg.strategy_id}")
    print(f"Policy: \t\t{cfg.policy_id}")
    print(f"Train variants: \t{list(cfg.train_variants)}")
    print(f"Test variants: \t\t{list(cfg.test_variants)}")
    print(f"Agent Seed: \t\t{cfg.agent_seed}")
    print(f"Environment Seed: \t{cfg.environment_seed}")
    print()


# ── Hydra entry point ──────────────────────────────────────────────────────────


@hydra.main(
    config_path="../config",
    config_name="ant_mutilated_kandel",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = HydraConfig.get()
    cfg["strategy_id"] = hydra_cfg.runtime.choices["strategy"]
    cfg["policy_id"] = hydra_cfg.runtime.choices["policy"]
    evolve(cfg)


# ── Main evolution loop ────────────────────────────────────────────────────────


def evolve(cfg: DictConfig) -> None:
    # ── Environment setup ──────────────────────────────────────────────────────

    env_overrides = {}
    if "env" in cfg:
        env_overrides = OmegaConf.to_container(cfg.env, resolve=True)

    train_variants = list(cfg.train_variants)
    test_variants = list(cfg.test_variants)
    n_train_variants = len(train_variants)
    n_test_variants = len(test_variants)

    # Single base env — all variants share the same topology.
    base_env = create_base_env(env_overrides)
    ref_env = base_env

    # ── Policy setup ──────────────────────────────────────────────────────────

    policy = networks[cfg.policy_id](
        input_dim=ref_env.observation_size,
        output_dim=ref_env.action_size,
        **cfg.policy,
    )
    dummy_params = policy.initialize(jax.random.key(DUMMY_POLICY_SEED))
    dummy_opt_params = policy.opt_params(dummy_params)

    policy_parameter_reshaper = ParameterReshaper(dummy_params)
    evolution_parameter_reshaper = ParameterReshaper(dummy_opt_params)

    # ── Strategy setup ────────────────────────────────────────────────────────

    strategy = strategies[cfg.strategy_id](
        num_dims=evolution_parameter_reshaper.total_params,
        **cfg.strategy,
    )

    # ── Timescale setup ───────────────────────────────────────────────────────

    sim_interval = ref_env.dt
    update_interval = sim_interval
    if cfg.update_interval is not None:
        update_interval = cfg.update_interval

    total_time = cfg.rollout_length * sim_interval
    total_steps = int(total_time / min(sim_interval, update_interval))
    sim_steps = cfg.rollout_length
    update_steps = int(total_time / update_interval)
    relative_sim_steps = int(total_steps / sim_steps)
    relative_update_steps = int(total_steps / update_steps)

    # ── Batch sizes ───────────────────────────────────────────────────────────

    pop_size = cfg.strategy.population_size
    # Each policy sees every variant cfg.repeats times per generation.
    # Round-robin assignment: env i → variants[i % n_variants].
    train_repeats_total = cfg.repeats * n_train_variants  # slots per policy
    test_repeats_total = cfg.repeats * n_test_variants
    pop_train_size = pop_size * train_repeats_total
    pop_test_size = pop_size * test_repeats_total

    # ── Wrapped environments (mujoco_playground domain-rand convention) ───────

    train_step_fn, train_reset_fn = make_wrapped_env(
        base_env, train_variants, pop_train_size
    )
    test_step_fn, test_reset_fn = make_wrapped_env(
        base_env, test_variants, pop_test_size
    )

    # ── Policy functions ──────────────────────────────────────────────────────

    policy_forward_fcn = jax.jit(jax.vmap(policy.forward))
    policy_update_fcn = jax.jit(jax.vmap(policy.update))
    policy_reset_fcn = jax.jit(jax.vmap(policy.reset))
    norm_update_fcn = jax.jit(update)
    obs_normalize_fcn = jax.jit(normalize)

    # ── Rollout functions ─────────────────────────────────────────────────────

    rollout_kwargs = dict(
        policy_forward_fcn=policy_forward_fcn,
        policy_update_fcn=policy_update_fcn,
        norm_update_fcn=norm_update_fcn,
        obs_normalize_fcn=obs_normalize_fcn,
        cfg=cfg,
        relative_sim_steps=relative_sim_steps,
        relative_update_steps=relative_update_steps,
        total_steps=total_steps,
        track_terminations=cfg.track_terminations,
    )
    train_rollout = make_envs_fn(envs_step_fn=train_step_fn, **rollout_kwargs)
    test_rollout = make_envs_fn(envs_step_fn=test_step_fn, **rollout_kwargs)

    # ── Logging metadata ──────────────────────────────────────────────────────

    time_stamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    run_id = (
        f"{time_stamp}-ant-mutilated"
        f"-{cfg.policy_id.lower()}"
        f"-{cfg.strategy_id.lower()}"
        f"-{cfg.agent_seed}"
    )
    if cfg.environment_seed is not None:
        run_id += f"-{cfg.environment_seed}"

    cfg["run_id"] = run_id
    cfg["simulation_interval"] = sim_interval
    cfg["update_interval"] = update_interval
    cfg["num_evo_params"] = evolution_parameter_reshaper.total_params
    cfg["num_total_params"] = policy_parameter_reshaper.total_params
    cfg["action_dim"] = ref_env.action_size
    cfg["observation_dim"] = ref_env.observation_size
    cfg["package_versions"] = {
        "hebbian-attractors": version("hebbian-attractors"),
        "mujoco": version("mujoco"),
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "flax": version("flax"),
        "playground": version("playground"),
    }

    huzzah(cfg)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    if cfg.checkpoint_interval > 0:
        save_directory = pathlib.Path(cfg.checkpoint_directory)
        checkpoint_directory = save_directory / run_id
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpointer = CheckpointManager(
            save_interval_steps=cfg.checkpoint_interval,
            save_directory=checkpoint_directory,
            number_elites=cfg.checkpoint_elites,
        )
        OmegaConf.save(cfg, checkpoint_directory / "metadata.yaml")

    # ── W&B ───────────────────────────────────────────────────────────────────

    if cfg.wandb:
        wandb.init(
            project=f"ant-mutilated_{cfg.wandb_project}",
            name=run_id,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg),
        )

    # ── RNG setup ─────────────────────────────────────────────────────────────

    rng_agent = jax.random.key(cfg.agent_seed)
    rng_env, rng_agent = jax.random.split(rng_agent)
    if cfg.environment_seed is not None:
        rng_env = jax.random.key(cfg.environment_seed)

    # ── Initialise population ─────────────────────────────────────────────────

    rng_agent, rng_strategy, rng_policy = jax.random.split(rng_agent, 3)
    keys = jax.random.split(rng_policy, pop_size)
    policy_params_init = jax.vmap(policy.initialize)(keys)

    strategy_state = strategy.initialize(rng_strategy)
    opt_params = policy.opt_params(policy_params_init)
    flattened_params = evolution_parameter_reshaper.flatten(opt_params)
    mean = jnp.average(flattened_params, axis=0)
    strategy_state = strategy.set_mean(strategy_state, mean)

    # ── Observation normalisation ─────────────────────────────────────────────

    dummy_obs = jnp.zeros((ref_env.observation_size,))
    norm_state = init_state(dummy_obs)

    if cfg.verbose:
        print("Training Hebbian rules across morphological variants.")

    # ── Progress bar ──────────────────────────────────────────────────────────

    generations = range(cfg.num_generations)
    if cfg.progress_bar:
        generations = tqdm(
            range(cfg.num_generations),
            ascii=True,
            desc="Generation",
            position=0,
        )

    steps_global = 0.0

    # ── Evolution loop ────────────────────────────────────────────────────────

    for generation in generations:
        start_generation_time = time.time()

        # Ask
        rng_agent, _rng_strategy = jax.random.split(rng_agent)
        start_ask_time = time.time()
        opt_params, strategy_state = strategy.ask(
            _rng_strategy, strategy_state
        )
        ask_time = time.time() - start_ask_time

        policy_params = policy.update_opt_params(
            policy_params_init,
            evolution_parameter_reshaper.reshape(opt_params),
        )

        # Tile policies: each policy gets (repeats * n_train_variants) env slots.
        eval_policy_params = tile_policy(
            policy_params, policy_parameter_reshaper, train_repeats_total
        )

        # Reset Hebbian kernel weights for this generation.
        if cfg.generation_reset:
            rng_agent, _rng_policy = jax.random.split(rng_agent)
            reset_keys = jax.random.split(_rng_policy, pop_train_size)
            eval_policy_params = policy_reset_fcn(
                eval_policy_params, reset_keys
            )

        # Reset environments.
        rng_env, _rng_env = jax.random.split(rng_env)
        env_keys = jax.random.split(_rng_env, pop_train_size)
        env_states = train_reset_fn(env_keys)

        active_envs = jnp.ones((pop_train_size,), dtype=jnp.int32)
        accumulated_rewards = jnp.zeros((pop_train_size,), dtype=jnp.float32)

        # Single parallel rollout over all variants simultaneously.
        carry = (
            env_states,
            eval_policy_params,
            norm_state,
            active_envs,
            accumulated_rewards,
        )
        start_simulation_time = time.time()
        carry, _ = train_rollout(carry)
        simulation_time = time.time() - start_simulation_time

        (_, _, norm_state, _, accumulated_rewards) = carry

        # Reward layout (round-robin variant assignment):
        #   [P0_V0_R0, P0_V1_R0, P0_V0_R1, P0_V1_R1, ..., P1_V0_R0, ...]
        # Reshape to (pop, repeats, n_variants) for per-variant stats.
        per_pv = accumulated_rewards.reshape(
            pop_size, cfg.repeats, n_train_variants
        )
        per_variant_means = per_pv.mean(axis=(0, 1))  # [n_variants]
        fitness = detile_rewards(
            accumulated_rewards, train_repeats_total
        )  # [pop_size]

        # Tell
        start_tell_time = time.time()
        strategy_state = strategy.tell(opt_params, fitness, strategy_state)
        tell_time = time.time() - start_tell_time

        generation_time = time.time() - start_generation_time

        # ── Periodic test evaluation ──────────────────────────────────────────

        test_metrics = {}
        if (
            cfg.eval_test_interval > 0
            and generation % cfg.eval_test_interval == 0
        ):
            test_eval_params = tile_policy(
                policy_params, policy_parameter_reshaper, test_repeats_total
            )
            if cfg.generation_reset:
                rng_agent, _rng_policy = jax.random.split(rng_agent)
                test_reset_keys = jax.random.split(_rng_policy, pop_test_size)
                test_eval_params = policy_reset_fcn(
                    test_eval_params, test_reset_keys
                )

            rng_env, _rng_test = jax.random.split(rng_env)
            test_env_keys = jax.random.split(_rng_test, pop_test_size)
            test_env_states = test_reset_fn(test_env_keys)

            test_active = jnp.ones((pop_test_size,), dtype=jnp.int32)
            test_acc = jnp.zeros((pop_test_size,), dtype=jnp.float32)

            test_carry = (
                test_env_states,
                test_eval_params,
                norm_state,  # use train norm_state; discard updates
                test_active,
                test_acc,
            )
            test_carry, _ = test_rollout(test_carry)
            (_, _, _, _, test_acc) = test_carry

            # Round-robin layout → reshape (pop, repeats, n_test_variants).
            test_per_pv = test_acc.reshape(
                pop_size, cfg.repeats, n_test_variants
            )
            for i, variant in enumerate(test_variants):
                v_rewards = test_per_pv[:, :, i].mean(axis=1)  # [pop_size]
                test_metrics[f"test/{variant}_fitness_min"] = float(
                    v_rewards.min()
                )
                test_metrics[f"test/{variant}_fitness_mean"] = float(
                    v_rewards.mean()
                )
                test_metrics[f"test/{variant}_fitness_max"] = float(
                    v_rewards.max()
                )

        # ── Checkpointing ─────────────────────────────────────────────────────

        if cfg.checkpoint_interval > 0:
            checkpointer.save(
                policy_params=policy_params,
                rewards=fitness,
                meta_data={
                    "norm_state": {
                        "mean": norm_state.mean.tolist(),
                        "std": norm_state.std.tolist(),
                        "summed_variance": norm_state.summed_variance.tolist(),
                        "count": int(norm_state.count),
                    }
                },
            )

        # ── Logging ───────────────────────────────────────────────────────────

        if cfg.wandb:
            steps_global += pop_train_size * cfg.rollout_length

            generation_log = {
                "evolution/generation": generation,
                "evolution/fitness_min": float(fitness.min()),
                "evolution/fitness_mean": float(fitness.mean()),
                "evolution/fitness_max": float(fitness.max()),
                "evolution/max_opt_param": float(opt_params.max()),
                "evolution/min_opt_param": float(opt_params.min()),
                "evolution/mean_opt_param": float(opt_params.mean()),
                "evolution/std_opt_param": float(opt_params.std()),
                "time/simulation_time": simulation_time,
                "time/ask_time": ask_time,
                "time/tell_time": tell_time,
                "time/generation_time": generation_time,
                "steps/global": steps_global,
            }
            for i, variant in enumerate(train_variants):
                generation_log[f"train/{variant}_fitness_mean"] = float(
                    per_variant_means[i]
                )
            generation_log.update(test_metrics)

            strategy_metrics = to_state_dict(strategy_state)
            generation_log.update(
                {f"strategy/{k}": v for k, v in strategy_metrics.items()}
            )
            wandb.log(generation_log)

        if cfg.verbose:
            msg = (
                f"Generation {generation}: "
                f"train {fitness.min():.2f} / {fitness.mean():.2f} / {fitness.max():.2f}"
            )
            if test_metrics:
                for variant in test_variants:
                    key = f"test/{variant}_fitness_mean"
                    if key in test_metrics:
                        msg += f" | {variant} {test_metrics[key]:.2f}"
            print(msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    if cfg.checkpoint_interval > 0:
        checkpointer.close()
    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
