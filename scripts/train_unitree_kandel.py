# This is the training script to train feedforward and meta policies from
# `kandel`. Some parts of the code are inspired by the evaluator of `evosax`
# and `evojax`.
# - `evosax`: https://github.com/RobertTLange/evosax
# - `evojax`: https://github.com/google/evojax

import datetime
import os
import pathlib
import time
from collections import deque
from functools import partial
from importlib.metadata import version
from typing import Any, Sequence, Union

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax.serialization import to_state_dict
from hydra.core.hydra_config import HydraConfig
from ml_collections import config_dict
from mujoco_playground import registry
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import crawler_playground  # noqa: F401 # pylint:disable=unused-import
from crawler_playground.envs.unitree_go1.randomize import domain_randomize
from kandel.flatten import ParameterReshaper
from kandel.model import networks
from kandel.strategy import strategies
from kandel.utility.checkpoint_loader import (
    load_playground_environment,
    load_policy,
)
from kandel.utility.checkpoint_manager import CheckpointManager
from kandel.utility.running_statistics import (
    init_state,
    normalize,
    update,
)

# Constants
TIMESTAMP_FORMAT = "%y%m%d%H%M%S"
DUMMY_POLICY_SEED = 0
NORMALIZATION_COUNT_OVERRIDE = 1e6

# Set GPU parameters
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
jax.config.update("jax_default_matmul_precision", "highest")


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
    print(f"Strategy: \t\t{cfg.strategy_id}")
    print(f"Policy: \t\t{cfg.policy_id}")
    print(f"Environments: \t\t{cfg.environment_id}")
    print(
        f"Directory: \t\t{os.path.join(cfg.checkpoint_directory, cfg.run_id)}"
    )
    print(f"Agent Seed: \t\t{cfg.agent_seed}")
    print(f"Environment Seed: \t{cfg.environment_seed}")
    print()


@hydra.main(
    config_path="../config",
    config_name="unitree_kandel",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = HydraConfig.get()
    cfg["strategy_id"] = hydra_cfg.runtime.choices["strategy"]
    cfg["policy_id"] = hydra_cfg.runtime.choices["policy"]
    cfg["environment_id"] = hydra_cfg.runtime.choices["playground"]

    evolve(cfg)


def tile_policy(
    policy_params: Any, parameter_reshaper: ParameterReshaper, repeats: int
):
    """Duplicate policy parameters to parallelize repeated evaluation.
    Parameter vector:
    """
    flat_policy_params = parameter_reshaper.flatten(policy_params)
    tiled_flat_policy_params = jnp.repeat(
        flat_policy_params, repeats=repeats, axis=0
    )
    tiled_policy_params = parameter_reshaper.reshape(tiled_flat_policy_params)
    return tiled_policy_params


def detile_rewards(
    rewards: jax.typing.ArrayLike, repeats: int
) -> jax.typing.ArrayLike:
    """Detile rewards from parallelized evaluation."""
    return jnp.mean(rewards.ravel().reshape((-1, repeats)), axis=-1)


def compute_accumulated_rewards(
    rollout_rewards, rollout_dones, track_terminations=False
):
    if track_terminations:
        return jnp.sum(rollout_rewards * rollout_dones, axis=1)
    return jnp.sum(rollout_rewards, axis=1)


def compute_accumulated_lengths(rollout_dones):
    return jnp.sum(rollout_dones, axis=1)


def setup_environment(
    rng: Union[int, Sequence[int]],
    cfg: DictConfig,
    reward_override_cfg: dict = None,
    use_randomization: bool = False,
) -> Any:
    """Update domain randomization environments with config overrides."""
    raw = OmegaConf.to_container(cfg.playground, resolve=True)  # pure dict

    # Apply any config overrides
    if reward_override_cfg:
        for key, value in reward_override_cfg.items():
            raw["reward_config"]["scales"][key] = value

    overrides_cfg = config_dict.create(**raw)
    envs = registry.load(
        cfg.environment_id,
        config_overrides=overrides_cfg,
    )

    envs_batch_size = cfg.strategy.population_size * cfg.repeats

    # Apply domain randomization if enabled
    if use_randomization:
        rngs = jax.random.split(rng, envs_batch_size)
        rand_fn = partial(domain_randomize, rng=rngs)
        envs = BraxDomainRandomizationVmapWrapper(
            env=envs, randomization_fn=rand_fn
        )
        envs_reset_fn = jax.jit(envs.reset)
        envs_step_fn = jax.jit(envs.step)
    else:
        envs_reset_fn = jax.jit(jax.vmap(envs.reset))
        envs_step_fn = jax.jit(jax.vmap(envs.step))

    return envs, envs_reset_fn, envs_step_fn


def update_adaptive_training(
    cfg: DictConfig,
    threshold_buffer: deque,
    mean_reward: float,
    randomization_enabled: bool,
    reward_override_enabled: bool,
    rng_env,
    reward_override_cfg: dict,
) -> tuple:
    """Handle adaptive randomization and config updates."""

    # Skip processing if both randomization and reward overrides are already active
    if randomization_enabled and reward_override_enabled:
        return (
            False,
            randomization_enabled,
            reward_override_enabled,
            reward_override_cfg,
            None,
            None,
            None,
        )

    threshold_buffer.append(mean_reward)

    # Early return if buffer not full
    if len(threshold_buffer) < cfg.curriculum_window_size:
        return (
            False,
            randomization_enabled,
            reward_override_enabled,
            reward_override_cfg,
            None,
            None,
            None,
        )

    buffer_mean = jnp.mean(jnp.array(threshold_buffer))
    env_update_flag = False

    # Enable randomization if threshold is passed and not already enabled
    if buffer_mean > cfg.randomization_threshold and not randomization_enabled:
        randomization_enabled = True
        env_update_flag = True

        if cfg.verbose:
            print(
                f"Domain randomization enabled at threshold {cfg.randomization_threshold}."
            )

    # Update reward overrides if conditions met
    if (
        buffer_mean > cfg.reward_override_threshold
        and not reward_override_enabled
    ):
        reward_override_cfg = OmegaConf.to_container(
            cfg.reward_override_cfg, resolve=True
        )
        for key, value in reward_override_cfg.items():
            if key in reward_override_cfg:
                reward_override_cfg[key] = value
        env_update_flag = True
        reward_override_enabled = True

        if cfg.verbose:
            print("Reward function updated.")

    # Update environment if needed
    envs, envs_reset_fn, envs_step_fn = None, None, None
    if env_update_flag:
        envs, envs_reset_fn, envs_step_fn = setup_environment(
            rng=rng_env,
            cfg=cfg,
            use_randomization=randomization_enabled,
            reward_override_cfg=reward_override_cfg,
        )
        threshold_buffer.clear()

    return (
        env_update_flag,
        randomization_enabled,
        reward_override_enabled,
        reward_override_cfg,
        envs,
        envs_reset_fn,
        envs_step_fn,
    )


def log_generation_metrics(
    cfg: DictConfig,
    generation: int,
    accumulated_rewards,
    accumulated_lengths,
    opt_params,
    strategy_state,
    timing_metrics: dict,
    steps_global: float,
    steps_wo_termination: float,
    randomization_enabled: bool,
    reward_override_cfg: dict,
    eval_metrics: dict = {},
):
    """Log generation metrics to wandb."""
    if not cfg.wandb:
        return

    generation_log = {
        "evolution/generation": generation,
        "evolution/fitness_min": accumulated_rewards.min(),
        "evolution/fitness_mean": accumulated_rewards.mean(),
        "evolution/fitness_max": accumulated_rewards.max(),
        "evolution/length_min": accumulated_lengths.min(),
        "evolution/length_mean": accumulated_lengths.mean(),
        "evolution/length_max": accumulated_lengths.max(),
        "evolution/max_opt_param": opt_params.max(),
        "evolution/min_opt_param": opt_params.min(),
        "evolution/mean_opt_param": opt_params.mean(),
        "evolution/std_opt_param": opt_params.std(),
        "steps/global": steps_global,
        "steps/non_terminated": steps_wo_termination,
        "training/randomization_enabled": int(randomization_enabled),
    }

    # Add reward override values from config overrides (if active)
    generation_log.update(
        {f"reward/{k}": v for k, v in reward_override_cfg.items()}
    )

    # Evaluation metrics
    generation_log.update({f"eval/{k}": v for k, v in eval_metrics.items()})

    # Add timing metrics
    generation_log.update({f"time/{k}": v for k, v in timing_metrics.items()})

    # Add strategy metrics
    strategy_metrics = to_state_dict(strategy_state)
    generation_log.update(
        {f"strategy/{k}": v for k, v in strategy_metrics.items()}
    )

    wandb.log(generation_log, step=generation)


def evaluate_best_policy(
    cfg: DictConfig,
    policy_params,  # Full population policy params
    accumulated_rewards,  # Population rewards to find best policy
    policy_parameter_reshaper,
    policy_reset_fcn,
    envs_reset_fn,
    norm_state,
    rng_key,
    run_rollout,  # Reuse the existing run_rollout function
):
    """Evaluate the best policy and return metrics by reusing the main run_rollout function."""

    # Get best policy from current population using the correct tiling approach
    best_idx = jnp.argmax(accumulated_rewards)
    num_evals = cfg.repeats * cfg.strategy.population_size

    # Flatten all policies, select the best one, and tile it properly
    flat_policy_params = policy_parameter_reshaper.flatten(policy_params)
    best_flat_policy_params = flat_policy_params[best_idx].reshape(1, -1)
    tiled_flat_policy_params = jnp.repeat(
        best_flat_policy_params, repeats=num_evals, axis=0
    )
    eval_policy_params = policy_parameter_reshaper.reshape(
        tiled_flat_policy_params
    )

    # Create evaluation episodes
    rng_keys = jax.random.split(rng_key, num_evals)
    eval_env_states = envs_reset_fn(rng_keys)

    # Reset policy parameters for Hebbian learning (if enabled)
    if cfg.generation_reset:
        reset_keys = jax.random.split(rng_key, num_evals)
        eval_policy_params = policy_reset_fcn(eval_policy_params, reset_keys)

    # Initialize tracking arrays (same structure as training)
    active_envs = jnp.ones((num_evals,), dtype=jnp.int32)
    rollout_dones = jnp.zeros((num_evals, cfg.rollout_length), dtype=jnp.int32)
    rollout_rewards = jnp.zeros(
        (num_evals, cfg.rollout_length), dtype=jnp.float32
    )

    # Run evaluation rollout using the existing run_rollout function
    carry = (
        eval_env_states,
        eval_policy_params,
        norm_state,
        active_envs,
        rollout_dones,
        rollout_rewards,
    )
    _, acc_metrics = run_rollout(carry)

    eval_stats = {}

    for key in acc_metrics.keys():
        metric_values = acc_metrics[key]

        # Average over time for each episode, then compute stats across episodes
        metric_summed = jnp.sum(metric_values, axis=0)
        eval_stats[f"{key}_sum_mean"] = float(jnp.mean(metric_summed))

        metric_mean = jnp.mean(metric_values, axis=0)
        eval_stats[f"{key}_mean_median"] = float(jnp.median(metric_mean))

    return eval_stats


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
):
    """Factory to create jitted eval_step and run_rollout for current env and policy functions."""

    def eval_step(carry, t):
        """Single evaluation step (env + policy update)."""

        def step_fn(carry):
            (
                env_states,
                policy_params,
                norm_state,
                active_envs,
                rollout_dones,
                rollout_rewards,
            ) = carry

            norm_state = norm_update_fcn(norm_state, env_states.obs)
            obs = obs_normalize_fcn(env_states.obs, norm_state)
            policy_params, actions = policy_forward_fcn(policy_params, obs)
            actions = jnp.clip(
                actions, cfg.action_clip_min, cfg.action_clip_max
            )
            env_states = envs_step_fn(env_states, actions)

            current_done = env_states.done.astype(jnp.int32)
            active_envs = active_envs * (1 - current_done)
            rollout_dones = rollout_dones.at[:, t].set(active_envs)
            rollout_rewards = rollout_rewards.at[:, t].set(env_states.reward)

            return (
                env_states,
                policy_params,
                norm_state,
                active_envs,
                rollout_dones,
                rollout_rewards,
            )

        # Perform environment step if needed
        carry = jax.lax.cond(
            t % relative_sim_steps == 0,
            step_fn,
            lambda x: x,  # skip step
            carry,
        )

        def update_fn(carry):
            (
                env_states,
                policy_params,
                norm_state,
                active_envs,
                rollout_dones,
                rollout_rewards,
            ) = carry
            policy_params = policy_update_fcn(policy_params)
            return (
                env_states,
                policy_params,
                norm_state,
                active_envs,
                rollout_dones,
                rollout_rewards,
            )

        # Perform update step if needed
        carry = jax.lax.cond(
            t % relative_update_steps == 0,
            update_fn,
            lambda x: x,  # skip update
            carry,
        )

        # Always accumulate the current state, with a marker if it was a sim step
        metrics = carry[
            0
        ].metrics  # full environment state creates huge footprint.
        return carry, metrics

    jitted_eval_step = jax.jit(eval_step)
    jitted_run_rollout = jax.jit(
        lambda carry: jax.lax.scan(
            jitted_eval_step, carry, jnp.arange(total_steps)
        )
    )
    return jitted_run_rollout


def evolve(cfg: DictConfig) -> None:
    """Main evolution loop for training policies."""

    # Environment settings
    rng_agent = jax.random.key(cfg.agent_seed)
    rng_env, rng_agent = jax.random.split(rng_agent)
    if cfg.environment_seed is not None:
        rng_env = jax.random.key(cfg.environment_seed)

    # Vectorized environment functions
    randomization_enabled = False
    reward_override_enabled = False
    reward_override_cfg = OmegaConf.to_container(
        cfg.playground.reward_config.scales, resolve=True
    )

    envs, envs_reset_fn, envs_step_fn = setup_environment(
        rng=rng_env,
        cfg=cfg,
        use_randomization=randomization_enabled,
        reward_override_cfg=reward_override_cfg,
    )

    sim_interval = envs.dt
    update_interval = sim_interval
    if cfg.update_interval is not None:
        update_interval = cfg.update_interval

    # Policy settings
    policy = networks[cfg.policy_id](
        input_dim=envs.observation_size,
        output_dim=envs.action_size,
        **cfg.policy,
    )
    dummy_params = policy.initialize(jax.random.key(DUMMY_POLICY_SEED))
    dummy_opt_params = policy.opt_params(dummy_params)

    policy_parameter_reshaper = ParameterReshaper(dummy_params)
    evolution_parameter_reshaper = ParameterReshaper(dummy_opt_params)

    # Opimizer settings
    strategy = strategies[cfg.strategy_id](
        num_dims=evolution_parameter_reshaper.total_params,
        **cfg.strategy,
    )

    # Load logging settings
    time_stamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    run_id = (
        f"{time_stamp}"
        f"-{cfg.environment_id.lower()}"
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
    cfg["action_dim"] = envs.action_size
    cfg["observation_dim"] = envs.observation_size
    cfg["default_pose"] = envs._default_pose.tolist()

    package_versions = {
        "hebbian-attractors": version("hebbian-attractors"),
        "mujoco": version("mujoco"),
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "flax": version("flax"),
        "playground": version("playground"),
    }
    cfg["package_versions"] = package_versions

    # Print configuration
    huzzah(cfg)

    # Setup checkpointing
    if cfg.checkpoint_interval > 0:
        # Adding kandel version for export and import purposes
        save_directory = pathlib.Path(cfg.checkpoint_directory)
        checkpoint_directory = save_directory / run_id
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        # save population checkpoints
        checkpointer = CheckpointManager(
            save_interval_steps=cfg.checkpoint_interval,
            save_directory=checkpoint_directory,
            number_elites=cfg.checkpoint_elites,
        )
        # save metadata
        OmegaConf.save(cfg, checkpoint_directory / "metadata.yaml")

        if cfg.ckpt_load_dir is not None:
            ckpt_load_dir = pathlib.Path(cfg.ckpt_load_dir)
            cfg_ckpt = OmegaConf.load(ckpt_load_dir / "metadata.yaml")
            OmegaConf.save(
                cfg_ckpt, checkpoint_directory / "metadata_ckpt.yaml"
            )

    # Setup experiment tracking
    if cfg.wandb:
        wandb.init(
            project=f"{cfg.environment_id.lower()}_{cfg.wandb_project}",
            name=run_id,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg),
        )

    # Vectorized environment and policy functions
    policy_forward_fcn = jax.jit(jax.vmap(policy.forward))
    policy_update_fcn = jax.jit(jax.vmap(policy.update))
    policy_reset_fcn = jax.jit(jax.vmap(policy.reset))

    # Observation normalization
    norm_update_fcn = jax.jit(update)
    obs_normalize_fcn = jax.jit(normalize)

    # Decoupled Hebbian update frequency, compute relative time steps
    # update steps are scaled to simulation steps, therefore number of
    # simulation interactions depends solely on rollout length.
    total_time = cfg.rollout_length * sim_interval
    total_steps = int(total_time / min(sim_interval, update_interval))

    sim_steps = cfg.rollout_length
    update_steps = int(total_time / update_interval)

    relative_sim_steps = int(total_steps / sim_steps)
    relative_update_steps = int(total_steps / update_steps)

    # Create initial jitted functions
    run_rollout = make_envs_fn(
        envs_step_fn,
        policy_forward_fcn,
        policy_update_fcn,
        norm_update_fcn,
        obs_normalize_fcn,
        cfg,
        relative_sim_steps,
        relative_update_steps,
        total_steps,
    )

    # Initialize observation normalization
    dummy_obs = jnp.zeros((envs.observation_size,))
    norm_state = init_state(dummy_obs)

    # Load policy params (either from ckpt or scratch)
    rng_agent, rng_policy = jax.random.split(rng_agent, 2)
    keys = jax.random.split(rng_policy, cfg.strategy.population_size)
    policy_params_init = jax.vmap(policy.initialize)(keys)

    print("Enter checkpoint loading ...")
    if cfg.ckpt_load_dir is not None:
        # TODO: Loading is still not correct.
        # assert NotImplementedError

        _, loaded_policy_params = load_policy(
            checkpoint_directory=cfg.ckpt_load_dir,
            checkpoint=cfg.ckpt_load_ckpt,
            policy_id=cfg.ckpt_load_plc_id,
        )
        opt_params = policy.opt_params(loaded_policy_params)
        mean = evolution_parameter_reshaper.flatten_single(opt_params)

        _, norm_state, _ = load_playground_environment(
            checkpoint_directory=cfg.ckpt_load_dir,
            checkpoint=cfg.ckpt_load_ckpt,
        )

        norm_state = norm_state.replace(count=NORMALIZATION_COUNT_OVERRIDE)

        if cfg.verbose:
            print("Loaded policy from checkpoint.")
    else:
        # Initialize policy parameters from scratch
        opt_params = policy.opt_params(policy_params_init)
        flattened_params = evolution_parameter_reshaper.flatten(opt_params)
        mean = jnp.average(flattened_params, axis=0)

        if cfg.verbose:
            print("Training policy from scratch.")

    rng_agent, rng_strategy = jax.random.split(rng_agent, 2)
    strategy_state = strategy.initialize(rng_strategy)
    strategy_state = strategy.set_mean(strategy_state, mean)

    # Consider ignore terminations / check termination / auto reset
    active_envs = jnp.ones(
        (cfg.strategy.population_size * cfg.repeats,), dtype=jnp.int32
    )
    rollout_dones = jnp.zeros(
        (cfg.strategy.population_size * cfg.repeats, cfg.rollout_length),
        dtype=jnp.int32,
    )
    rollout_rewards = jnp.zeros(
        (cfg.strategy.population_size * cfg.repeats, cfg.rollout_length),
        dtype=jnp.float32,
    )

    # Randomization buffer
    threshold_buffer = deque(maxlen=cfg.curriculum_window_size)

    # Setup progress bar
    generations = range(cfg.num_generations)
    if cfg.progress_bar:
        generations = tqdm(
            range(cfg.num_generations),
            ascii=True,
            desc="Generation",
            position=0,
        )

    steps_global = 0.0
    steps_wo_termination = 0.0
    for generation in generations:
        start_generation_time = time.time()

        # Reset environments
        start_reset_time = time.time()
        rng_env, _rng_env = jax.random.split(rng_env)
        keys = jax.random.split(
            _rng_env, cfg.strategy.population_size * cfg.repeats
        )
        if cfg.fair_repeats:
            raise NotImplementedError  # TODO
            # All policies are compared with same testing conditions.
            # Policy: P1 P1 P1 P2 P2 P2 P3 P3 P3
            # Random: R1 R2 R3 R1 R2 R3 R1 R2 R3
            keys = jax.random.split(_rng_env, cfg.repeats)
            keys = jnp.tile(keys, (cfg.strategy.population_size, 1))

        env_states = envs_reset_fn(keys)
        reset_time = time.time() - start_reset_time

        # Generate population parameters from strategy
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

        # Tile policy parameters and reshape for evaluation
        eval_policy_params = tile_policy(
            policy_params, policy_parameter_reshaper, cfg.repeats
        )

        # Reset policy parameters to random init for Hebbian learning
        if cfg.generation_reset:
            rng_agent, _rng_policy = jax.random.split(rng_agent)

            # All polices are compared with varying testing conditions.
            # Policy: P1 P1 P1 P2 P2 P2 P3 P3 P3
            # Random: R1 R2 R3 R4 R5 R6 R7 R8 R9
            keys = jax.random.split(
                _rng_policy, cfg.strategy.population_size * cfg.repeats
            )
            if cfg.fair_repeats:
                raise NotImplementedError  # TODO
                # All policies are compared with same testing conditions.
                # Policy: P1 P1 P1 P2 P2 P2 P3 P3 P3
                # Random: R1 R2 R3 R1 R2 R3 R1 R2 R3
                reset_keys = jax.random.split(_rng_policy, cfg.repeats)
                keys = jnp.tile(
                    reset_keys, (cfg.strategy.population_size * cfg.repeats, 1)
                )

            eval_policy_params = policy_reset_fcn(eval_policy_params, keys)

        # Evaluate the population
        carry = (
            env_states,
            eval_policy_params,
            norm_state,
            active_envs,
            rollout_dones,
            rollout_rewards,
        )
        start_simulation_time = time.time()
        carry, _ = run_rollout(carry)
        simulation_time = time.time() - start_simulation_time

        (
            env_states,
            eval_policy_params,
            norm_state,
            active_envs,
            rollout_dones,
            rollout_rewards,
        ) = carry

        accumulated_rewards = compute_accumulated_rewards(
            rollout_rewards,
            rollout_dones,
            track_terminations=cfg.track_terminations,
        )
        accumulated_lengths = compute_accumulated_lengths(rollout_dones)

        # Detile rewards
        accumulated_rewards = detile_rewards(accumulated_rewards, cfg.repeats)

        # Update strategy
        start_tell_time = time.time()
        strategy_state = strategy.tell(
            opt_params, accumulated_rewards, strategy_state
        )
        tell_time = time.time() - start_tell_time

        # Compute total generation time
        generation_time = time.time() - start_generation_time

        # Run evaluation if enabled and at the right interval
        eval_stats = {}
        if cfg.eval_interval > 0 and generation % cfg.eval_interval == 0:
            rng_env, eval_rng = jax.random.split(rng_env)
            eval_stats = evaluate_best_policy(
                cfg=cfg,
                policy_params=policy_params,
                accumulated_rewards=accumulated_rewards,
                policy_parameter_reshaper=policy_parameter_reshaper,
                policy_reset_fcn=policy_reset_fcn,
                envs_reset_fn=envs_reset_fn,
                norm_state=norm_state,
                rng_key=eval_rng,
                run_rollout=run_rollout,
            )

            if cfg.verbose:
                print(
                    f"Eval Gen {generation}: "
                    f"reward={eval_stats.get('eval/reward_sum_mean', 0.0):.2f}"
                )

        if cfg.checkpoint_interval > 0:
            environment_metadata = {}
            environment_metadata[cfg.environment_id] = {
                "mean": norm_state.mean.tolist(),
                "std": norm_state.std.tolist(),
                "summed_variance": norm_state.summed_variance.tolist(),
                "count": int(norm_state.count),
            }

            checkpointer.save(
                policy_params=policy_params,
                rewards=accumulated_rewards,
                meta_data=environment_metadata,
            )

        # Performing logging metrics to wandb
        if cfg.wandb:
            steps_global += (
                cfg.strategy.population_size * cfg.repeats * cfg.rollout_length
            )
            steps_wo_termination += jnp.sum(rollout_dones)

            # Log generation metrics
            timing_metrics = {
                "ask_time": ask_time,
                "simulation_time": simulation_time,
                "tell_time": tell_time,
                "generation_time": generation_time,
                "reset_time": reset_time,
            }

            log_generation_metrics(
                cfg,
                generation,
                accumulated_rewards,
                accumulated_lengths,
                opt_params,
                strategy_state,
                timing_metrics,
                steps_global,
                steps_wo_termination,
                randomization_enabled,
                reward_override_cfg,
                eval_stats,
            )

        if cfg.verbose:
            print(
                f"Generation {generation}: {accumulated_rewards.min():.2f} / "
                f"{accumulated_rewards.mean():.2f} / {accumulated_rewards.max():.2f}"
            )

        # Update adaptive training parameters (skip if both systems are already active)
        if not (randomization_enabled and reward_override_enabled):
            (
                env_update_flag,
                randomization_enabled,
                reward_override_enabled,
                reward_override_cfg,
                new_envs,
                new_reset_fn,
                new_step_fn,
            ) = update_adaptive_training(
                cfg,
                threshold_buffer,
                accumulated_rewards.mean(),
                randomization_enabled,
                reward_override_enabled,
                rng_env,
                reward_override_cfg,
            )

            if env_update_flag:
                print(reward_override_cfg)
                # Re-jitting step and reset is costly,
                # only do when required.
                envs, envs_reset_fn, envs_step_fn = (
                    new_envs,
                    new_reset_fn,
                    new_step_fn,
                )

                # Re-create jitted functions with updated environment functions
                run_rollout = make_envs_fn(
                    envs_step_fn,
                    policy_forward_fcn,
                    policy_update_fcn,
                    norm_update_fcn,
                    obs_normalize_fcn,
                    cfg,
                    relative_sim_steps,
                    relative_update_steps,
                    total_steps,
                )

    if cfg.checkpoint_interval > 0:
        checkpointer.close()

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

    # Future features:
    # TODO: [~] checkpointing: continue training from checkpoint -> also load strategy_state?
    # TODO: [ ] checkpointing: auto best policy export

    # TODO: [ ] logging: run evaluation runs and log metrics
    # TODO: [ ] seeding: fair policy repeats
    # TODO: [ ] algorithm: warm start policy
    # TODO: [ ] compute: docker container for cluster deployment

    # TODO: [ ] bug: running statistics `count` overflow
    # TODO: [ ] bug: wandb `steps_global` overflow
    # TODO: [ ] strategy.tell: jit compiling
