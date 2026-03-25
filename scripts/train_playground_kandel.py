# This is the training script to train feedforward and meta policies from
# `kandel`. Some parts of the code are inspired by the evaluator of `evosax`
# and `evojax`.
# - `evosax`: https://github.com/RobertTLange/evosax
# - `evojax`: https://github.com/google/evojax

import datetime
import os
import pathlib
import time
from functools import partial
from importlib.metadata import version
from typing import Any, Sequence, Union

import hydra
import jax
import jax.numpy as jnp
import wandb
from flax.serialization import to_state_dict
from hydra.core.hydra_config import HydraConfig
from ml_collections import ConfigDict
from mujoco_playground import registry
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import crawler_playground  # noqa: F401 # pylint:disable=unused-import
from kandel.flatten import ParameterReshaper
from kandel.model import networks
from kandel.strategy import strategies
from kandel.utility.checkpoint_manager import CheckpointManager
from kandel.utility.running_statistics import (
    init_state,
    normalize,
    update,
)

# Set GPU parameters
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
jax.config.update("jax_default_matmul_precision", "highest")

# Constants
TIMESTAMP_FORMAT = "%y%m%d%H%M%S"
DUMMY_POLICY_SEED = 0


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
    config_name="playground_kandel",
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

    @jax.jit
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

            # fill rollout dones mask
            current_done = env_states.done.astype(jnp.int32)
            active_envs = active_envs * (1 - current_done)
            rollout_dones = rollout_dones.at[:, t].set(active_envs)

            # fill reward mask
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

        return carry, None

    @jax.jit
    def run_rollout(carry):
        return jax.lax.scan(eval_step, carry, jnp.arange(total_steps))

    return run_rollout


def evolve(cfg: DictConfig) -> None:
    """Main evolution loop for training policies."""

    # Environment settings
    envs = registry.load(
        cfg.environment_id,
        # config_overrides=ConfigDict(cfg.playground),
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

    # Setup experiment tracking
    if cfg.wandb:
        wandb.init(
            project=f"{cfg.environment_id.lower()}_{cfg.wandb_project}",
            name=run_id,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg),
        )

    rng_agent = jax.random.key(cfg.agent_seed)
    rng_env, rng_agent = jax.random.split(rng_agent)
    if cfg.environment_seed is not None:
        rng_env = jax.random.key(cfg.environment_seed)

    # Vectorized environment functions
    envs_reset_fcn = jax.jit(jax.vmap(envs.reset))
    envs_step_fcn = jax.jit(jax.vmap(envs.step))

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
        envs_step_fcn,
        policy_forward_fcn,
        policy_update_fcn,
        norm_update_fcn,
        obs_normalize_fcn,
        cfg,
        relative_sim_steps,
        relative_update_steps,
        total_steps,
    )

    # Initialize policy parameters
    rng_agent, rng_strategy, rng_policy = jax.random.split(rng_agent, 3)
    keys = jax.random.split(rng_policy, cfg.strategy.population_size)
    policy_params_init = jax.vmap(policy.initialize)(keys)

    strategy_state = strategy.initialize(rng_strategy)

    # Set mean from policy distribution.
    opt_params = policy.opt_params(policy_params_init)
    flattened_params = evolution_parameter_reshaper.flatten(opt_params)
    mean = jnp.average(flattened_params, axis=0)
    strategy_state = strategy.set_mean(strategy_state, mean)

    if cfg.verbose:
        print("Training policy from scratch.")

    # Initialize observation normalization
    dummy_obs = jnp.zeros((envs.observation_size,))
    norm_state = init_state(dummy_obs)

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

        env_states = envs_reset_fcn(keys)
        reset_time = time.time() - start_reset_time

        # Reset rollout buffers for this generation
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

        if cfg.wandb:
            steps_global += (
                cfg.strategy.population_size * cfg.repeats * cfg.rollout_length
            )
            steps_wo_termination += jnp.sum(rollout_dones)

            # Logging generation metrics
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
                "time/ask_time": ask_time,
                "time/simulation_time": simulation_time,
                "time/tell_time": tell_time,
                "time/generation_time": generation_time,
                "time/reset_time": reset_time,
                "steps/global": steps_global,
                "steps/non_terminated": steps_wo_termination,
            }

            # Logging strategy metrics
            strategy_metrics = to_state_dict(strategy_state)
            strategy_metrics = {
                f"strategy/{k}": v for k, v in strategy_metrics.items()
            }
            generation_log.update(strategy_metrics)

            # Handover to wandb
            wandb.log(generation_log)

        if cfg.verbose:
            print(
                f"Generation {generation}: {accumulated_rewards.min():.2f} / "
                f"{accumulated_rewards.mean():.2f} / {accumulated_rewards.max():.2f}"
            )

    if cfg.checkpoint_interval > 0:
        checkpointer.close()

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

    # Future features:
    # TODO: [ ] checkpointing: continue training from checkpoint -> also load strategy_state?
    # TODO: [ ] checkpointing: auto best policy export

    # TODO: [ ] seeding: fair policy repeats
    # TODO: [ ] algorithm: warm start policy
    # TODO: [ ] compute: docker container for cluster deployment
    # TODO: [ ] logging: run evaluation runs and log metrics

    # TODO: [ ] bug: running statistics `count` overflow
    # TODO: [ ] bug: wandb `steps_global` overflow
    # TODO: [ ] strategy.tell: jit compiling
