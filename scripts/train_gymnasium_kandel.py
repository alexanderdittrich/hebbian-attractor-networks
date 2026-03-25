import datetime
import importlib.metadata
import pathlib
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from flax.serialization import to_state_dict
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import crawler_gym
from kandel.flatten import ParameterReshaper
from kandel.model import networks
from kandel.sim_api.gymnasium import (
    evaluate,
    extract_normalization_params,
    generate_single_env,
    generate_vector_env,
)
from kandel.strategy import strategies
from kandel.utility.checkpoint_manager import CheckpointManager


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
    print(f"Environment: \t\t{cfg.environment_id}")
    print(f"Directory: \t\t{cfg.checkpoint_directory}")
    print(f"Seed: \t\t\t{cfg.seed}")
    print()


@hydra.main(
    config_path="../config", config_name="gymnasium_kandel", version_base="1.3"
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = HydraConfig.get()
    cfg["strategy_id"] = hydra_cfg.runtime.choices["strategy"]
    cfg["policy_id"] = hydra_cfg.runtime.choices["policy"]
    cfg["environment_id"] = hydra_cfg.runtime.choices["gymnasium"]

    huzzah(cfg)
    evolve(cfg)


def evolve(cfg):
    rng = jax.random.key(seed=cfg.seed)
    rng, rng_policy, rng_strategy = jax.random.split(rng, 3)

    time_stamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    run_id = (
        f"{time_stamp}-"
        f"{cfg.environment_id[:-3].lower()}-"
        f"{cfg.policy_id.lower()}-"
        f"{cfg.strategy_id.lower()}-"
        f"{cfg.seed}"
    )

    # Compute update time
    dummy_env = generate_single_env(cfg.environment_id, **dict(cfg.gymnasium))
    step_interval = dummy_env.unwrapped.dt
    observation_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    dummy_env.close()
    del dummy_env

    # admittingly it requires some testing to find a good update interval as
    # it is also influence by the time spans set in the environment.
    update_interval = step_interval
    if cfg.update_interval is not None:
        update_interval = cfg.update_interval

    # Initialize environment
    # FEATURE: parallel training for highly parallelized environments
    envs = generate_vector_env(
        env_id=cfg.environment_id,
        num_envs=cfg.strategy.population_size,  # population
        multithreading=cfg.multithreading,
        max_episode_steps=cfg.episode_length,
        **cfg.gymnasium,
    )
    # FEATURE: Load from previous checkpoint.
    policy = networks[cfg.policy_id](
        input_dim=observation_dim,
        output_dim=action_dim,
        **cfg.policy,
    )

    # Single policy structure
    params = policy.initialize(rng_policy)
    opt_params = policy.opt_params(params)

    policy_reshaper = ParameterReshaper(params)
    evo_reshaper = ParameterReshaper(opt_params)

    # Policy ensemble
    keys = jax.random.split(rng_policy, cfg.strategy.population_size)
    params = jax.vmap(policy.initialize)(keys)
    opt_params = policy.opt_params(params)

    forward_fcn = jax.jit(jax.vmap(policy.forward))
    update_fcn = jax.jit(jax.vmap(policy.update))
    reset_fcn = jax.jit(jax.vmap(policy.reset))

    # Algorithm configuration
    strategy = strategies[cfg.strategy_id](
        num_dims=evo_reshaper.total_params,
        **cfg.strategy,
    )

    strategy_state = strategy.initialize(rng_strategy)

    # <------------ initial mean computation ------------>
    if cfg.warmstart:
        total_fitness = np.zeros((cfg.strategy.population_size, cfg.repeats))
        for i in range(cfg.repeats):
            rng, rng_reset, rng_eval = jax.random.split(rng, 3)

            # Reset initial random parameters uniformly
            if cfg.weight_reset:
                keys = jax.random.split(
                    rng_reset, cfg.strategy.population_size
                )
                params = reset_fcn(params, keys)

            # Evaluate population
            fitness, time_steps, _ = evaluate(
                random_key=rng_eval,
                # environment
                envs=envs,
                # policy
                forward_fcn=forward_fcn,
                update_fcn=update_fcn,
                policy_params=params,
                # timing params
                episode_steps=cfg.episode_length,
                environment_interval=step_interval,
                hebbian_update_interval=update_interval,
                # logging functionality
                progress_bar=False,
                ignore_terminations=True,
            )
            total_fitness[:, i] = fitness

        avg_fitness = np.mean(total_fitness, axis=1)

        # Initial mean from fitness-weighted population params
        flattened_params = evo_reshaper.flatten(opt_params)
        mean = np.average(flattened_params, weights=avg_fitness, axis=0)
        strategy_state = strategy_state.replace(mean=mean)
    else:
        # Initial mean from mean population params
        flattened_params = evo_reshaper.flatten(opt_params)
        mean = np.average(flattened_params, axis=0)
        strategy_state = strategy.set_mean(strategy_state, mean)

    # Logging
    cfg["step_interval"] = step_interval
    cfg["update_interval"] = update_interval
    cfg["num_evo_params"] = evo_reshaper.total_params
    cfg["num_total_params"] = policy_reshaper.total_params
    cfg["action_dim"] = action_dim
    cfg["observation_dim"] = observation_dim
    cfg["run_id"] = run_id

    package_versions = {
        "hebbian-dynamics": importlib.metadata.version("hebbian-dynamics"),
        "mujoco": importlib.metadata.version("mujoco"),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "flax": importlib.metadata.version("flax"),
        "gymnasium": importlib.metadata.version("gymnasium"),
    }
    cfg["package_versions"] = package_versions

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

    if cfg.wandb:
        wandb.init(
            project=f"{cfg.environment_id[:-3].lower()}_{cfg.wandb_project}",
            name=run_id,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg),
        )

    # Setup progress bar
    generations = range(cfg.num_generations)
    if cfg.generation_progress_bar:
        generations = tqdm.tqdm(
            range(cfg.num_generations),
            ascii=True,
            desc="Generation",
            position=0,
        )

    # Evolution loop
    global_time_steps = 0

    # Training loop
    for gen in generations:
        # Generation log
        generation_log = {}

        # Generate random keys
        rng, rng_gen = jax.random.split(rng)

        # Generate population
        ask_start_time = time.perf_counter()
        opt_params, strategy_state = strategy.ask(rng_gen, strategy_state)
        ask_time = time.perf_counter() - ask_start_time

        # Update policy parameters
        params = policy.update_opt_params(
            params, evo_reshaper.reshape(opt_params)
        )

        # Fitness parameters
        total_fitness = np.zeros((cfg.strategy.population_size, cfg.repeats))
        total_lengths = np.zeros((cfg.strategy.population_size, cfg.repeats))

        for i in range(cfg.repeats):
            rng, rng_reset, rng_eval = jax.random.split(rng, 3)

            # Reset
            if cfg.weight_reset:
                keys = jax.random.split(
                    rng_reset, cfg.strategy.population_size
                )
                params = reset_fcn(params, keys)

            # Evaluate population
            fitness, time_steps, eval_metrics = evaluate(
                random_key=rng_eval,
                # environment
                envs=envs,
                # policy
                forward_fcn=forward_fcn,
                update_fcn=update_fcn,
                policy_params=params,
                # timing params
                episode_steps=cfg.episode_length,
                environment_interval=step_interval,
                hebbian_update_interval=update_interval,
                # logging functionality
                progress_bar=cfg.episode_progress_bar,
                record_env_info=False,
                record_policy=False,
                ignore_terminations=cfg.ignore_terminations,
            )

            total_fitness[:, i] = fitness
            total_lengths[:, i] = time_steps
            global_time_steps += time_steps.sum()

            eval_metrics = {
                f"{cfg.environment_id[:-3].lower()}-{i}/{k}": v
                for k, v in eval_metrics.items()
            }
            generation_log.update(eval_metrics)

        # Fitness metrics
        avg_fitness = np.mean(total_fitness, axis=1)

        tell_start_time = time.perf_counter()
        strategy_state = strategy.tell(opt_params, avg_fitness, strategy_state)
        tell_time = time.perf_counter() - tell_start_time

        # Checkpointing
        if cfg.checkpoint_interval > 0:
            envs_meta_data = extract_normalization_params(env=envs)
            checkpointer.save(
                policy_params=params,
                rewards=avg_fitness,
                meta_data=envs_meta_data,
            )

        if cfg.wandb:
            strategy_metrics = to_state_dict(strategy_state)
            strategy_metrics = {
                f"strategy/{k}": v for k, v in strategy_metrics.items()
            }
            generation_log.update(strategy_metrics)
            generation_log.update(
                {
                    "population/avg_fitness": np.mean(total_fitness, axis=1),
                    "population/avg_length": np.mean(total_lengths, axis=1),
                    "population/min_fitness": np.min(total_fitness, axis=1),
                    "population/min_length": np.min(total_lengths, axis=1),
                    "population/max_fitness": np.max(total_fitness, axis=1),
                    "population/max_length": np.max(total_lengths, axis=1),
                    "population/total_fitness": total_fitness,
                    "evolution/generation": gen,
                    "evolution/fitness_min": avg_fitness.min(),
                    "evolution/fitness_max": avg_fitness.max(),
                    "evolution/fitness_mean": avg_fitness.mean(),
                    "evolution/max_opt_param": opt_params.max(),
                    "evolution/min_opt_param": opt_params.min(),
                    "evolution/mean_opt_param": opt_params.mean(),
                    "evolution/std_opt_param": opt_params.std(),
                    "time/ask_time": ask_time,
                    "time/tell_time": tell_time,
                    "time/global_timesteps": global_time_steps,
                }
            )
            wandb.log(generation_log)

        if cfg.verbose:
            print(
                f"Generation: {gen}: {avg_fitness.min():.2f} / "
                f"{avg_fitness.mean():.2f} / {avg_fitness.max():.2f}"
            )

    envs.close()

    if cfg.checkpoint_interval > 0:
        checkpointer.close()

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
