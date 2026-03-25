import logging
from time import perf_counter

# import crawler_gym  # noqa: F401
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from jax.random import randint
from numpy import (
    array,
    float32,
    int32,
    logical_or,
    ones,
    sum,
    zeros,
    zeros_like,
)
from tqdm import tqdm

# import tensegrity_gym  # noqa: F401

VectorEnvType = gym.vector.VectorEnv


def generate_vector_env(
    env_id: str,
    num_envs: int,
    multithreading: bool = False,
    **env_kwargs,
):
    vectorization_mode = "sync"
    if multithreading:
        vectorization_mode = "async"

    envs = gym.make_vec(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        **env_kwargs,
    )
    envs = gym.wrappers.vector.FlattenObservation(envs)
    envs = gym.wrappers.vector.ClipAction(envs)
    envs = gym.wrappers.vector.NormalizeObservation(envs)

    return envs


def generate_single_env(
    env_id: str,
    **kwargs,
):
    env = gym.make(env_id, **kwargs)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)

    return env


def extract_normalization_params(env: VectorEnvType) -> dict:
    observation_mean = env.obs_rms.mean.tolist()
    observation_var = env.obs_rms.var.tolist()
    count = env.obs_rms.count

    return {
        "mean": observation_mean,
        "var": observation_var,
        "count": count,
    }


def insert_normalization_params(
    env: VectorEnvType, obs_rms_params: dict
) -> VectorEnvType:
    mean = array(obs_rms_params["mean"])
    var = array(obs_rms_params["var"])
    count = obs_rms_params["count"]

    env.obs_rms.mean = mean
    env.obs_rms.var = var
    env.obs_rms.count = count

    return env


def evaluate(
    random_key,
    # environment
    envs,
    # policy
    forward_fcn,
    update_fcn,
    policy_params,
    # timing params
    episode_steps: int,
    environment_interval: float,
    hebbian_update_interval: float,
    # logging functionality
    progress_bar: bool = False,
    # life-time logging
    record_policy: bool = False,
    record_env_info: bool = False,
    record_stepwise_rewards: bool = False,
    # record_sensordata: bool = False,
    ignore_terminations: bool = False,
):
    num_envs = 1
    if hasattr(envs, "num_envs"):
        num_envs = envs.num_envs

    # episode statistics
    active_episode = ones((num_envs,), dtype=int32)
    cumulative_lengths = zeros((num_envs,), dtype=int32)
    cumulative_rewards = zeros((num_envs,), dtype=float32)

    # Update management
    # Network weights are update asynchronously to forward passes.
    # The network's dynamics can be independent of environment flow.
    assert environment_interval > 0.0, (
        "Environment interaction interval must be greater than 0."
    )
    assert hebbian_update_interval > 0.0, (
        "Hebbian update interval must be greater than 0."
    )

    # floating-point operations are not reliable for tracking the frequencies,
    # therefore we transform the intervals relative to `episode_steps` to integers.
    # Episode steps define the number of simulation steps and therefore the world
    # frequency.
    total_time = episode_steps * environment_interval
    total_steps = int(
        total_time / min(environment_interval, hebbian_update_interval)
    )

    total_environment_steps = int(total_time / environment_interval)
    total_update_steps = int(total_time / hebbian_update_interval)

    relative_environment_steps = int(total_steps / total_environment_steps)
    relative_update_steps = int(total_steps / total_update_steps)

    # debug step computation
    step_compute = {
        "episode_steps": episode_steps,
        "environment_interval": environment_interval,
        "hebbian_update_interval": hebbian_update_interval,
        "total_time": total_time,
        "total_steps": total_steps,
        "total_environment_steps": total_environment_steps,
        "total_update_steps": total_update_steps,
        "relative_environment_steps": relative_environment_steps,
        "relative_update_steps": relative_update_steps,
    }
    logging.debug(step_compute)

    # initialize progress bar
    if progress_bar:
        pbar = tqdm(
            range(total_steps),
            ascii=True,
            desc="Episode steps",
            leave=False,
            position=1,
        )

    # computation time keeping
    inference_duration = 0.0
    simulation_duration = 0.0
    update_duration = 0.0

    # start environment evaluation
    seeds = randint(random_key, (num_envs,), 0, 2e15).tolist()
    if num_envs == 1:
        seeds = seeds[0]

    observations, infos = envs.reset(seed=seeds)

    # log policy parameters
    if record_policy:
        policy_params_history = [policy_params]

    # log environment information
    if record_env_info:
        environment_metrics = [infos]

    if record_stepwise_rewards:
        stepwise_rewards = []

    # if record_sensordata:
    #     sensor_data_history = []

    # environment loop
    for time_step in range(total_steps):
        # environment and forward step
        if time_step % relative_environment_steps == 0:
            # <--- forward pass --->
            start_inference_duration = perf_counter()
            policy_params, actions = forward_fcn(policy_params, observations)
            inference_duration += perf_counter() - start_inference_duration

            # <--- environment step --->
            start_simulation_duration = perf_counter()
            (
                observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = envs.step(actions)
            simulation_duration += perf_counter() - start_simulation_duration

            if ignore_terminations:
                dones = zeros_like(terminations)
            else:
                dones = logical_or(terminations, truncations)

            if record_stepwise_rewards:
                stepwise_rewards.append(rewards)

            cumulative_lengths += active_episode
            cumulative_rewards += rewards * active_episode
            active_episode *= 1 - dones

        # <--- update step --->
        if time_step % relative_update_steps == 0:
            start_update_duration = perf_counter()
            policy_params = update_fcn(policy_params, rewards)
            update_duration += perf_counter() - start_update_duration

        # <--- logging --->
        if record_policy:
            policy_params_history.append(policy_params)

        if record_env_info:
            environment_metrics.append(infos)

        if progress_bar:
            pbar.update(1)

        if not ignore_terminations and sum(active_episode) == 0:
            break

    # policy: just log population parameters here. Individual metrics, like
    # partial rewards can be logged by evaluating specific populations.
    evaluation_metrics = {
        # Step statistics
        "cumulative_lengths_max": cumulative_lengths.max(),
        "cumulative_lengths_mean": cumulative_lengths.mean(),
        "cumulative_lengths_min": cumulative_lengths.min(),
        # Reward statistics
        "cumulative_rewards_max": cumulative_rewards.max(),
        "cumulative_rewards_mean": cumulative_rewards.mean(),
        "cumulative_rewards_min": cumulative_rewards.min(),
        # Time statistics
        "episode_inference_duration": inference_duration,
        "episode_simulation_duration": simulation_duration,
        "episode_update_duration": update_duration,
        "episode_avg_inference_duration": inference_duration
        / total_environment_steps,
        "episode_avg_simulation_duration": simulation_duration
        / total_environment_steps,
        "episode_avg_update_duration": update_duration / total_update_steps,
    }

    # Generate function output
    out = (
        cumulative_rewards,
        cumulative_lengths,
        evaluation_metrics,
    )

    if record_policy:
        out += (policy_params_history,)

    if record_env_info:
        out += (environment_metrics,)

    if record_stepwise_rewards:
        out += (stepwise_rewards,)

    return out
