from pathlib import Path
from typing import Dict, Optional, Union

import jax
from gymnasium.wrappers import RecordVideo

from kandel.sim_api.gymnasium import evaluate
from kandel.utility.checkpoint_loader import load_environment, load_policy


def video_rendering(
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    video_folder: Union[str, Path] = "/tmp/",
    name_prefix: str = "video",
    eval_fcn: callable = evaluate,
    video_resolution: tuple = (640, 480),
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    episode_length: int = 1000,
    seed: int = 0,
    verbose: bool = True,
    default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
    **env_kwargs,
):
    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        render_mode="rgb_array",
        width=video_resolution[0],
        height=video_resolution[1],
        camera_id=camera_id,
        camera_name=camera_name,
        default_camera_config=default_camera_config,
        **env_kwargs,
    )

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: True,
    )

    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    reward, length, _ = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
    )

    if verbose:
        print(f"Reward: {reward}, Length: {length}")

    env.close()


def viewer_rendering(
    checkpoint_directory: Union[str, Path],
    checkpoint: int,
    policy_id: int = 0,
    episode_length: int = 1000,
    verbose: bool = True,
    eval_fcn: callable = evaluate,
    environment_id: Optional[str] = None,
    seed: int = 0,
    viewer_resolution: tuple = (640, 480),
    **env_kwargs,
):
    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        render_mode="human",
        width=viewer_resolution[0],
        height=viewer_resolution[1],
        environment_id=environment_id,
        episode_length=episode_length,
        **env_kwargs,
    )

    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    reward, length, _ = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
    )

    if verbose:
        print(f"Reward: {reward}, Length: {length}")

    env.close()
