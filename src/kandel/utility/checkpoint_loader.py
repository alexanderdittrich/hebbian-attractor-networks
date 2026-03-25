import json
from pathlib import Path
from typing import Optional, Union

import jax.numpy as jnp
from jax.random import key
from ml_collections import config_dict
from mujoco_playground import registry
from omegaconf import OmegaConf
from orbax.checkpoint import StandardCheckpointer

import crawler_playground
from kandel.model import networks
from kandel.sim_api.gymnasium import (
    generate_single_env,
    insert_normalization_params,
)
from kandel.utility.checkpoint_manager import (
    ENSEMBLE_NAME,
    EPISODE_NAME,
    INDIVIDUAL_NAME,
)
from kandel.utility.running_statistics import init_state


def fetch_config(path: Union[str, Path]):
    return OmegaConf.load(path)


def generate_dummy_policy(cfg: dict):
    policy = networks[cfg.policy_id](
        input_dim=cfg.observation_dim,
        output_dim=cfg.action_dim,
        **cfg.policy,
    )
    dummy_policy_params = policy.initialize(key(0))
    return policy, dummy_policy_params


def load_policy(
    checkpoint_directory: Union[str, Path],
    checkpoint: int,
    policy_id: Optional[int] = None,
):
    # determine policy directories
    checkpoint_directory = Path(checkpoint_directory)
    config_path = checkpoint_directory / "metadata.yaml"

    if isinstance(policy_id, int):
        policy_id = f"{INDIVIDUAL_NAME}_{policy_id}"
    else:
        policy_id = ENSEMBLE_NAME

    policy_path = (
        checkpoint_directory / f"{EPISODE_NAME}_{checkpoint}" / policy_id
    )

    cfg = fetch_config(config_path)
    policy, dummy_policy_params = generate_dummy_policy(cfg)

    checkpointer = StandardCheckpointer()
    policy_params = checkpointer.restore(policy_path, dummy_policy_params)

    return policy, policy_params


def load_environment(
    checkpoint_directory: Union[str, Path],
    checkpoint: int,
    environment_id: Optional[str] = None,
    update_running_mean=False,
    episode_length: int = 1000,
    **kwargs,
):
    checkpoint_directory = Path(checkpoint_directory)
    config_path = checkpoint_directory / "metadata.yaml"
    cfg = fetch_config(config_path)

    save_path = (
        checkpoint_directory
        / f"{EPISODE_NAME}_{checkpoint}"
        / "checkpoint_metadata.json"
    )

    with open(save_path, "r") as f:
        meta_data = json.load(f)

    if "environments" in cfg:
        cfg_env_id = cfg.environments[0]
    else:
        cfg_env_id = cfg.environment_id

    if environment_id is None:
        environment_id = cfg_env_id

    if cfg_env_id in meta_data:
        norm_params = meta_data[cfg_env_id]
    else:
        norm_params = meta_data

    if cfg_env_id in meta_data:
        env_params = cfg[cfg_env_id]
    else:
        env_params = cfg["gymnasium"]

    env = generate_single_env(
        env_id=environment_id,
        max_episode_steps=episode_length,
        **env_params,
        **kwargs,
    )
    env = insert_normalization_params(env, norm_params)
    env.update_running_mean = update_running_mean

    info = {
        "step_interval": cfg.step_interval,
        "update_interval": cfg.update_interval,
    }

    return env, info


def load_playground_environment(
    checkpoint_directory: Union[str, Path],
    checkpoint: int,
):
    checkpoint_directory = Path(checkpoint_directory)
    config_path = checkpoint_directory / "metadata.yaml"
    cfg = fetch_config(config_path)

    save_path = (
        checkpoint_directory
        / f"{EPISODE_NAME}_{checkpoint}"
        / "checkpoint_metadata.json"
    )

    with open(save_path, "r") as f:
        meta_data = json.load(f)

    raw = OmegaConf.to_container(cfg.playground, resolve=True)
    overrides_cfg = config_dict.create(**raw)
    env = registry.load(
        cfg.environment_id,
        config_overrides=overrides_cfg,
    )

    dummy_obs = jnp.zeros((env.observation_size,))
    norm_state = init_state(dummy_obs)

    norm_state = norm_state.replace(
        mean=jnp.array(meta_data[cfg.environment_id]["mean"]),
        std=jnp.array(meta_data[cfg.environment_id]["std"]),
        summed_variance=jnp.array(
            meta_data[cfg.environment_id]["summed_variance"]
        ),
        count=meta_data[cfg.environment_id]["count"],
    )

    return env, norm_state, cfg
