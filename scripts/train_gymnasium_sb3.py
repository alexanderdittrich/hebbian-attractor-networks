import os
from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)
from wandb.integration.sb3 import WandbCallback


@hydra.main(
    config_path="../config", config_name="gymnasium_sb3", version_base="1.3"
)
def main(cfg: DictConfig):
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    run_id = f"{time_stamp}-{cfg.env.name[:-3].lower()}-{cfg.algorithm.name}-{cfg.seed}"

    run = wandb.init(
        project=f"{cfg.env.name[:-3].lower()}_{cfg.wandb.project}",
        name=run_id,
        id=run_id,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg),
        sync_tensorboard=True,
        save_code=True,
    )

    # Create checkpoint directory path
    ckpt_directory = getattr(cfg, "ckpt_directory", "checkpoints")
    log_dir = os.path.join(ckpt_directory, f"{run_id}/runs/")
    model_save_path = os.path.join(ckpt_directory, f"{run_id}/models/")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    vec_env_cls = (
        SubprocVecEnv if cfg.env.vec_type == "subproc" else DummyVecEnv
    )
    vec_env = make_vec_env(
        cfg.env.name,
        n_envs=cfg.env.num_envs,
        seed=cfg.seed,
        vec_env_cls=vec_env_cls,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # Policy kwargs for network architecture
    policy_kwargs = {}
    if hasattr(cfg.policy, "net_arch"):
        net_arch = dict(cfg.policy.net_arch)
        print(net_arch, type(net_arch))
        policy_kwargs["net_arch"] = net_arch

    model = PPO(
        cfg.policy.type,
        vec_env,
        verbose=1,
        learning_rate=cfg.algorithm.get("learning_rate", 3e-4),
        tensorboard_log=log_dir,
        seed=cfg.seed,
        device="cpu",
        max_grad_norm=cfg.algorithm.get("max_grad_norm", 0.5),
        ent_coef=cfg.algorithm.get("ent_coef", 0.0),
        gamma=cfg.algorithm.get("gamma", 0.99),
        batch_size=cfg.algorithm.get("batch_size", 64),
        gae_lambda=cfg.algorithm.get("gae_lambda", 0.95),
        n_epochs=cfg.algorithm.get("n_epochs", 10),
        n_steps=cfg.algorithm.get("n_steps", 2048),
        policy_kwargs=policy_kwargs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=4000,  # every 4000 x env steps
        save_path=model_save_path,  # where to store checkpoints
        name_prefix="ppo_model",  # prefix for filenames
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=0,
    )

    callbacks = CallbackList(
        [
            WandbCallback(
                gradient_save_freq=0,
                model_save_path=None,
                verbose=0,
            ),
            checkpoint_callback,
        ]
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    run.finish()


if __name__ == "__main__":
    main()
