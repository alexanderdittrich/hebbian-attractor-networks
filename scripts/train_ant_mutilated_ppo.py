# PPO baseline for the Najarro et al. (2020) ant mutilated experiment.
#
# A single PPO policy is trained with domain randomisation over morphological
# variants of the MuJoCo ant. The DR follows the mujoco_playground pattern:
# only the fields that differ between variants (geom geometry + body physics)
# are given a batch axis via tree_replace; all other model fields are shared.
#
# Batch layout inside the brax environment (n_variants=2, num_envs=2048):
#   [V0]*1024  +  [V1]*1024   per-env model assignment
#
# Environment wrapping:
#   BraxDomainRandomizationVmapWrapper  — tiles variant models, vmaps step/reset
#   EpisodeWrapper                      — tracks per-episode step count
#   BraxAutoResetWrapper                — auto-resets done environments
#
# The randomization_fn(mjx_model, rng) signature is brax-compatible:
# brax computes rng = split(key, batch_size) and passes it as a keyword arg.
# We ignore rng content (it carries no physical noise) and use rng.shape[0]
# solely to determine the per-variant batch split.

import datetime
import functools
import logging
import math
import os
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional

import hydra
import jax
import wandb
from brax.training import acting
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from omegaconf import DictConfig, OmegaConf

from crawler_playground.envs.ant.ant import Run as AntRun
from crawler_playground.envs.ant.ant import default_config
from crawler_playground.envs.ant.randomize import domain_randomize

# ── Suppress noisy orbax async-checkpoint warning ─────────────────────────────

logging.getLogger("absl").addFilter(
    lambda r: "_SignalingThread" not in r.getMessage()
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


# ── Environment helpers ────────────────────────────────────────────────────────


def create_base_env(env_overrides: Optional[Dict] = None) -> AntRun:
    """Load the base (undamaged) ant environment."""
    return AntRun(
        config=default_config(),
        config_overrides=env_overrides or {},
    )


def make_randomization_fn(base_env: AntRun, variants: List[str]):
    """Return a brax-compatible randomization_fn(mjx_model, rng).

    Brax calls this as ``functools.partial(fn, rng=rng_batch)(mjx_model)``.
    The batch size is inferred from ``rng.shape[0]``, so the same function
    works for both the training env and the eval env.
    """
    return functools.partial(
        domain_randomize,
        mj_model=base_env.mj_model,
        variants=variants,
    )


# ── Training entry point ───────────────────────────────────────────────────────


@hydra.main(
    config_path="../config", config_name="ant_mutilated_ppo", version_base=None
)
def train(cfg: DictConfig) -> None:
    timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    exp_name = f"ant-mutilated-ppo-{timestamp}"

    # W&B
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.get("wandb_entity", None),
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Base environment (undamaged; variants are applied via randomization_fn)
    env_overrides = (
        OmegaConf.to_container(cfg.env, resolve=True)
        if hasattr(cfg, "env")
        else {}
    )
    base_env = create_base_env(env_overrides)

    train_variants = list(cfg.train_variants)
    n_variants = len(train_variants)

    if n_variants > 1:
        # Domain randomisation over all training variants.
        randomization_fn = make_randomization_fn(base_env, train_variants)
    else:
        # Single-variant training: set the model directly, no DR needed.
        from crawler_playground.envs.ant.randomize import (
            ANKLE_GEOM,
            make_damaged_mjx_model,
        )

        v = train_variants[0]
        if v != "none":
            base_env._mjx_model = make_damaged_mjx_model(
                base_env.mjx_model, base_env.mj_model, ANKLE_GEOM[v]
            )
        randomization_fn = None

    # Validate divisibility before handing off to brax.
    if randomization_fn is not None:
        for name, count in [
            ("num_envs", cfg.ppo.num_envs),
            ("num_eval_envs", cfg.ppo.num_eval_envs),
        ]:
            if count % n_variants != 0:
                raise ValueError(
                    f"ppo.{name}={count} must be divisible by "
                    f"n_variants={n_variants} (train_variants={train_variants})"
                )

    # Network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.ppo.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.ppo.value_hidden_layer_sizes),
    )

    # Checkpointing
    checkpoint_dir = pathlib.Path(cfg.checkpoint_directory) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if cfg.verbose:
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Train variants : {train_variants}")
        print(f"Test variants  : {list(cfg.test_variants)}")

    # ── Test-variant evaluator (unseen morphologies) ───────────────────────────

    test_variants = list(cfg.test_variants)
    _policy_state: Dict[str, Any] = {
        "make_policy": None,
        "params": None,
        "evaluator": None,
    }

    def _make_test_evaluator(make_policy: Callable) -> acting.Evaluator:
        n_test = len(test_variants)
        if cfg.ppo.num_eval_envs % n_test != 0:
            raise ValueError(
                f"ppo.num_eval_envs={cfg.ppo.num_eval_envs} must be divisible "
                f"by len(test_variants)={n_test}"
            )
        action_repeat = cfg.ppo.get("action_repeat", 1)
        test_rand_fn = make_randomization_fn(base_env, test_variants)
        rng = jax.random.PRNGKey(cfg.seed + 42)
        rand_rng = jax.random.split(rng, cfg.ppo.num_eval_envs)
        v_rand_fn = functools.partial(test_rand_fn, rng=rand_rng)
        test_eval_env = wrapper.wrap_for_brax_training(
            base_env,
            episode_length=cfg.env.episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_rand_fn,
        )
        return acting.Evaluator(
            test_eval_env,
            functools.partial(make_policy, deterministic=True),
            num_eval_envs=cfg.ppo.num_eval_envs,
            episode_length=cfg.env.episode_length,
            action_repeat=action_repeat,
            key=jax.random.PRNGKey(cfg.seed + 43),
        )

    def policy_params_fn(
        _num_steps: int, make_policy: Callable, params: Any
    ) -> None:
        """Capture current params; lazily init the test evaluator."""
        _policy_state["make_policy"] = make_policy
        _policy_state["params"] = params
        if _policy_state["evaluator"] is None and test_variants:
            _policy_state["evaluator"] = _make_test_evaluator(make_policy)

    # ── Progress callback (called by brax after each eval) ─────────────────────

    times = [time.monotonic()]

    def progress(num_steps: int, metrics: Dict[str, Any]) -> None:
        times.append(time.monotonic())
        reward = metrics.get("eval/episode_reward", float("nan"))
        sps = metrics.get("training/sps", float("nan"))

        # Evaluate on unseen test variants (params captured by policy_params_fn)
        test_reward = float("nan")
        if (
            _policy_state["params"] is not None
            and _policy_state["evaluator"] is not None
        ):
            test_metrics = _policy_state["evaluator"].run_evaluation(
                _policy_state["params"], {}
            )
            test_reward = float(
                test_metrics.get("eval/episode_reward", float("nan"))
            )

        if cfg.verbose:
            sps_str = "N/A" if num_steps == 0 else f"{sps:,.0f}"
            test_str = (
                f"{test_reward:7.3f}"
                if not math.isnan(test_reward)
                else "    N/A"
            )
            print(
                f"step={num_steps:>12,d}  "
                f"reward={reward:7.3f}  "
                f"test={test_str}  "
                f"sps={sps_str}"
            )
        if cfg.wandb:
            log_metrics = dict(metrics)
            if not math.isnan(test_reward):
                log_metrics["eval/test_episode_reward"] = test_reward
            wandb.log(log_metrics, step=num_steps)

    # Build and run PPO training
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=cfg.ppo.num_timesteps,
        num_envs=cfg.ppo.num_envs,
        num_eval_envs=cfg.ppo.num_eval_envs,
        episode_length=cfg.env.episode_length,
        action_repeat=cfg.ppo.get("action_repeat", 1),
        unroll_length=cfg.ppo.unroll_length,
        batch_size=cfg.ppo.batch_size,
        num_minibatches=cfg.ppo.num_minibatches,
        num_updates_per_batch=cfg.ppo.num_updates_per_batch,
        discounting=cfg.ppo.discounting,
        learning_rate=cfg.ppo.learning_rate,
        entropy_cost=cfg.ppo.entropy_cost,
        reward_scaling=cfg.ppo.reward_scaling,
        clipping_epsilon=cfg.ppo.clipping_epsilon,
        normalize_observations=cfg.ppo.normalize_observations,
        max_grad_norm=cfg.ppo.max_grad_norm,
        learning_rate_schedule=cfg.ppo.get("learning_rate_schedule", None),
        desired_kl=cfg.ppo.get("desired_kl", None),
        num_evals=cfg.ppo.num_evals,
        network_factory=network_factory,
        seed=cfg.seed,
        save_checkpoint_path=str(checkpoint_dir),
        # DR: brax creates rng batches and passes them to randomization_fn
        randomization_fn=randomization_fn,
        # mujoco_playground wrapper: adds VmapWrapper/DomainRandomizationVmapWrapper,
        # EpisodeWrapper, and BraxAutoResetWrapper
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    make_inference_fn, params, _ = train_fn(
        environment=base_env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
    )

    if len(times) > 1:
        print(f"JIT compile time : {times[1] - times[0]:.1f}s")
        print(f"Training time    : {times[-1] - times[1]:.1f}s")

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
