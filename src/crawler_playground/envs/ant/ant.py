"""Ant environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

ROOT_PATH = epath.Path(__file__).parent
_XML_PATH = ROOT_PATH / "xmls" / "ant.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        reward_config=config_dict.create(
            healthy_reward_weight=1.0,
            forward_reward_weight=1.0,
            control_cost_weight=0.5,
        ),
    )


class Run(mjx_env.MjxEnv):
    """Ant running environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[
            Dict[str, Union[str, int, list[Any]]]
        ] = None,
    ):
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )

        self._get_reward = self._get_forward_reward

        self._xml_path = _XML_PATH.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), common.get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self) -> None:
        self._torso_id = self.mj_model.body("torso").id
        self._init_qpos = jp.array(self.mj_model.qpos0).copy()
        self._init_qvel = jp.zeros((self.mj_model.nv,)).copy()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        noise_low = -self._config.reset_noise_scale
        noise_high = self._config.reset_noise_scale

        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos_noise = jax.random.uniform(
            rng1, (self.mjx_model.nq,), minval=noise_low, maxval=noise_high
        )
        qvel_noise = self._config.reset_noise_scale * jax.random.normal(
            rng2, (self.mjx_model.nv,)
        )

        qpos = self._init_qpos + qpos_noise
        qvel = self._init_qvel + qvel_noise

        data = mjx_env.make_data(self.mjx_model, qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "reward/reward_forward": jp.zeros(()),
            "reward/reward_ctrl": jp.zeros(()),
            "reward/reward_survive": jp.zeros(()),
        }
        info = {"rng": rng}

        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        xy_position_before = state.data.xpos[self._torso_id][:2].copy()
        data = mjx_env.step(
            self.mjx_model, state.data, action, self.n_substeps
        )
        xy_position_after = data.xpos[self._torso_id][:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity = xy_velocity[0]

        # Termination condition
        done = self._get_termination(data)

        # Compute rewards
        forward_reward = self._get_forward_reward(x_velocity)
        ctrl_cost = self._get_control_cost(action)
        healthy_reward = self._get_healthy_reward(done)

        reward = healthy_reward + forward_reward - ctrl_cost

        state.metrics["reward/reward_forward"] = forward_reward
        state.metrics["reward/reward_ctrl"] = ctrl_cost
        state.metrics["reward/reward_survive"] = healthy_reward

        # Compute observation
        obs = self._get_obs(data, state.info)

        return mjx_env.State(
            data, obs, reward, done, state.metrics, state.info
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        jnt_angles = data.qpos
        jnt_velocities = data.qvel

        if self._config.exclude_current_positions_from_observation:
            jnt_angles = jnt_angles[2:]  # Exclude x- and y-positions.
        return jp.concatenate((jnt_angles, jnt_velocities))

    def _get_forward_reward(self, x_velocity: jax.Array) -> jax.Array:
        forward_weight = self._config.reward_config.forward_reward_weight
        return forward_weight * x_velocity

    def _get_control_cost(self, action: jax.Array) -> jax.Array:
        """Control cost."""
        ctrl_cost = jp.sum(jp.square(action))
        ctrl_cost_weight = self._config.reward_config.control_cost_weight
        return ctrl_cost_weight * ctrl_cost

    def _get_healthy_reward(self, done: jax.Array) -> jax.Array:
        """Healthy reward."""
        done_reward = 1.0 - done.astype(jp.float32)
        healthy_weight = self._config.reward_config.healthy_reward_weight
        return healthy_weight * done_reward

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """Termination condition."""
        height = data.qpos[2]
        height_unhealthy = jp.logical_or(height <= 0.25, height >= 1.0)
        sim_unhealthy = jp.logical_or(
            jp.isnan(data.qpos).any(), jp.isnan(data.qvel).any()
        )
        done = jp.logical_or(height_unhealthy, sim_unhealthy)
        return done.astype(jp.float32)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
