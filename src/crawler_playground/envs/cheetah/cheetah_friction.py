# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cheetah environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cheetah.xml"
# Running speed above which reward is 1.
_RUN_SPEED = 10
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        wait_duration=[300, 700],
        # ground friction
        min_friction=0.4,
        max_friction=0.95,
    )


class Run(mjx_env.MjxEnv):
    """Cheetah running environment."""

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

        self._xml_path = _XML_PATH.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), common.get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self) -> None:
        self._lowers = self._mj_model.jnt_range[3:, 0]
        self._uppers = self._mj_model.jnt_range[3:, 1]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[3:].set(
            jax.random.uniform(
                rng1,
                (self.mjx_model.nq - 3,),
                minval=self._lowers,
                maxval=self._uppers,
            )
        )

        data = mjx_env.make_data(self.mjx_model, qpos=qpos)

        # Stabilize.
        data = mjx_env.step(
            self.mjx_model, data, jp.zeros(self.mjx_model.nu), 200
        )
        data = data.replace(time=0.0)

        # Adaptation trigger.
        steps_until_next_adaptation = jax.random.randint(
            rng2,
            shape=(1,),
            minval=self._config.wait_duration[0],
            maxval=self._config.wait_duration[1],
        )

        metrics = {}
        info = {
            "rng": rng,
            "steps_until_next_adaptation": steps_until_next_adaptation,
            "motor_failure_mask": jp.ones(self.mjx_model.nu),
        }

        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(
            self.mjx_model, state.data, action, self.n_substeps
        )
        reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, state.info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        state.info["steps_until_next_adaptation"] -= 1
        state = jax.lax.cond(
            state.info["steps_until_next_adaptation"][0] == 0,
            lambda: self._maybe_change_ground_friction(state),
            lambda: state,
        )

        return mjx_env.State(
            data, obs, reward, done, state.metrics, state.info
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        return jp.concatenate(
            [
                data.qpos[1:],
                data.qvel,
            ]
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        del action, info, metrics  # Unused.
        speed = mjx_env.get_sensor_data(
            self.mj_model, data, "torso_subtreelinvel"
        )[0]  # x-axis only.
        return reward.tolerance(
            speed,
            bounds=(_RUN_SPEED, float("inf")),
            margin=_RUN_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )

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

    def _maybe_change_ground_friction(
        self,
        state: mjx_env.State,
    ) -> mjx.Model:
        state.info["rng"], friction_rng = jax.random.split(state.info["rng"])

        self._mjx_model_v, self._in_axes = randomize_friction(
            self.mjx_model,
            self._config.min_friction,
            self._config.max_friction,
        )

        def reset(mjx_model: mjx.Model, rng):
            env = self.env
            env.unwrapped._mjx_model = mjx_model
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(
            self._mjx_model_v, friction_rng
        )

        return state


def randomize_friction(
    model: mjx.Model, rng: jax.Array, minval: float = 0.4, maxval: float = 0.95
) -> tuple[mjx.Model, jax.Array]:
    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=minval, maxval=maxval)
        )

        return (geom_friction,)

    friction = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"geom_friction": 0})
    model = model.tree_replace({"geom_friction": friction})

    return model, in_axes
