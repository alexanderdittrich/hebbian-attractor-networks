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
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import jax.scipy.spatial.transform as jsp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

from crawler_playground.envs.unitree_go1 import go1_base
from crawler_playground.envs.unitree_go1 import go1_constants as consts

FLOOR_GEOM_ID = 0


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            target_x_vel=2.0,
            scales=config_dict.create(
                global_vel_x=1.0,
                local_yaw=0.2,
                # Regularization.
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        pert_config=config_dict.create(
            wait_duration=[450, 550],
            exclude_RL_leg=True,
        ),
    )


class Walk(go1_base.Go1Env):
    """Recover from a fall and stand up.

    Observation space:
        - Gyroscope readings (3)
        - Gravity vector (3)
        - Joint angles (12)
        - Last action (12)

    Action space: Joint angles (12) scaled by a factor and added to the current
    joint angles. We tried using the same action space used in the joystick task
    where the output of the policy is added to the nominal "home" pose but it
    didn't work as well as adding to the current joint configuration. I suspect
    this is because the latter gives the policy a wider initial range of motion.

    Reward function:
        - Orientation: The torso should be upright.
        - Torso height: The torso should be at a desired height. This is to
            prevent the robot from flipping over and just lying on the ground.
        - Posture: The robot should be in the neural pose. This reward is only
            given when the robot is upright and at the desired height.
        - Stand still: Policy outputs should be zero once the robot is upright
            and at the desired height. This minimizes jittering.
        The next two rewards aren't really needed but promote better sim2real
            transfer (in theory):
        - Torques: Minimize joint torques.
        - Action rate: Minimize the first and second derivative of actions.
    """

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[
            Dict[str, Union[str, int, list[Any]]]
        ] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
        self._damage_pose = jp.array(
            [
                -0.352275,
                1.18554,
                -2.80738,
                0.360892,
                1.1806,
                -2.80281,
                -0.381197,
                1.16812,
                -2.79123,
                0.391054,
                1.1622,
                -2.78576,
            ]
        )

        # Note: First joint is freejoint.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = (
            self._lowers * self._config.soft_joint_pos_limit_factor
        )
        self._soft_uppers = (
            self._uppers * self._config.soft_joint_pos_limit_factor
        )

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dx = jax.random.uniform(key, minval=-5.0, maxval=5.0)
        qpos = qpos.at[0].set(qpos[0] + dx)

        rng, key = jax.random.split(rng)
        dy = jax.random.uniform(key, minval=-0.2, maxval=0.2)
        qpos = qpos.at[1].set(qpos[1] + dy)

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14 / 6, maxval=3.14 / 6)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2)
        )

        data = mjx_env.make_data(
            self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:]
        )

        # Target velocity commands.
        target_vel = self._config.reward_config.target_x_vel

        # Adaptation experiments.
        rng, key = jax.random.split(rng)
        steps_until_next_adaptation = jax.random.randint(
            key,
            (),
            minval=self._config.pert_config.wait_duration[0],
            maxval=self._config.pert_config.wait_duration[1],
        )

        info = {
            "rng": rng,
            "target_vel": target_vel,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "swing_peak": jp.zeros(4),
            "leg_damage_type": 4,
            "steps_until_next_adaptation": steps_until_next_adaptation,
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # <---------------- Simulator step ---------------->
        motor_targets = self._compute_motor_targets(state, action)
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # <---------------- Evaluate change ---------------->
        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
        )
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)

        # <---------------- Task adaptation ---------------->
        state.info["steps_until_next_adaptation"] -= 1
        # Motor failure adaptation.
        state = jax.lax.cond(
            state.info["steps_until_next_adaptation"] == 0,
            lambda: self._maybe_switch_off_actuators(state),
            lambda: state,
        )

        # <---------------- Locomotion metrics ---------------->
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_upvector(data)[-1] < 0.0
        return fall_termination

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = self.get_gravity(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_global_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )

        obs = jp.hstack(
            [
                noisy_linvel,  # 3
                noisy_gyro,  # 3
                noisy_gravity,  # 3
                noisy_joint_angles - self._default_pose,  # 12
                noisy_joint_vel,  # 12
            ]
        )

        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            "global_vel_x": self._reward_global_vel_x(
                info["target_vel"], self.get_global_linvel(data)
            ),
            "local_yaw": self._reward_local_yaw(self.get_yaw(data)),
        }

    # Custom rewards.

    def _reward_global_vel_x(
        self, target_vel: jax.Array, global_vel: jax.Array
    ) -> jax.Array:
        lin_vel_error = jp.square(global_vel[0] - target_vel)
        # scales gaussian so that R(0) = 0 and R(v>0) > 0
        scaling = (target_vel**2) / (-2 * np.log(5e-2))
        return jp.exp(-lin_vel_error / scaling)

    def _reward_local_yaw(
        self,
        yaw: jax.Array,
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw).
        ang_vel_error = jp.square(yaw)
        return jp.exp(-ang_vel_error / 2.0)

    def get_yaw(self, data: mjx.Data) -> jax.Array:
        quat = self.get_orientation(data)
        # JAX SciPy expects [x, y, z, w] quaternion ordering
        quat_xyzw = quat[jp.array([1, 2, 3, 0])]
        euler_xyz = jsp.Rotation.from_quat(quat_xyzw).as_euler("xyz")
        yaw = euler_xyz[2]
        return yaw

    def _maybe_switch_off_actuators(
        self, state: mjx_env.State
    ) -> mjx_env.State:
        rng, permute_rng = jax.random.split(state.info["rng"])

        potential_leg_damages = 4
        if self._config.pert_config.exclude_RL_leg:
            # Exclude RL leg motors from being switched off.
            potential_leg_damages = 3

        leg_damage_type = jax.random.randint(
            key=permute_rng,
            shape=(),
            minval=0,
            maxval=potential_leg_damages,
        )

        new_info = state.info.copy()
        new_info["leg_damage_type"] = leg_damage_type
        new_info["rng"] = rng

        return state.replace(info=new_info)

    def _compute_motor_targets(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ):
        new_motor_targets = (
            self._default_pose + action * self._config.action_scale
        )

        masks = jp.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0 -> FR damaged
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # 1 -> FL damaged
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],  # 2 -> RR damaged
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 3 -> RL damaged
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 4 -> no damange
            ]
        )

        motor_targets = jp.where(
            masks[state.info["leg_damage_type"]] == 1.0,
            new_motor_targets,
            self._damage_pose,
        )

        return motor_targets
