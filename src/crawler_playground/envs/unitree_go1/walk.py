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
        Kp=50.0,
        Kd=1.0,
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
                linacc=2.5,
            ),
        ),
        reward_config=config_dict.create(
            cmd_x_vel=0.5,
            cmd_y_vel=0.0,
            cmd_yaw_vel=0.0,
            scales=config_dict.create(
                # Base reward.
                # Tracking.
                tracking_global_linvel_x=1.0,
                tracking_global_linvel_y=0.1,
                tracking_yaw=0.2,
                tracking_lin_vel=0.5,
                tracking_ang_vel=0.0,
                # Base reward.
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                # Other.
                dof_pos_limits=-1.0,
                pose=0.5,
                # Other.
                termination=-1.0,
                stand_still=-1.0,
                # Action Regularization.
                torques=-0.0002,
                square_torques=-0.0001,
                poly_torques=-1.0,
                action_rate=-0.01,
                energy=-0.001,
                # Ground reaction forces.
                cfrc_ext=-0.0,
                # Feet.
                feet_clearance=-2.0,
                feet_height=-0.2,
                feet_slip=-0.1,
                feet_air_time=0.1,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        impl="jax",
        nconmax=4 * 8192,
        njmax=40,
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
        dx = jax.random.uniform(key, minval=-0.2, maxval=0.2)
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
        cmd = jp.array(
            [
                self._config.reward_config.cmd_x_vel,
                self._config.reward_config.cmd_y_vel,
                self._config.reward_config.cmd_yaw_vel,
            ]
        )
        info = {
            "rng": rng,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            metrics[f"metric/{k}"] = jp.zeros(())

            if v != 0.0:
                metrics[f"reward/{k}"] = jp.zeros(())

        metrics["swing_peak"] = jp.zeros(())

        obs = self._get_obs(data, info)
        # metrics = self._get_metrics(data, info)

        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # <---------------- Simulator step ---------------->
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # <---------------- Contact data ------------------->
        contact = jp.array(
            [
                data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
                for sensorid in self._feet_floor_found_sensor
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        # <---------------- Reward evaluation ---------------->
        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
            first_contact,
            contact,
        )

        # Define performance metrics
        for k, v in rewards.items():
            state.metrics[f"metric/{k}"] = v

        # Multiply with weighting and consider timesteps
        rewards = {
            k: v * self._config.reward_config.scales.get(k, 0.0)
            for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt

        for k, v in rewards.items():
            scale = self._config.reward_config.scales.get(k, 0.0)
            if scale != 0.0:
                state.metrics[f"reward/{k}"] = v

        # <---------------- Locomotion metrics ---------------->
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)

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

        linacc = self.get_accelerometer(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linacc = (
            linacc
            + (2 * jax.random.uniform(noise_rng, shape=linacc.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linacc
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
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            # Reward forward motions.
            "tracking_global_linvel_x": self._reward_global_vel_x(
                info["command"], self.get_global_linvel(data)
            ),
            "tracking_global_linvel_y": self._cost_global_vel_y(
                info["command"], self.get_global_linvel(data)
            ),
            "tracking_yaw": self._reward_local_yaw(
                info["command"], self.get_yaw(data)
            ),
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], self.get_gyro(data)
            ),
            # "local_yaw": self._reward_local_yaw(self.get_yaw(data)),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "stand_still": self._cost_stand_still(
                info["command"], data.qpos[7:]
            ),
            "termination": self._cost_termination(done),
            "pose": self._reward_pose(data.qpos[7:]),
            "torques": self._cost_torques(data.actuator_force),
            "square_torques": self._cost_square_torques(data.actuator_force),
            "poly_torques": self._cost_poly_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_clearance": self._cost_feet_clearance(data),
            "feet_height": self._cost_feet_height(
                info["swing_peak"], first_contact, info
            ),
            "feet_air_time": self._reward_feet_air_time(
                info["feet_air_time"], first_contact, info["command"]
            ),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "cfrc_ext": self._cost_cfrc_ext(data),
        }

    def _get_metrics(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> dict[str, jax.Array]:
        return {
            "global_linvel_x": self.get_global_linvel(data)[0],
            "global_linvel_y": self.get_global_linvel(data)[1],
            "global_linvel_z": self.get_global_linvel(data)[2],
            "global_angvel_x": self.get_global_angvel(data)[0],
            "global_angvel_y": self.get_global_angvel(data)[1],
            "global_angvel_z": self.get_global_angvel(data)[2],
            "gravity_x": self.get_gravity(data)[0],
            "gravity_y": self.get_gravity(data)[1],
            "gravity_z": self.get_gravity(data)[2],
            "local_linvel_x": self.get_local_linvel(data)[0],
            "local_linvel_y": self.get_local_linvel(data)[1],
            "local_linvel_z": self.get_local_linvel(data)[2],
            "local_angvel_x": self.get_gyro(data)[0],
            "local_angvel_y": self.get_gyro(data)[1],
            "local_angvel_z": self.get_gyro(data)[2],
            "feet_air_time_fr": info["feet_air_time"][0],
            "feet_air_time_fl": info["feet_air_time"][1],
            "feet_air_time_rr": info["feet_air_time"][2],
            "feet_air_time_rl": info["feet_air_time"][3],
            "swing_peak_fr": info["swing_peak"][0],
            "swing_peak_fl": info["swing_peak"][1],
            "swing_peak_rr": info["swing_peak"][2],
            "swing_peak_rl": info["swing_peak"][3],
            "torques_fr_hip": data.actuator_force[0],
            "torques_fr_thigh": data.actuator_force[1],
            "torques_fr_calf": data.actuator_force[2],
            "torques_fl_hip": data.actuator_force[3],
            "torques_fl_thigh": data.actuator_force[4],
            "torques_fl_calf": data.actuator_force[5],
            "torques_rr_hip": data.actuator_force[6],
            "torques_rr_thigh": data.actuator_force[7],
            "torques_rr_calf": data.actuator_force[8],
            "torques_rl_hip": data.actuator_force[9],
            "torques_rl_thigh": data.actuator_force[10],
            "torques_rl_calf": data.actuator_force[11],
            "jnt_pos_fr_hip": data.qpos[7],
            "jnt_pos_fr_thigh": data.qpos[8],
            "jnt_pos_fr_calf": data.qpos[9],
            "jnt_pos_fl_hip": data.qpos[10],
            "jnt_pos_fl_thigh": data.qpos[11],
            "jnt_pos_fl_calf": data.qpos[12],
            "jnt_pos_rr_hip": data.qpos[13],
            "jnt_pos_rr_thigh": data.qpos[14],
            "jnt_pos_rr_calf": data.qpos[15],
            "jnt_pos_rl_hip": data.qpos[16],
            "jnt_pos_rl_thigh": data.qpos[17],
            "jnt_pos_rl_calf": data.qpos[18],
        }

    # Sensor functions

    def get_yaw(self, data: mjx.Data) -> jax.Array:
        quat = self.get_orientation(data)
        # JAX SciPy expects [x, y, z, w] quaternion ordering
        quat_xyzw = quat[jp.array([1, 2, 3, 0])]
        euler_xyz = jsp.Rotation.from_quat(quat_xyzw).as_euler("xyz")
        yaw = euler_xyz[2]
        return yaw

    # Custom rewards.

    def _reward_global_vel_x(
        self, commands: jax.Array, global_vel: jax.Array
    ) -> jax.Array:
        lin_vel_error = jp.square(global_vel[0] - commands[0])
        # scales gaussian so that R(0) = 0 and R(v>0) > 0
        scaling = (commands[0] ** 2) / (-2 * np.log(5e-2))
        return jp.exp(-lin_vel_error / scaling)

    def _cost_global_vel_y(
        self, commands: jax.Array, global_vel: jax.Array
    ) -> jax.Array:
        # Penalize global velocity in the Y direction.
        return jp.square(global_vel[1] - commands[1])

    def _cost_local_yaw(
        self,
        commands: jax.Array,
        yaw: jax.Array,
    ) -> jax.Array:
        # Penalize yaw velocity.
        # Tracking of angular velocity commands (yaw).
        return jp.square(commands[2] - yaw)

    def _reward_local_yaw(
        self,
        commands: jax.Array,
        yaw: jax.Array,
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw).
        ang_vel_error = jp.square(commands[2] - yaw)
        return jp.exp(-ang_vel_error / 2.0)

    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        local_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes).
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(
            -lin_vel_error / self._config.reward_config.tracking_sigma
        )

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw).
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(
            -ang_vel_error / self._config.reward_config.tracking_sigma
        )

    # Base-related rewards.

    def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
        # Penalize z axis base linear velocity.
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
        # Penalize xy axes base angular velocity.
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        # Penalize non flat base orientation.
        return jp.sum(jp.square(torso_zaxis[:2]))

    # Energy related rewards.

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques.
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_square_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques.
        return jp.sum(jp.square(torques))

    def _cost_poly_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques.
        power = 4
        torque_limit = 28.44
        return jp.sum(jp.abs(torques) ** power / (torque_limit) ** power)

    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        # Penalize energy consumption.
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))

    # Other rewards.

    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        # Stay close to the default pose.
        weight = jp.array([1.0, 1.0, 0.1] * 4)
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    def _cost_stand_still(
        self,
        commands: jax.Array,
        qpos: jax.Array,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        # Penalize early termination.
        return done.astype(jp.float32)

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        # Penalize joints if they cross soft limits.
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    # Feet related rewards.

    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

    def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(
        self,
        swing_peak: jax.Array,
        first_contact: jax.Array,
        info: dict[str, Any],
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        # Reward air time.
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
        return rew_air_time

    # New custom rewards.
    def _cost_cfrc_ext(self, data: mjx.Data) -> jax.Array:
        # Penalty for high ground reaction forces
        # cfrc_ext contains external contact forces for each body
        # Format: [force_x, force_y, force_z, torque_x, torque_y, torque_z] for each body
        # Only consider force magnitudes (first 3 components of each 6-element body force)
        # Extract forces (x,y,z)
        forces = data._impl.cfrc_ext[:, :3]
        return jp.sum(jp.square(forces))
