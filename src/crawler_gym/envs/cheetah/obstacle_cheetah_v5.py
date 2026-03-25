import os
from typing import Dict, Optional, Union

import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetah(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah_model.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[
            str, Union[float, int]
        ] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_file = os.path.join(current_dir, "assets", xml_file)

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.default_camera_config = default_camera_config

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = 18 - exclude_current_positions_from_observation
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": 9 - 1 * exclude_current_positions_from_observation,
            "qvel": 9,
        }

        self.generate_arena()
        self.perturbation_flag = False
        self.slope_projection_flag = False
        self.slope = 0.0

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        if self.perturbation_flag == True:
            self.remove_perturbation()

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()[:9]
        velocity = self.data.qvel.flatten()[:9]

        if self.slope_projection_flag:
            x = self.data.qpos[0]
            z = self.data.qpos[1]
            alpha_deg = self.slope
            alpha = np.deg2rad(alpha_deg)

            if x > 45:
                z_proj = z - (x - 45.0) * np.tan(alpha)
                position[1] = z_proj

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale
            * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
        }

    def generate_arena(
        self, obstacle_type: Optional[str] = None, mass=5.0, angle_deg=5.0
    ):
        max_geom: int = 1000
        visual_options: Dict[int, bool] = {}

        mjcf_desc = mjcf.from_path(self.fullpath)

        self.slope_projection_flag = False

        if obstacle_type is None:
            mjcf_desc = add_floor(mjcf_desc=mjcf_desc)

        if obstacle_type == "slant":
            self.slope_projection_flag = True
            self.slope = angle_deg
            mjcf_desc = add_slant(mjcf_desc=mjcf_desc, angle_deg=angle_deg)

        if obstacle_type == "single":
            mjcf_desc = add_floor(mjcf_desc=mjcf_desc)
            mjcf_desc = add_single_pendulum(
                mjcf_desc=mjcf_desc, pos=[150.0, 0.0, 2.0], mass=mass, id=0
            )

        if obstacle_type == "multiple":
            mjcf_desc = add_floor(mjcf_desc=mjcf_desc)

            mjcf_desc = add_single_pendulum(
                mjcf_desc=mjcf_desc, pos=[50.0, 0.0, 2.0], mass=mass, id=0
            )
            mjcf_desc = add_single_pendulum(
                mjcf_desc=mjcf_desc, pos=[52.0, 0.0, 2.0], mass=mass, id=1
            )
            mjcf_desc = add_single_pendulum(
                mjcf_desc=mjcf_desc, pos=[54.0, 0.0, 2.0], mass=mass, id=2
            )
            mjcf_desc = add_single_pendulum(
                mjcf_desc=mjcf_desc, pos=[56.0, 0.0, 2.0], mass=mass, id=3
            )
            mjcf_desc = add_single_pendulum(
                mjcf_desc=mjcf_desc, pos=[58.0, 0.0, 2.0], mass=mass, id=4
            )

        self.model = mujoco.MjModel.from_xml_string(mjcf_desc.to_xml_string())

        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            self.default_camera_config,
            self.width,
            self.height,
            max_geom,
            self.camera_id,
            self.camera_name,
            visual_options,
        )

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    def act_force_perturbation(
        self, additional_force=[-1500.0, 0.0, 0.0], body_id: int = 1
    ):
        self.additional_force = additional_force
        self.perturbation_flag = True
        self.body_id = body_id
        self.data.xfrc_applied[self.body_id] += [
            *self.additional_force,
            0.0,
            0.0,
            0.0,
        ]
        print("Force applied.")

    def remove_perturbation(self):
        self.perturbation_flag = False
        self.data.xfrc_applied[self.body_id] -= [
            *self.additional_force,
            0.0,
            0.0,
            0.0,
        ]
        print("Force removed.")

    def turn_on_wind(self, wind_strength=[500.0, 20.0, 20.0]):
        """Activates wind in the simulation."""
        self.model.opt.wind = np.array(wind_strength)

    def turn_off_wind(self):
        """Deactivates wind in the simulation."""
        self.model.opt.wind = np.array([0.0, 0.0, 0.0])


def add_single_pendulum(
    mjcf_desc, pos=[40.0, 0.0, 2.0], mass=50.0, pendulum_length=1.35, id=0
):
    PENDULUM_X = pos[0]
    PENDULUM_Z = pos[2]
    PENDULUM_L = pendulum_length
    PENDULUM_MASS = mass

    obstacle = mjcf_desc.worldbody.add(
        "body",
        name=f"obstacle_attachment_{id}",
        pos=[PENDULUM_X, 0, PENDULUM_Z],
    )

    obstacle.add(
        "geom",
        name=f"attachment_point_{id}",
        type="sphere",
        pos=[0, 0, 0],
        size=[0.05, 0.05, 0.05],
    )

    obstacle.add(
        "site",
        name=f"attachment_site_{id}",
        pos=[0, 0, 0],
    )

    obstacle_mass = mjcf_desc.worldbody.add(
        "body",
        name=f"obstacle_body_{id}",
        pos=[PENDULUM_X, 0, PENDULUM_Z - PENDULUM_L],
    )

    obstacle_mass.add(
        "joint",
        type="free",
    )

    obstacle_mass.add(
        "geom",
        name=f"obstacle_mass_{id}",
        type="cylinder",
        pos=[0, 0, 0],
        conaffinity=1,
        condim=4,
        mass=PENDULUM_MASS,
        size=[0.1, 0.1, 0.1],
    )

    obstacle_mass.add(
        "site",
        name=f"mass_site_{id}",
        pos=[0, 0, 0],
    )

    tendon_spatial = mjcf_desc.tendon.add(
        "spatial",
        limited="true",
        range=[0, PENDULUM_L],
        width=0.005,
    )

    tendon_spatial.add(
        "site",
        site=f"attachment_site_{id}",
    )

    tendon_spatial.add(
        "site",
        site=f"mass_site_{id}",
    )

    return mjcf_desc


def add_floor(mjcf_desc):
    mjcf_desc.worldbody.add(
        "geom",
        name="floor",
        type="plane",
        pos=[155, 0, 0],
        rgba=[0.8, 0.9, 0.8, 1],
        size=[160.0, 8.0, 1.0],
        material="striped_material",
        conaffinity=1,
        condim=3,
    )

    return mjcf_desc


def add_slant(mjcf_desc, angle_deg: float):
    L_1 = 50.0
    L_2 = 100.0
    ALPHA = angle_deg
    OFFSET = 5.0

    alpha_rad = np.deg2rad(ALPHA)
    ramp_pos_x = L_1 + L_2 / 2 * np.cos(alpha_rad) - OFFSET
    ramp_pos_z = L_2 / 2 * np.sin(alpha_rad)

    mjcf_desc.worldbody.add(
        "geom",
        name="floor_flat",
        type="plane",
        pos=[L_1 / 2 - OFFSET, 0, 0],
        rgba=[0.8, 0.9, 0.8, 1],
        size=[L_1 / 2, 8.0, 1.0],
        material="striped_material",
        conaffinity=1,
        condim=3,
    )

    mjcf_desc.worldbody.add(
        "geom",
        name="floor_slant",
        type="plane",
        pos=[ramp_pos_x, 0, ramp_pos_z],
        rgba=[0.8, 0.9, 0.8, 1],
        size=[L_2 / 2, 8.0, 1.0],
        material="striped_material2",
        conaffinity=1,
        condim=3,
        euler=[0, -alpha_rad, 0],
    )

    return mjcf_desc


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    env = HalfCheetah(render_mode="human")
    env.generate_arena(obstacle_type="multiple", mass=50)

    env.reset(seed=2)

    for t in range(1000):
        action = env.action_space.sample()
        obs, rew, tru, term, info = env.step(action)

    env.close()
