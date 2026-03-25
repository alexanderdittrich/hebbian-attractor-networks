import jax.numpy as jp
from mujoco_playground._src import mjx_env

from crawler_playground.envs.unitree_go1.walk import Walk


class WalkFrozenHip(Walk):
    """Go1 walk with all hip joints frozen (action space = 8)."""

    def step(self, state, action):
        # <---------------- Map actions ---------------->
        idx = jp.array([1, 2, 4, 5, 7, 8, 10, 11])

        full_action = jp.zeros(12, dtype=jp.float32)
        action = full_action.at[idx].set(action)

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

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu - 4
