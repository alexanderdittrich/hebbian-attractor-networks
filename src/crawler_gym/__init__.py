from gymnasium.envs.registration import register

register(
    id="ObstacleCheetah-v5",
    entry_point="crawler_gym.envs.cheetah.obstacle_cheetah_v5:HalfCheetah",
)

register(
    id="AntMotorFail-v5",
    entry_point="crawler_gym.envs.ant.ant_motor_fail_v5:Ant",
)
