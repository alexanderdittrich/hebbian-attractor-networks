import functools

from mujoco_playground._src.locomotion import register_environment

from crawler_playground.envs.ant import ant
from crawler_playground.envs.cheetah import (
    cheetah_friction,
    cheetah_motor_fail,
)
from crawler_playground.envs.unitree_go1 import (
    walk,
    walk_frozen_hip,
    walk_motor_fail,
)

register_environment(
    "Go1WalkFlatTerrain",
    functools.partial(walk.Walk, task="flat_terrain"),
    walk.default_config,
)

register_environment(
    "Go1WalkFrozenHip",
    functools.partial(walk_frozen_hip.WalkFrozenHip, task="flat_terrain"),
    walk.default_config,
)

register_environment(
    "Go1WalkRoughTerrain",
    functools.partial(walk.Walk, task="rough_terrain"),
    walk.default_config,
)

register_environment(
    "Go1WalkMotorFailure",
    functools.partial(walk_motor_fail.Walk, task="flat_terrain"),
    walk_motor_fail.default_config,
)

register_environment(
    "CheetahRunMotorFailure",
    functools.partial(cheetah_motor_fail.Run),
    cheetah_motor_fail.default_config,
)

register_environment(
    "CheetahRunFriction",
    functools.partial(cheetah_friction.Run),
    cheetah_friction.default_config,
)

register_environment(
    "AntRun",
    functools.partial(ant.Run),
    ant.default_config,
)
