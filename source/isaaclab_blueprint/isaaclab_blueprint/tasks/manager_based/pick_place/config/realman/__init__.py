# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


##
# Register Gym environments.
##

res = gym.register(
    id="Template-Isaaclab-Blueprint-PickPlace-Cube-Realman-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:RealmanCubePickPlaceEnvCfg",
    }
)

res = gym.register(
    id="Template-Isaaclab-Blueprint-PickPlace-Cube-Realman-IK-Rel-Dump-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_dump_rgb_env_cfg:RealmanCubePickPlaceEnvCfg",
    }
)

gym.register(
    id="Template-Isaaclab-Blueprint-PickPlace-Cube-Realman-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.ik_rel_mimic_env:RealmanCubePickPlaceIKRelMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_mimic_env_cfg:RealmanCubePickPlaceMimicEnvCfg",
    },
)

gym.register(
    id="Template-Isaaclab-Blueprint-PickPlace-Cube-Realman-IK-Rel-Mimic-Dump-RGB-v0",
    entry_point=f"{__name__}.ik_rel_mimic_env:RealmanCubePickPlaceIKRelMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_mimic_dump_rgb_env_cfg:RealmanCubePickPlaceMimicEnvCfg",
    },
)
