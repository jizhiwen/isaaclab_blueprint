# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ... import mdp
from . import ik_rel_env_cfg



@configclass
class RealmanCubePickPlaceEnvCfg(ik_rel_env_cfg.RealmanCubePickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (realman)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*"], scale=1.0, use_default_offset=False
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger[1-2]_joint"],
            open_command_expr={"finger[1-2]_joint": 0.0},
            close_command_expr={"finger[1-2]_joint": 0.04},
        )

