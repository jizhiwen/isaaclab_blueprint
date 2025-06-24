# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_blueprint.robots.realman import REALMAN_HIGH_PD_CFG

from isaaclab_blueprint.tasks.manager_based.pick_place.pick_place_env_cfg import PickPlaceEnvCfg
from isaaclab_blueprint.tasks.manager_based.pick_place.mdp import realman_pick_place_events


@configclass
class EventCfg:
    """Configuration for events."""

    init_realman_arm_pose = EventTerm(
        func=realman_pick_place_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [1.6073, 0.684, 1.3142, -0.006, 1.1221, 0.0643, 0.0, 0.0],
        },
    )

    # randomize_realman_joint_state = EventTerm(
    #     func=realman_pick_place_events.randomize_joint_by_gaussian_offset,
    #     mode="reset",
    #     params={
    #         "mean": 0.0,
    #         "std": 0.02,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    reset_realman_arm_pose = EventTerm(
        func=realman_pick_place_events.reset_joint_by_default_joint_pose,
        mode="reset",
    )

    randomize_cube_positions = EventTerm(
        func=realman_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            # X: Left/Right Y: Front/Back
            #"pose_range": {"x": (-0.06, 0.12), "y": (-0.38, -0.26), "z": (-0.0140 , -0.0140), "yaw": (-1, 1, 0)},
            "pose_range": {"x": (-0.10, 0.18), "y": (-0.43, -0.20), "z": (-0.0140 , -0.0140), "yaw": (0, 0, 0)},
            "min_separation": 0.02,
            "asset_cfgs": [SceneEntityCfg("cube_2")],
            # "radius": 0.40,
        },
    )



@configclass
class RealmanCubePickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = REALMAN_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_base.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Add semantics to objs
        self.scene.cube_2.semantic_tags = [("class", "cube")]
        self.scene.bin.semantic_tags = [("class", "bin")]

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

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rm65/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/rm65/Link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.10667],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/pgi/finger2_link",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/pgi/finger1_link",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )
