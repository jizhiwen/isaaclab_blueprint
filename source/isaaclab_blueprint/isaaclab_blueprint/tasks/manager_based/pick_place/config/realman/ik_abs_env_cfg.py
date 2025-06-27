# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ... import mdp
from . import joint_pos_env_cfg

##
# Pre-defined configs
##
# from isaaclab_assets.robots.realman import REALMAN_HIGH_PD_CFG # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.one_cube_object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        table_cam_normals = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "normals",
                "normalize": True,
                "save_image_to_file": False,
                "image_path": "table_cam",
            },
        )
        table_cam_segmentation = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "table_cam",
            },
        )
        table_cam_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "_isaaclab_out_/table_cam",
            },
        )
        table_high_cam_normals = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "normals",
                "normalize": True,
                "save_image_to_file": False,
                "image_path": "table_high_cam",
            },
        )
        table_high_cam_segmentation = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "table_high_cam",
            },
        )
        table_high_cam_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "_isaaclab_out_/table_high_cam",
            },
        )
        table_side_cam_normals = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_side_cam"),
                "data_type": "normals",
                "normalize": True,
                "save_image_to_file": False,
                "image_path": "table_side_cam",
            },
        )
        table_side_cam_segmentation = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_side_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "table_side_cam",
            },
        )
        table_side_cam_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_side_cam"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "_isaaclab_out_/table_side_cam",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
                "gripper_open_val": torch.tensor([0.0]),
                "gripper_threshold": 0.004,
                "diff_threshold": 0.09,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class RealmanCubePickPlaceEnvCfg(joint_pos_env_cfg.RealmanCubePickPlaceEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        # self.scene.robot = REALMAN_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set actions for the specific robot type (realman)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            body_name="Link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        MAPPING = {
            "class:cube_2": (255, 184, 48, 255),
            "class:bin": (255, 184, 48, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (125, 125, 125, 255),
            "class:UNLABELLED": (10, 10, 10, 255),
            "class:BACKGROUND": (10, 10, 10, 255),
        }

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pgi/pgi_base/table_cam",
            update_period=0.0666,
            height=480,
            width=640,
            data_types=["rgb", "semantic_segmentation", "normals"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.0, focus_distance=400.0, horizontal_aperture=35.0, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.01, 0.0, -0.15), rot=(0.0, 0.0, 0.42262, 0.90631), convention="ros"),
        )

        # Set table view camera
        self.scene.table_high_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_high_cam",
            update_period=0.0666,
            height=480,
            width=640,
            data_types=["rgb", "semantic_segmentation", "normals"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=MAPPING,
            # h = 1.37456192*f
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0, focus_distance=400.0, horizontal_aperture=50.0, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.02255, -0.84269, 0.20111), rot=(-0.57608, 0.81666, -0.02775, 0.02076), convention="ros"),
        )

        # Set table view camera
        self.scene.table_side_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_side_cam",
            update_period=0.0666,
            height=480,
            width=640,
            data_types=["rgb", "semantic_segmentation", "normals"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=MAPPING,
            # h = 1.37456192*f
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=40.0, focus_distance=400.0, horizontal_aperture=30.0, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.55604, -0.36226, 0.12774), rot=(-0.41194, 0.57891, 0.56742, -0.41616), convention="ros"),
        )
