# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from . import ik_abs_env_cfg


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
            "save_image_to_file": True,
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
            "save_image_to_file": True,
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
            "save_image_to_file": True,
            "image_path": "_isaaclab_out_/table_side_cam",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class RealmanCubePickPlaceEnvCfg(ik_abs_env_cfg.RealmanCubePickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set policy for RGB camera
        self.observations.rgb_camera = RGBCameraPolicyCfg()

