# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    # Check cube positions
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)

    # Check gripper positions
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )

    return stacked

def cubes_approached(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    approached = pose_diff < diff_threshold

    return approached

def cube_placed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
    xy_threshold: float = 0.1,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.0001,
    rtol=0.0001,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]

    pos_diff = cube_2.data.root_pos_w - bin.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)

    # Compute cube height difference
    h_dist = torch.norm(pos_diff[:, 2:], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, h_dist < height_threshold)

    # Check gripper positions
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )

    has_failed = torch.tensor(False, dtype=torch.bool)
    if invalid_ee_frame_reached(env):
        has_failed = torch.tensor(True, dtype=torch.bool)

    stacked = torch.logical_and(stacked, has_failed.logical_not())

    return stacked


def invalid_ee_frame_reached(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    def quaternion_to_euler(w, x, y, z):
        # Roll (X-axis)
        roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x**2 + y**2))
        # Pitch (Y-axis)
        sin_p = 2 * (w*y - z*x)
        pitch = np.arcsin(sin_p)
        # Yaw (Z-axis)
        yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    end_effector_pos = ee_frame.data.target_pos_w.cpu().numpy()[:, 0, :]
    end_effector_quat = ee_frame.data.target_quat_w.cpu().numpy()[:, 0, :]

    roll, pitch, yaw = quaternion_to_euler(end_effector_quat[0,0],
                    end_effector_quat[0,1],
                    end_effector_quat[0,2],
                    end_effector_quat[0,3])

    pitch_tensor = torch.tensor(pitch)
    roll_tensor = torch.tensor(roll)

    res1 = torch.logical_or(pitch_tensor < -10, pitch_tensor > 10)
    res2 = torch.logical_or(roll_tensor <-100, roll_tensor>-80)
    reached = torch.logical_or(res1, res2)

    return reached
