"""
Epipolar consistency checking for multi-view depth validation.

Validates depth estimates by checking cross-view consistency
using epipolar geometry constraints.
"""

import torch
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


@dataclass
class EpipolarResult:
    """Result of epipolar consistency check."""
    consistency_map: torch.Tensor  # [H, W] consistency scores
    mean_reprojection_error: float
    inlier_ratio: float
    num_valid_pairs: int


class EpipolarConsistencyChecker:
    """
    Checks depth consistency across multiple views using epipolar geometry.

    For each pixel, projects it to 3D using its depth, then reprojects
    to other views to check consistency.
    """

    def __init__(
        self,
        reprojection_threshold: float = 2.0,
        min_valid_views: int = 2,
        depth_consistency_threshold: float = 0.1
    ):
        """
        Initialize epipolar checker.

        Args:
            reprojection_threshold: Maximum reprojection error in pixels
            min_valid_views: Minimum views for valid consistency check
            depth_consistency_threshold: Relative depth difference threshold
        """
        self.reprojection_threshold = reprojection_threshold
        self.min_valid_views = min_valid_views
        self.depth_consistency_threshold = depth_consistency_threshold

    def check(
        self,
        depth_maps: torch.Tensor,
        camera_poses: List[torch.Tensor],
        camera_intrinsics: List[CameraIntrinsics],
        reference_view: int = 0
    ) -> EpipolarResult:
        """
        Check epipolar consistency for depth maps.

        Args:
            depth_maps: Depth maps [N, H, W]
            camera_poses: List of 4x4 camera pose matrices
            camera_intrinsics: List of camera intrinsics
            reference_view: Index of reference view

        Returns:
            EpipolarResult with consistency map and statistics
        """
        n_views = depth_maps.shape[0]
        H, W = depth_maps.shape[1], depth_maps.shape[2]
        device = depth_maps.device

        if n_views < 2:
            logger.warning("Need at least 2 views for epipolar check")
            return EpipolarResult(
                consistency_map=torch.ones(H, W, device=device),
                mean_reprojection_error=0.0,
                inlier_ratio=1.0,
                num_valid_pairs=0
            )

        # Get reference view data
        ref_depth = depth_maps[reference_view]
        ref_pose = self._to_numpy(camera_poses[reference_view])
        ref_K = camera_intrinsics[reference_view].to_matrix()

        # Create pixel grid
        u_coords = torch.arange(W, device=device).float()
        v_coords = torch.arange(H, device=device).float()
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')

        # Back-project reference pixels to 3D
        points_3d = self._backproject(
            u_grid, v_grid, ref_depth,
            ref_K, ref_pose
        )

        # Check consistency with other views
        consistency_scores = torch.zeros(H, W, device=device)
        valid_counts = torch.zeros(H, W, device=device)
        total_reprojection_error = 0.0
        total_inliers = 0
        total_points = 0

        for view_idx in range(n_views):
            if view_idx == reference_view:
                continue

            target_pose = self._to_numpy(camera_poses[view_idx])
            target_K = camera_intrinsics[view_idx].to_matrix()
            target_depth = depth_maps[view_idx]

            # Project to target view
            u_reproj, v_reproj, depth_reproj = self._project(
                points_3d, target_K, target_pose
            )

            # Check bounds
            valid_mask = (
                (u_reproj >= 0) & (u_reproj < W-1) &
                (v_reproj >= 0) & (v_reproj < H-1) &
                (depth_reproj > 0)
            )

            if valid_mask.sum() == 0:
                continue

            # Sample target depth at reprojected locations
            u_int = u_reproj[valid_mask].long()
            v_int = v_reproj[valid_mask].long()
            sampled_depth = target_depth[v_int, u_int]

            # Compute depth consistency
            depth_diff = torch.abs(depth_reproj[valid_mask] - sampled_depth)
            relative_diff = depth_diff / (sampled_depth + 1e-6)

            # Points are consistent if relative depth difference is small
            consistent = relative_diff < self.depth_consistency_threshold

            # Update consistency scores
            consistency_update = torch.zeros(H, W, device=device)
            consistency_update[valid_mask] = consistent.float()
            consistency_scores += consistency_update
            valid_counts[valid_mask] += 1

            # Statistics
            total_points += valid_mask.sum().item()
            total_inliers += consistent.sum().item()

            # Mean reprojection error (in depth space)
            total_reprojection_error += depth_diff.mean().item()

        # Normalize consistency scores
        valid_counts = torch.clamp(valid_counts, min=1)
        consistency_map = consistency_scores / valid_counts

        # Require minimum valid views
        insufficient_views = valid_counts < self.min_valid_views
        consistency_map[insufficient_views] = 0.5  # Uncertain

        # Compute statistics
        n_pairs = n_views - 1
        mean_error = total_reprojection_error / max(n_pairs, 1)
        inlier_ratio = total_inliers / max(total_points, 1)

        logger.debug(
            f"Epipolar check: mean_error={mean_error:.4f}, "
            f"inlier_ratio={inlier_ratio:.2%}"
        )

        return EpipolarResult(
            consistency_map=consistency_map,
            mean_reprojection_error=mean_error,
            inlier_ratio=inlier_ratio,
            num_valid_pairs=n_pairs
        )

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def _backproject(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        depth: torch.Tensor,
        K: np.ndarray,
        pose: np.ndarray
    ) -> torch.Tensor:
        """Back-project pixels to 3D world coordinates."""
        device = depth.device

        # Inverse intrinsics
        K_inv = np.linalg.inv(K)
        K_inv_torch = torch.from_numpy(K_inv).float().to(device)

        # Homogeneous pixel coordinates
        ones = torch.ones_like(u)
        pixels = torch.stack([u, v, ones], dim=-1)  # [H, W, 3]

        # Camera coordinates
        rays = torch.einsum('ij,hwj->hwi', K_inv_torch, pixels)
        points_cam = rays * depth.unsqueeze(-1)

        # World coordinates (apply inverse of extrinsic)
        R = pose[:3, :3]
        t = pose[:3, 3]

        R_inv = np.linalg.inv(R)
        R_inv_torch = torch.from_numpy(R_inv).float().to(device)
        t_torch = torch.from_numpy(t).float().to(device)

        # Transform: X_world = R^-1 @ (X_cam - t)
        points_world = torch.einsum(
            'ij,hwj->hwi',
            R_inv_torch,
            points_cam - t_torch
        )

        return points_world

    def _project(
        self,
        points_3d: torch.Tensor,
        K: np.ndarray,
        pose: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project 3D points to image coordinates."""
        device = points_3d.device

        R = pose[:3, :3]
        t = pose[:3, 3]

        R_torch = torch.from_numpy(R).float().to(device)
        t_torch = torch.from_numpy(t).float().to(device)
        K_torch = torch.from_numpy(K).float().to(device)

        # Transform to camera coordinates
        points_cam = torch.einsum('ij,hwj->hwi', R_torch, points_3d) + t_torch

        # Get depth
        depth = points_cam[..., 2]

        # Project to image
        points_proj = points_cam / (depth.unsqueeze(-1) + 1e-6)
        pixels = torch.einsum('ij,hwj->hwi', K_torch, points_proj)

        u = pixels[..., 0]
        v = pixels[..., 1]

        return u, v, depth


def compute_reprojection_error(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    pose1: np.ndarray,
    pose2: np.ndarray,
    K: np.ndarray
) -> float:
    """
    Compute mean reprojection error between two depth maps.

    Args:
        depth1: First depth map [H, W]
        depth2: Second depth map [H, W]
        pose1: Camera pose for first view [4, 4]
        pose2: Camera pose for second view [4, 4]
        K: Camera intrinsic matrix [3, 3]

    Returns:
        Mean reprojection error in pixels
    """
    H, W = depth1.shape
    device = depth1.device

    # Create pixel grid
    u = torch.arange(W, device=device).float()
    v = torch.arange(H, device=device).float()
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

    # Back-project from view 1
    K_inv = np.linalg.inv(K)
    K_inv_torch = torch.from_numpy(K_inv).float().to(device)

    pixels = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)
    rays = torch.einsum('ij,hwj->hwi', K_inv_torch, pixels)
    points_cam1 = rays * depth1.unsqueeze(-1)

    # Transform to view 2
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]

    # World coordinates
    R1_inv = np.linalg.inv(R1)
    R1_inv_torch = torch.from_numpy(R1_inv).float().to(device)
    t1_torch = torch.from_numpy(t1).float().to(device)

    points_world = torch.einsum('ij,hwj->hwi', R1_inv_torch, points_cam1 - t1_torch)

    # Camera 2 coordinates
    R2_torch = torch.from_numpy(R2).float().to(device)
    t2_torch = torch.from_numpy(t2).float().to(device)

    points_cam2 = torch.einsum('ij,hwj->hwi', R2_torch, points_world) + t2_torch

    # Project to image 2
    depth_cam2 = points_cam2[..., 2]
    valid_mask = depth_cam2 > 0

    points_proj = points_cam2 / (depth_cam2.unsqueeze(-1) + 1e-6)
    K_torch = torch.from_numpy(K).float().to(device)
    pixels2 = torch.einsum('ij,hwj->hwi', K_torch, points_proj)

    u2 = pixels2[..., 0]
    v2 = pixels2[..., 1]

    # Check bounds
    valid_mask = valid_mask & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)

    if valid_mask.sum() == 0:
        return float('inf')

    # Sample depth2 at reprojected locations
    u2_int = torch.clamp(u2.long(), 0, W-1)
    v2_int = torch.clamp(v2.long(), 0, H-1)
    sampled_depth2 = depth2[v2_int, u2_int]

    # Compute depth difference
    depth_diff = torch.abs(depth_cam2 - sampled_depth2)
    mean_error = depth_diff[valid_mask].mean().item()

    return mean_error


def compute_essential_matrix(
    pose1: np.ndarray,
    pose2: np.ndarray
) -> np.ndarray:
    """
    Compute essential matrix between two camera poses.

    Args:
        pose1: First camera pose [4, 4]
        pose2: Second camera pose [4, 4]

    Returns:
        Essential matrix [3, 3]
    """
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]

    # Relative pose
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Skew-symmetric matrix of translation
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])

    # Essential matrix
    E = tx @ R_rel

    return E


def compute_fundamental_matrix(
    pose1: np.ndarray,
    pose2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray
) -> np.ndarray:
    """
    Compute fundamental matrix between two views.

    Args:
        pose1: First camera pose [4, 4]
        pose2: Second camera pose [4, 4]
        K1: First camera intrinsics [3, 3]
        K2: Second camera intrinsics [3, 3]

    Returns:
        Fundamental matrix [3, 3]
    """
    E = compute_essential_matrix(pose1, pose2)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F
