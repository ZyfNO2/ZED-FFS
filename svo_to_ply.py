# -*- coding: utf-8 -*-
"""
ZED SVO -> PLY Point Cloud Reconstruction using Fast-FoundationStereo

Recommended Parameters (Best Quality):
    python svo_to_ply.py --svo "path/to/your.svo2" \\
        --depth_edge_threshold 0.01 \\
        --temporal_warmup_frames 0 --temporal_min_half_frames 1 \\
        --nb_neighbors 100 --std_ratio 0.2 \\
        --minimal_filtering

Usage:
    python svo_to_ply.py --svo "path/to/your.svo2"
    python svo_to_ply.py --svo "path/to/your.svo2" --scale 0.5 --frame_skip 5
"""

import os
import sys
import argparse
import logging
import time
import shutil
import numpy as np
import cv2
import torch
import open3d as o3d
from pathlib import Path
from typing import Tuple, Optional, List, Generator

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, code_dir)

import pyzed.sl as sl
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed,
    depth2xyzmap, toOpen3dCloud, vis_disparity,
)

os.environ['TORCHDYNAMO_DISABLE'] = '1'
torch.backends.cudnn.benchmark = True


class SVOReader:
    """ZED SVO file reader - extracts stereo pairs, poses, intrinsics"""

    def __init__(self, svo_path: str, z_far: float = 20.0):
        self.svo_path = svo_path
        self.z_far = z_far
        self.zed = None
        self.K = None
        self.baseline = None
        self.resolution = None

    def __enter__(self):
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init.depth_maximum_distance = self.z_far
        init.set_from_svo_file(self.svo_path)

        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO: {status}")

        camera_info = self.zed.get_camera_information()
        calib = camera_info.camera_configuration.calibration_parameters
        left_cam = calib.left_cam
        right_cam = calib.right_cam

        self.K = np.array([
            [left_cam.fx, 0, left_cam.cx],
            [0, left_cam.fy, left_cam.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.baseline = calib.get_camera_baseline()
        self.resolution = (
            camera_info.camera_configuration.resolution.width,
            camera_info.camera_configuration.resolution.height
        )

        pos_tracking = sl.PositionalTrackingParameters()
        pos_tracking.enable_area_memory = True
        pos_tracking.enable_pose_smoothing = True
        ret = self.zed.enable_positional_tracking(pos_tracking)
        if ret != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable positional tracking: {ret}")

        logging.info(f"SVO opened: {self.svo_path}")
        logging.info(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        logging.info(f"Baseline: {self.baseline:.4f}m")
        logging.info(f"Intrinsics fx={self.K[0,0]:.2f} fy={self.K[1,1]:.2f}")
        return self

    def __exit__(self, *args):
        if self.zed is not None:
            self.zed.close()

    def stream_frames(self, frame_skip: int = 5, max_ok_frames: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Yield (left_rgb, right_rgb, pose_4x4) for each frame with valid pose tracking"""
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 50
        left_mat = sl.Mat()
        right_mat = sl.Mat()
        pose = sl.Pose()
        frame_count = 0
        yielded = 0
        ok_frames = 0
        bad_frames = 0
        first_ok_captured = []

        while True:
            frame_count += 1

            grab_result = self.zed.grab(runtime)
            if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if grab_result != sl.ERROR_CODE.SUCCESS:
                continue

            if (frame_count - 1) % (frame_skip + 1) != 0:
                continue

            self.zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            self.zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            err = self.zed.get_position(pose)
            if err == sl.POSITIONAL_TRACKING_STATE.OK:
                ok_frames += 1
                if len(first_ok_captured) < max_ok_frames:
                    first_ok_captured.append(frame_count)
            else:
                bad_frames += 1
                continue

            pose_data = pose.pose_data()

            left_img = left_mat.get_data().copy()
            right_img = right_mat.get_data().copy()
            if hasattr(pose_data, 'to_matrix'):
                pose_matrix = np.array(pose_data.to_matrix(), dtype=np.float64)
            elif hasattr(pose_data, 'm'):
                pose_matrix = np.array(pose_data.m, dtype=np.float64).reshape(4, 4)
            else:
                try:
                    pose_matrix = np.array(pose_data, dtype=np.float64)
                except (ValueError, TypeError):
                    t = pose.get_translation().get()
                    o = pose.get_orientation().get()
                    import scipy.spatial.transform as st
                    rot = st.Rotation.from_quat([o[0], o[1], o[2], o[3]])
                    pose_matrix = np.eye(4, dtype=np.float64)
                    pose_matrix[:3, :3] = rot.as_matrix()
                    pose_matrix[:3, 3] = [t[0], t[1], t[2]]

            if left_img.shape[2] == 4:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGRA2RGB)
            else:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

            if right_img.shape[2] == 4:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGRA2RGB)
            else:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            yielded += 1
            yield left_img, right_img, pose_matrix

            if yielded >= max_ok_frames:
                logging.info(f"Captured {max_ok_frames} OK frames, stopping early")
                break

        if first_ok_captured:
            logging.info(f"First OK frames at indices: {first_ok_captured}")
        logging.info(f"EXITING GENERATOR: Total frames scanned: {frame_count}, yielded: {yielded} (OK:{ok_frames}, bad:{bad_frames})")


class FFSInference:
    """Fast-FoundationStereo inference engine"""

    def __init__(self, model_dir: str, device: str = 'cuda'):
        self.device = device
        cfg_path = os.path.join(os.path.dirname(model_dir), 'cfg.yaml')

        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        logging.info(f"Loading FFS model from: {model_dir}")
        model = torch.load(model_dir, map_location='cpu', weights_only=False)
        model.args.valid_iters = cfg.get('valid_iters', 8)
        model.args.max_disp = cfg.get('max_disp', 192)
        self.model = model.to(device).eval()
        self.cfg = cfg
        logging.info("FFS model loaded successfully")

    @torch.no_grad()
    def infer(self, left_img: np.ndarray, right_img: np.ndarray,
              scale: float = 0.5, valid_iters: int = 8,
              min_depth: float = 0.5, max_depth: float = 15.0,
              depth_edge_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: (disparity, depth, xyz_map_in_camera_coords)

        Filtering:
        1. Removes points outside min_depth..max_depth range
        2. Removes points at depth discontinuities (object edges)
        """
        if scale != 1.0:
            left_img = cv2.resize(left_img, fx=scale, fy=scale, dsize=None)
            right_img = cv2.resize(right_img, fx=scale, fy=scale, dsize=None)

        H, W = left_img.shape[:2]

        img0 = torch.as_tensor(left_img).to(self.device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_img).to(self.device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = self.model.forward(img0, img1, iters=valid_iters,
                                       test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0

        K_scaled = self.K.copy().astype(np.float32)
        K_scaled[:2] *= scale

        d_near_th = int(K_scaled[0, 0] * self.baseline / min_depth)
        d_far_th  = max(16, int(K_scaled[0, 0] * self.baseline / max_depth))

        valid_disp_mask = (disp > d_far_th) & (disp < d_near_th)
        combined_invalid = invalid | (~valid_disp_mask)
        disp_clean = disp.copy()
        disp_clean[combined_invalid] = np.inf

        depth_raw = (K_scaled[0, 0] * self.baseline / disp_clean).astype(np.float32)
        depth_raw[depth_raw < min_depth] = 0
        depth_raw[depth_raw > max_depth] = 0

        if depth_edge_threshold > 0:
            sobel_x = cv2.Sobel(depth_raw, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(depth_raw, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            valid_depth_mask = depth_raw > 0.05
            rel_grad = np.zeros_like(depth_raw)
            np.divide(grad_mag, depth_raw, out=rel_grad, where=valid_depth_mask)
            edge_mask = (rel_grad > depth_edge_threshold) & valid_depth_mask
            combined_invalid = combined_invalid | edge_mask

        depth = depth_raw.copy()
        depth[combined_invalid] = 0

        xyz_map = depth2xyzmap(depth, K_scaled)
        return disp, depth, xyz_map


import yaml


class PointCloudFuser:
    """Multi-frame point cloud fusion with pose transform"""

    def __init__(self, voxel_size: float = 0.02, nb_neighbors: int = 100,
                 std_ratio: float = 1.0, sparse_bin_factor: float = 2.0,
                 min_pts_per_bin: int = 3,
                 radius_nb_points: int = 10, radius_radius: float = 0.05,
                 dbscan_eps: float = 0.08, dbscan_min_pts: int = 15,
                 cone_n_theta: int = 36, cone_n_phi: int = 18,
                 cone_count_ratio: float = 3.0, cone_discard_outer_ratio: float = 0.5,
                 temporal_warmup_frames: int = 5,
                 temporal_min_half_frames: int = 2,
                 skip_dbscan: bool = False,
                 minimal_filtering: bool = False):
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.sparse_bin_factor = sparse_bin_factor
        self.min_pts_per_bin = min_pts_per_bin
        self.radius_nb_points = radius_nb_points
        self.radius_radius = radius_radius
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_pts = dbscan_min_pts
        self.cone_n_theta = cone_n_theta
        self.cone_n_phi = cone_n_phi
        self.cone_count_ratio = cone_count_ratio
        self.cone_discard_outer_ratio = cone_discard_outer_ratio
        self.temporal_warmup_frames = temporal_warmup_frames
        self.temporal_min_half_frames = temporal_min_half_frames
        self.skip_dbscan = skip_dbscan
        self.minimal_filtering = minimal_filtering
        self.all_points = []
        self.all_colors = []
        self.bin_temporal = {}

    def add_frame(self, xyz_map: np.ndarray, color_img: np.ndarray,
                  pose_4x4: np.ndarray, valid_depth_range: Tuple[float, float] = (0.1, 100.0)):
        """Transform camera-frame points to world frame and accumulate

        Coordinate systems:
          depth2xyzmap produces OpenCV convention: X→right, Y↓down, Z→forward
          ZED pose expects RIGHT_HANDED_Y_UP:      X→right, Y↑up,   Z→backward
          Conversion: x'=x, y'=-y, z'=-z
        """
        points = xyz_map.reshape(-1, 3)
        colors = color_img.reshape(-1, 3)

        valid_mask = (points[:, 2] > valid_depth_range[0]) & (points[:, 2] < valid_depth_range[1])
        points = points[valid_mask]
        colors = colors[valid_mask]

        if len(points) == 0:
            return

        cv2zed = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        ones = np.ones((len(points), 1), dtype=np.float64)
        pts_homo = np.hstack([points.astype(np.float64), ones])
        pts_zed = (cv2zed @ pts_homo.T).T
        pts_world = (pose_4x4.astype(np.float64) @ pts_zed.T).T[:, :3]

        if len(self.all_points) == 0:
            logging.info(f"First frame pose:\n{pose_4x4}")
            logging.info(f"First frame world bbox: "
                        f"x=[{pts_world[:,0].min():.2f},{pts_world[:,0].max():.2f}] "
                        f"y=[{pts_world[:,1].min():.2f},{pts_world[:,1].max():.2f}] "
                        f"z=[{pts_world[:,2].min():.2f},{pts_world[:,2].max():.2f}]")

        self.all_points.append(pts_world)
        self.all_colors.append(colors)

        bin_size = self.voxel_size * self.sparse_bin_factor
        bin_keys = ((pts_world[:, 0] / bin_size).astype(np.int64) |
                    (pts_world[:, 1] / bin_size).astype(np.int64) << 10 |
                    (pts_world[:, 2] / bin_size).astype(np.int64) << 20)
        frame_idx = len(self.all_points) - 1
        for bk in np.unique(bin_keys):
            if bk not in self.bin_temporal:
                self.bin_temporal[bk] = set()
            self.bin_temporal[bk].add(frame_idx)

    def process_and_save(self, output_path: str) -> o3d.geometry.PointCloud:
        """Merge, sparse filter, downsample, three-stage denoise, save PLY"""
        if len(self.all_points) == 0:
            raise ValueError("No points to fuse")

        # Helper function to save intermediate results
        def save_intermediate(pcd, step_name, step_num):
            intermediate_path = output_path.replace('.ply', f'_{step_num:02d}_{step_name}.ply')
            o3d.io.write_point_cloud(intermediate_path, pcd)
            logging.info(f"Intermediate saved [{step_num}]: {intermediate_path} ({len(pcd.points)} points)")

        all_pts = np.vstack(self.all_points)
        all_clr = np.vstack(self.all_colors)
        logging.info(f"Fusing {len(all_pts)} raw points from {len(self.all_points)} frames")

        bin_size = self.voxel_size * self.sparse_bin_factor
        keys = ((all_pts[:, 0] / bin_size).astype(np.int64) |
                (all_pts[:, 1] / bin_size).astype(np.int64) << 10 |
                (all_pts[:, 2] / bin_size).astype(np.int64) << 20)
        unique_keys, counts = np.unique(keys, return_counts=True)
        valid_bins = set(unique_keys[counts >= self.min_pts_per_bin])
        keep_mask = np.array([k in valid_bins for k in keys])
        all_pts = all_pts[keep_mask]
        all_clr = all_clr[keep_mask]
        logging.info(f"Sparse filter: removed {(~keep_mask).sum()} noise points, kept {len(all_pts)}")

        # Save after sparse filter
        pcd_sparse = toOpen3dCloud(all_pts, all_clr)
        save_intermediate(pcd_sparse, 'sparse', 1)

        valid_keys = valid_bins & set(self.bin_temporal.keys())
        bin_keys_filtered = ((all_pts[:, 0] / bin_size).astype(np.int64) |
                            (all_pts[:, 1] / bin_size).astype(np.int64) << 10 |
                            (all_pts[:, 2] / bin_size).astype(np.int64) << 20)
        n_frames = len(self.all_points)
        mid_point = self.temporal_warmup_frames

        temporal_keep = []
        if mid_point == 0 or self.temporal_min_half_frames == 0:
            if self.temporal_min_half_frames == 0 or n_frames == 0:
                for k in bin_keys_filtered:
                    temporal_keep.append(k in valid_keys)
            else:
                for k in bin_keys_filtered:
                    if k not in valid_keys:
                        temporal_keep.append(False)
                        continue
                    frames_seen = self.bin_temporal.get(k, set())
                    after_count = len(frames_seen)
                    temporal_keep.append(after_count >= self.temporal_min_half_frames)
        else:
            before_half = set(range(0, mid_point))
            after_half = set(range(mid_point, n_frames))
            for k in bin_keys_filtered:
                if k not in valid_keys:
                    temporal_keep.append(False)
                    continue
                frames_seen = self.bin_temporal.get(k, set())
                before_count = len(frames_seen & before_half)
                after_count = len(frames_seen & after_half)
                if before_count >= self.temporal_min_half_frames and after_count >= self.temporal_min_half_frames:
                    temporal_keep.append(True)
                else:
                    temporal_keep.append(False)

        temporal_keep_mask = np.array(temporal_keep)
        pts_temporal_removed = (~temporal_keep_mask).sum()
        all_pts = all_pts[temporal_keep_mask]
        all_clr = all_clr[temporal_keep_mask]
        logging.info(f"Temporal filter (warmup={self.temporal_warmup_frames}, min_half={self.temporal_min_half_frames}): removed {pts_temporal_removed} noise points, kept {len(all_pts)}")

        # Save after temporal filter
        pcd_temporal = toOpen3dCloud(all_pts, all_clr)
        save_intermediate(pcd_temporal, 'temporal', 2)

        if self.minimal_filtering:
            logging.info("Minimal filtering mode: skipping voxel, keeping statistical/radius/DBSCAN")
            pcd = pcd_temporal
            after_voxel = len(pcd.points)

            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                    std_ratio=self.std_ratio)
            after_stat = len(pcd.points)
            logging.info(f"Statistical outlier removal (k={self.nb_neighbors}, std={self.std_ratio}): {after_voxel} -> {after_stat}")
            save_intermediate(pcd, 'statistical', 3)

            cl, ind = pcd.remove_radius_outlier(nb_points=self.radius_nb_points,
                                                radius=self.radius_radius)
            pcd = cl
            after_radius = len(pcd.points)
            logging.info(f"Radius outlier removal (nb={self.radius_nb_points}, r={self.radius_radius}): {after_stat} -> {after_radius}")
            save_intermediate(pcd, 'radius', 4)

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                labels = np.array(pcd.cluster_dbscan(eps=self.dbscan_eps,
                                                      min_points=self.dbscan_min_pts,
                                                      print_progress=False))
            if len(labels) > 0:
                max_label = labels.max()
                cluster_sizes = [(labels == i).sum() for i in range(max_label + 1)]
                valid_mask = labels >= 0
                for i, sz in enumerate(cluster_sizes):
                    if sz < self.dbscan_min_pts:
                        valid_mask[labels == i] = False
                pcd = pcd.select_by_index(np.where(valid_mask)[0])
                after_dbscan = len(pcd.points)
                logging.info(f"DBSCAN clustering (eps={self.dbscan_eps}, min_pts={self.dbscan_min_pts}): {after_radius} -> {after_dbscan} (kept {max_label+1} clusters)")
            else:
                logging.warning("DBSCAN returned no labels, skipping cluster filter")
                after_dbscan = after_radius
            save_intermediate(pcd, 'dbscan', 5)
            after_cone = after_dbscan
        else:
            pcd = pcd_temporal
            orig_len = len(pcd.points)

            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            after_voxel = len(pcd.points)
            logging.info(f"Voxel downsample: {orig_len} -> {after_voxel}")

            # Save after voxel downsample
            save_intermediate(pcd, 'voxel', 3)

            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                    std_ratio=self.std_ratio)
            after_stat = len(pcd.points)
            logging.info(f"Statistical outlier removal (k={self.nb_neighbors}, std={self.std_ratio}): {after_voxel} -> {after_stat}")

            # Save after statistical outlier removal
            save_intermediate(pcd, 'statistical', 4)

            cl, ind = pcd.remove_radius_outlier(nb_points=self.radius_nb_points,
                                                radius=self.radius_radius)
            pcd = cl
            after_radius = len(pcd.points)
            logging.info(f"Radius outlier removal (nb={self.radius_nb_points}, r={self.radius_radius}): {after_stat} -> {after_radius}")

            # Save after radius outlier removal
            save_intermediate(pcd, 'radius', 5)

            if self.skip_dbscan:
                logging.info("DBSCAN skipped by user")
                after_dbscan = after_radius
            else:
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                    labels = np.array(pcd.cluster_dbscan(eps=self.dbscan_eps,
                                                          min_points=self.dbscan_min_pts,
                                                          print_progress=False))
                if len(labels) > 0:
                    max_label = labels.max()
                    cluster_sizes = [(labels == i).sum() for i in range(max_label + 1)]
                    valid_mask = labels >= 0
                    for i, sz in enumerate(cluster_sizes):
                        if sz < self.dbscan_min_pts:
                            valid_mask[labels == i] = False
                    pcd = pcd.select_by_index(np.where(valid_mask)[0])
                    after_dbscan = len(pcd.points)
                    logging.info(f"DBSCAN clustering (eps={self.dbscan_eps}, min_pts={self.dbscan_min_pts}): {after_radius} -> {after_dbscan} (kept {max_label+1} clusters)")
                else:
                    logging.warning("DBSCAN returned no labels, skipping cluster filter")
                    after_dbscan = after_radius

            # Save after DBSCAN clustering
            save_intermediate(pcd, 'dbscan', 6)

            pts = np.asarray(pcd.points)
            valid_mask = self._detect_conical_artifacts(pts)
            removed_cone = (~valid_mask).sum()
            pcd = pcd.select_by_index(np.where(valid_mask)[0])
            after_cone = len(pcd.points)
            logging.info(f"Conical artifact removal (theta={self.cone_n_theta}, phi={self.cone_n_phi}, ratio={self.cone_count_ratio}): {after_dbscan} -> {after_cone} (removed {removed_cone})")

            # Save after conical artifact removal (final intermediate)
            save_intermediate(pcd, 'conical', 7)

        o3d.io.write_point_cloud(output_path, pcd)
        logging.info(f"Saved: {output_path}")
        logging.info(f"Final point cloud: {len(pcd.points)} points")
        return pcd

    def _detect_conical_artifacts(self, points: np.ndarray) -> np.ndarray:
        """Detect and remove conical/radial artifacts.

        These artifacts form cone/funnel shapes emanating from the origin:
        - dense near the camera, sparse far away
        - appear as planes/sheets extending from near to far
        Detection: in each (theta, phi) direction, if inner-count >> outer-count,
        the outer points in that direction are marked as conical noise.
        """
        vectors = points
        distances = np.linalg.norm(vectors, axis=1)

        d_min = np.percentile(distances, 2)
        d_max = np.percentile(distances, 98)
        if d_max <= d_min:
            return np.ones(len(points), dtype=bool)

        d_median = np.median(distances)
        inner_mask = distances <= d_median

        theta = np.arctan2(vectors[:, 1], vectors[:, 0])
        phi = np.arccos(np.clip(vectors[:, 2] / np.maximum(distances, 1e-6), -1, 1))

        theta_bins = np.linspace(-np.pi, np.pi, self.cone_n_theta + 1)
        phi_bins = np.linspace(0, np.pi, self.cone_n_phi + 1)

        ti = np.clip(np.digitize(theta, theta_bins) - 1, 0, self.cone_n_theta - 1)
        pi = np.clip(np.digitize(phi, phi_bins) - 1, 0, self.cone_n_phi - 1)

        inner_counts = np.zeros((self.cone_n_theta, self.cone_n_phi), dtype=np.int32)
        outer_counts = np.zeros((self.cone_n_theta, self.cone_n_phi), dtype=np.int32)
        np.add.at(inner_counts, (ti[inner_mask], pi[inner_mask]), 1)
        np.add.at(outer_counts, (ti[~inner_mask], pi[~inner_mask]), 1)

        ratio = (inner_counts.astype(float) + 1.0) / (outer_counts.astype(float) + 1.0)

        valid = np.ones(len(points), dtype=bool)
        discard_outer = int(self.cone_discard_outer_ratio * self.cone_n_theta) or 1
        discard_start = self.cone_n_theta - discard_outer

        for idx in range(len(points)):
            if inner_mask[idx]:
                continue
            ti_bin = ti[idx]
            if ti_bin >= discard_start and ratio[ti_bin, pi[idx]] > self.cone_count_ratio:
                valid[idx] = False

        return valid


def main():
    parser = argparse.ArgumentParser(
        description='ZED SVO -> PLY using Fast-FoundationStereo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python svo_to_ply.py --svo "C:/Users/ZYF/Documents/ZED/recording.svo2"
  python svo_to_ply.py --svo "recording.svo2" --scale 0.5 --frame_skip 5 --z_far 15
        """,
    )
    parser.add_argument('--svo', type=str, required=True, help='Path to SVO file (.svo2)')
    parser.add_argument('--model_dir', type=str,
                       default=os.path.join(code_dir, 'weights/model_best_bp2_serialize.pth'),
                       help='FFS model path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: output/)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Image scale for inference (default: 0.5)')
    parser.add_argument('--valid_iters', type=int, default=8,
                       help='Refinement iterations (default: 8)')
    parser.add_argument('--frame_skip', type=int, default=5,
                       help='Frames to skip between processed frames (default: 5)')
    parser.add_argument('--max_ok_frames', type=int, default=5,
                       help='Max frames with OK pose tracking to process (default: 5)')
    parser.add_argument('--min_depth', type=float, default=0.5,
                       help='Minimum valid depth in meters (default: 0.5)')
    parser.add_argument('--max_depth', type=float, default=15.0,
                       help='Maximum valid depth in meters (default: 15.0)')
    parser.add_argument('--depth_edge_threshold', type=float, default=0.1,
                       help='Relative depth gradient threshold for edge filtering (default: 0.1, 0 to disable)')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                       help='Voxel downsample size in meters (default: 0.02)')
    parser.add_argument('--min_pts_per_bin', type=int, default=3,
                       help='Min points per spatial bin for sparse filtering (default: 3)')
    parser.add_argument('--sparse_bin_factor', type=float, default=2.0,
                       help='Spatial bin size = voxel_size * this (default: 2.0)')
    parser.add_argument('--temporal_warmup_frames', type=int, default=5,
                       help='Number of warmup frames before temporal filtering starts (default: 5)')
    parser.add_argument('--temporal_min_half_frames', type=int, default=2,
                       help='Min frames a bin must appear in both halves to keep (default: 2)')
    parser.add_argument('--skip_dbscan', action='store_true',
                       help='Skip DBSCAN clustering step')
    parser.add_argument('--minimal_filtering', action='store_true',
                       help='Skip voxel/statistical/radius/DBSCAN/conical, only minimal filtering')
    parser.add_argument('--nb_neighbors', type=int, default=100,
                       help='Statistical outlier removal k-nearest neighbors (default: 100)')
    parser.add_argument('--std_ratio', type=float, default=0.2,
                       help='Statistical outlier removal std ratio threshold (default: 0.2)')
    parser.add_argument('--radius_nb_points', type=int, default=10,
                       help='Radius outlier removal min neighbors in sphere (default: 10)')
    parser.add_argument('--radius_radius', type=float, default=0.05,
                       help='Radius outlier removal sphere radius in meters (default: 0.05)')
    parser.add_argument('--dbscan_eps', type=float, default=0.08,
                       help='DBSCAN cluster epsilon distance in meters (default: 0.08)')
    parser.add_argument('--dbscan_min_pts', type=int, default=15,
                       help='DBSCAN minimum points per cluster (default: 15)')
    parser.add_argument('--cone_n_theta', type=int, default=36,
                       help='Conical detection theta bins (default: 36)')
    parser.add_argument('--cone_n_phi', type=int, default=18,
                       help='Conical detection phi bins (default: 18)')
    parser.add_argument('--cone_count_ratio', type=float, default=3.0,
                       help='Conical detection inner/outer count ratio threshold (default: 3.0)')
    parser.add_argument('--cone_discard_outer_ratio', type=float, default=0.5,
                       help='Fraction of outer theta bins to check for conical artifacts (default: 0.5)')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    if not os.path.exists(args.svo):
        logging.error(f"SVO file not found: {args.svo}")
        return

    if not os.path.exists(args.model_dir):
        logging.error(f"Model not found: {args.model_dir}")
        return

    output_dir = args.output or os.path.join(code_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    try:
        ffs = FFSInference(args.model_dir)
    except Exception as e:
        logging.error(f"Failed to load FFS model: {e}")
        return

    try:
        with SVOReader(args.svo, z_far=args.max_depth) as reader:
            ffs.K = reader.K
            ffs.baseline = reader.baseline
            fuser = PointCloudFuser(
                voxel_size=args.voxel_size,
                nb_neighbors=args.nb_neighbors,
                std_ratio=args.std_ratio,
                min_pts_per_bin=args.min_pts_per_bin,
                sparse_bin_factor=args.sparse_bin_factor,
                radius_nb_points=args.radius_nb_points,
                radius_radius=args.radius_radius,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_pts=args.dbscan_min_pts,
                cone_n_theta=args.cone_n_theta,
                cone_n_phi=args.cone_n_phi,
                cone_count_ratio=args.cone_count_ratio,
                cone_discard_outer_ratio=args.cone_discard_outer_ratio,
                temporal_warmup_frames=args.temporal_warmup_frames,
                temporal_min_half_frames=args.temporal_min_half_frames,
                skip_dbscan=args.skip_dbscan,
                minimal_filtering=args.minimal_filtering,
            )

            gen = reader.stream_frames(frame_skip=args.frame_skip, max_ok_frames=args.max_ok_frames)
            for idx, (left, right, pose) in enumerate(gen):
                t_frame = time.time()
                disp, depth, xyz = ffs.infer(left, right, scale=args.scale,
                                             valid_iters=args.valid_iters,
                                             min_depth=args.min_depth,
                                             max_depth=args.max_depth,
                                             depth_edge_threshold=args.depth_edge_threshold)
                if args.scale != 1.0:
                    left_scaled = cv2.resize(left, fx=args.scale, fy=args.scale, dsize=None)
                else:
                    left_scaled = left
                fuser.add_frame(xyz, left_scaled, pose,
                                valid_depth_range=(args.min_depth, args.max_depth))
                dt = time.time() - t_frame
                n_pts = (xyz.reshape(-1, 3)[:, 2] > 0.1).sum()
                logging.info(f"Frame {idx}: {n_pts} pts, {dt:.2f}s")

            svo_name = Path(args.svo).stem
            output_ply = os.path.join(output_dir, f"{svo_name}.ply")
            pcd = fuser.process_and_save(output_ply)

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - t_start
    logging.info("=" * 60)
    logging.info(f"DONE! Output: {output_ply}")
    logging.info(f"Final point cloud: {len(pcd.points)} points")
    logging.info(f"Total time: {elapsed:.1f}s")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
