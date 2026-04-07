# -*- coding: utf-8 -*-
"""
ZED SVO -> PLY Point Cloud Reconstruction using Fast-FoundationStereo

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

    def stream_frames(self, frame_skip: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Yield (left_rgb, right_rgb, pose_4x4) for each frame"""
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 50
        left_mat = sl.Mat()
        right_mat = sl.Mat()
        pose = sl.Pose()
        frame_count = 0
        yielded = 0

        while True:
            grab_result = self.zed.grab(runtime)
            if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if grab_result != sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                continue

            if frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                continue

            self.zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            self.zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            self.zed.get_position(pose)

            left_img = left_mat.get_data().copy()
            right_img = right_mat.get_data().copy()
            pose_data = pose.pose_data()
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
            frame_count += 1

        logging.info(f"Total frames scanned: {frame_count}, yielded: {yielded}")


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
              scale: float = 0.5, valid_iters: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: (disparity, depth, xyz_map_in_camera_coords)
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
        disp[invalid] = np.inf

        K_scaled = self.K.copy().astype(np.float32)
        K_scaled[:2] *= scale
        depth = (K_scaled[0, 0] * self.baseline / disp).astype(np.float32)
        depth[depth < 0.05] = 0
        depth[depth > self.cfg.get('zfar', 100)] = 0

        xyz_map = depth2xyzmap(depth, K_scaled)
        return disp, depth, xyz_map


import yaml


class PointCloudFuser:
    """Multi-frame point cloud fusion with pose transform"""

    def __init__(self, voxel_size: float = 0.02, nb_neighbors: int = 20,
                 std_ratio: float = 2.0):
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.all_points = []
        self.all_colors = []

    def add_frame(self, xyz_map: np.ndarray, color_img: np.ndarray,
                  pose_4x4: np.ndarray, valid_depth_range: Tuple[float, float] = (0.1, 100.0)):
        """Transform camera-frame points to world frame and accumulate"""
        points = xyz_map.reshape(-1, 3)
        colors = color_img.reshape(-1, 3)

        valid_mask = (points[:, 2] > valid_depth_range[0]) & (points[:, 2] < valid_depth_range[1])
        points = points[valid_mask]
        colors = colors[valid_mask]

        if len(points) == 0:
            return

        ones = np.ones((len(points), 1), dtype=np.float64)
        pts_homo = np.hstack([points.astype(np.float64), ones])
        pts_world = (pose_4x4.astype(np.float64) @ pts_homo.T).T[:, :3]

        self.all_points.append(pts_world)
        self.all_colors.append(colors)

    def process_and_save(self, output_path: str) -> o3d.geometry.PointCloud:
        """Merge, downsample, denoise, save PLY"""
        if len(self.all_points) == 0:
            raise ValueError("No points to fuse")

        all_pts = np.vstack(self.all_points)
        all_clr = np.vstack(self.all_colors)
        logging.info(f"Fusing {len(all_pts)} raw points from {len(self.all_points)} frames")

        pcd = toOpen3dCloud(all_pts, all_clr)
        orig_len = len(pcd.points)

        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        after_voxel = len(pcd.points)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                std_ratio=self.std_ratio)
        final_len = len(pcd.points)

        o3d.io.write_point_cloud(output_path, pcd)
        logging.info(f"Saved: {output_path}")
        logging.info(f"Points: {orig_len} -> {after_voxel} (voxel) -> {final_len} (outlier)")
        return pcd


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
    parser.add_argument('--z_far', type=float, default=20.0,
                       help='Max depth in meters (default: 20.0)')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                       help='Voxel downsample size in meters (default: 0.02)')
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
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    try:
        ffs = FFSInference(args.model_dir)
    except Exception as e:
        logging.error(f"Failed to load FFS model: {e}")
        return

    try:
        with SVOReader(args.svo, z_far=args.z_far) as reader:
            ffs.K = reader.K
            ffs.baseline = reader.baseline
            fuser = PointCloudFuser(voxel_size=args.voxel_size)

            for idx, (left, right, pose) in enumerate(reader.stream_frames(frame_skip=args.frame_skip)):
                t_frame = time.time()
                disp, depth, xyz = ffs.infer(left, right, scale=args.scale,
                                             valid_iters=args.valid_iters)
                if args.scale != 1.0:
                    left_scaled = cv2.resize(left, fx=args.scale, fy=args.scale, dsize=None)
                else:
                    left_scaled = left
                fuser.add_frame(xyz, left_scaled, pose,
                                valid_depth_range=(0.1, args.z_far))
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
