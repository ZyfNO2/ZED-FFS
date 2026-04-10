# -*- coding: utf-8 -*-
"""
Gradio App for Confidence-based Point Cloud Filtering

Usage:
    python confidence_slider_app.py --svo "path/to/video.svo2"
    python confidence_slider_app.py --svo "path/to/video.svo2" --server_port 7860
"""

import os
import sys
import argparse
import logging
import time
import tempfile
import shutil
import numpy as np
import cv2
import torch
import gradio as gr
import trimesh
from pathlib import Path

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, code_dir)

import pyzed.sl as sl
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, depth2xyzmap, toOpen3dCloud

os.environ['TORCHDYNAMO_DISABLE'] = '1'
torch.backends.cudnn.benchmark = True

import yaml


class SVOReader:
    """ZED SVO file reader"""

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
        return self

    def __exit__(self, *args):
        if self.zed is not None:
            self.zed.close()

    def stream_frames(self, frame_skip: int = 5, max_ok_frames: int = 100):
        """Yield (left_rgb, right_rgb, pose_4x4) for each frame"""
        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 50
        left_mat = sl.Mat()
        right_mat = sl.Mat()
        pose = sl.Pose()
        frame_count = 0
        yielded = 0

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
            if err != sl.POSITIONAL_TRACKING_STATE.OK:
                continue

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

            left_img = left_mat.get_data().copy()
            right_img = right_mat.get_data().copy()
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
                break

        logging.info(f"Total frames: {frame_count}, yielded: {yielded}")


class FFSInference:
    """Fast-FoundationStereo inference engine with confidence"""

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

    def normalize_image(self, img):
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (img / 255.0 - mean) / std

    @torch.no_grad()
    def infer(self, left_img: np.ndarray, right_img: np.ndarray,
              scale: float = 0.5, valid_iters: int = 8,
              min_depth: float = 0.5, max_depth: float = 15.0,
              depth_edge_threshold: float = 0.1,
              compute_confidence: bool = True,
              conf_temperature: float = 0.5):
        """Returns: (disparity, depth, xyz_map, confidence)"""
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
            if compute_confidence:
                disp_rl = self.model.forward(img1, img0, iters=valid_iters,
                                             test_mode=True, optimize_build_volume='pytorch1')

        disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)

        confidence = np.ones((H, W), dtype=np.float32)
        if compute_confidence:
            disp_rl = padder.unpad(disp_rl.float()).cpu().numpy().reshape(H, W)
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            x_reproj = (xx - disp).astype(np.int32)
            valid_reproj_mask = (x_reproj >= 0) & (x_reproj < W)
            conf = np.zeros((H, W), dtype=np.float32)
            for y in range(H):
                for x in range(W):
                    if not valid_reproj_mask[y, x]:
                        conf[y, x] = 0.0
                        continue
                    d = disp[y, x]
                    d_reproj = disp_rl[y, x_reproj[y, x]]
                    if d <= 0 or d_reproj <= 0 or np.isinf(d) or np.isinf(d_reproj):
                        conf[y, x] = 0.0
                        continue
                    error = abs(d - d_reproj)
                    conf[y, x] = np.exp(-error / conf_temperature)
            confidence = conf

        K_scaled = self.K.copy().astype(np.float32)
        K_scaled[:2] *= scale

        d_near_th = int(K_scaled[0, 0] * self.baseline / min_depth)
        d_far_th = max(16, int(K_scaled[0, 0] * self.baseline / max_depth))

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0
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
        confidence[combined_invalid] = 0.0

        xyz_map = depth2xyzmap(depth, K_scaled)
        return disp, depth, xyz_map, confidence


def filter_pointcloud_by_confidence(all_pts, all_clr, all_conf, conf_threshold):
    """Filter point cloud by confidence threshold"""
    if all_conf is None:
        return all_pts, all_clr
    valid_mask = all_conf > conf_threshold
    return all_pts[valid_mask], all_clr[valid_mask]


def save_pointcloud(points, colors, output_path):
    """Save point cloud as PLY file"""
    if len(points) == 0:
        return output_path
    pcd = toOpen3dCloud(points, colors)
    o3d.io.write_point_cloud(output_path, pcd)
    return output_path


def save_glb(points, colors, output_path):
    """Save point cloud as GLB for Gradio"""
    if len(points) == 0:
        return output_path
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors / 255.0))
    scene.export(output_path)
    return output_path


def process_svo(svo_path, model_dir, scale, frame_skip, max_frames,
                min_depth, max_depth, depth_edge_threshold, tmp_dir):
    """Process SVO and return raw point cloud data"""
    from core.foundation_stereo import FastFoundationStereo

    set_logging_format()
    set_seed(42)
    torch.autograd.set_grad_enabled(False)

    ffs = FFSInference(model_dir)

    all_pts_list = []
    all_clr_list = []
    all_conf_list = []

    with SVOReader(svo_path, z_far=max_depth) as reader:
        ffs.K = reader.K
        ffs.baseline = reader.baseline

        gen = reader.stream_frames(frame_skip=frame_skip, max_ok_frames=max_frames)
        for idx, (left, right, pose) in enumerate(gen):
            disp, depth, xyz, conf = ffs.infer(
                left, right, scale=scale,
                valid_iters=8, min_depth=min_depth, max_depth=max_depth,
                depth_edge_threshold=depth_edge_threshold,
                compute_confidence=True
            )

            left_scaled = cv2.resize(left, fx=scale, fy=scale, dsize=None) if scale != 1.0 else left

            points = xyz.reshape(-1, 3)
            colors = left_scaled.reshape(-1, 3)
            valid_mask = points[:, 2] > 0.1

            cv2zed = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
            ones = np.ones((len(points), 1), dtype=np.float64)
            pts_homo = np.hstack([points.astype(np.float64), ones])
            pts_zed = (cv2zed @ pts_homo.T).T
            pts_world = (pose.astype(np.float64) @ pts_zed.T).T[:, :3]

            all_pts_list.append(pts_world[valid_mask])
            all_clr_list.append(colors[valid_mask])
            all_conf_list.append(conf.reshape(-1)[valid_mask])

            if idx % 10 == 0:
                logging.info(f"Processed frame {idx}")

    all_pts = np.vstack(all_pts_list) if all_pts_list else np.array([]).reshape(0, 3)
    all_clr = np.vstack(all_clr_list) if all_clr_list else np.array([]).reshape(0, 3)
    all_conf = np.concatenate(all_conf_list) if all_conf_list else np.array([])

    logging.info(f"Total points: {len(all_pts)}, confidence range: [{all_conf.min():.3f}, {all_conf.max():.3f}]")

    raw_data = {
        'points': all_pts,
        'colors': all_clr,
        'confidence': all_conf
    }

    return raw_data


def main_demo(model_dir, tmp_dir):
    """Create Gradio demo"""

    raw_data_state = {'data': None, 'tmp_dir': tmp_dir}

    def run_inference(svo_path, scale, frame_skip, max_frames,
                      min_depth, max_depth, depth_edge_threshold):
        if svo_path is None:
            return None, "Please upload an SVO file"

        try:
            svo_path = svo_path.name if hasattr(svo_path, 'name') else svo_path
            logging.info(f"Processing SVO: {svo_path}")

            raw_data = process_svo(
                svo_path, model_dir, scale, frame_skip, max_frames,
                min_depth, max_depth, depth_edge_threshold, tmp_dir
            )
            raw_data_state['data'] = raw_data

            initial_conf = 0.5
            pts, clr = filter_pointcloud_by_confidence(
                raw_data['points'], raw_data['colors'],
                raw_data['confidence'], initial_conf
            )

            output_path = os.path.join(tmp_dir, "initial_filtered.glb")
            save_glb(pts, clr, output_path)

            conf_range = raw_data['confidence']
            return output_path, f"Done! {len(pts)} points (conf range: [{conf_range.min():.3f}, {conf_range.max():.3f}])"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"

    def updateVisualization(conf_threshold, num_points):
        if raw_data_state['data'] is None:
            return None

        try:
            pts, clr = filter_pointcloud_by_confidence(
                raw_data_state['data']['points'],
                raw_data_state['data']['colors'],
                raw_data_state['data']['confidence'],
                conf_threshold
            )

            if len(pts) > num_points:
                indices = np.random.choice(len(pts), num_points, replace=False)
                pts = pts[indices]
                clr = clr[indices]

            output_path = os.path.join(tmp_dir, f"filtered_{conf_threshold:.2f}.glb")
            save_glb(pts, clr, output_path)

            return output_path

        except Exception as e:
            return None

    with gr.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
                   title="Fast-FoundationStereo Confidence Filter") as demo:
        gr.Markdown("## Fast-FoundationStereo 置信度滤波 demo ✧(≖ ◡ ≖✿)")

        with gr.Row():
            with gr.Column(scale=1):
                svo_input = gr.File(label="上传 SVO 文件", file_types=[".svo", ".svo2"])

                gr.Markdown("### 推理参数")
                scale = gr.Slider(value=0.5, minimum=0.25, maximum=1.0, step=0.25,
                                 label="图像缩放", info="越小越快")
                frame_skip = gr.Slider(value=5, minimum=0, maximum=20, step=1,
                                      label="跳帧", info="每N帧处理一帧")
                max_frames = gr.Slider(value=50, minimum=10, maximum=200, step=10,
                                      label="最大帧数")
                min_depth = gr.Number(value=0.5, label="最小深度 (m)")
                max_depth = gr.Number(value=15.0, label="最大深度 (m)")
                depth_edge_threshold = gr.Slider(value=0.1, minimum=0.0, maximum=1.0, step=0.01,
                                                label="深度边缘阈值")

                run_btn = gr.Button("开始处理", variant="primary")

            with gr.Column(scale=2):
                status_text = gr.Textbox(label="状态", lines=2)
                model_view = gr.Model3D(height=600, clear_color=(0., 0., 0., 0.3),
                                        label="3D 点云 (置信度滤波后)")

                gr.Markdown("### 置信度滑块")
                gr.Markdown("调整下方滑块可实时过滤低置信度点")
                conf_threshold = gr.Slider(value=0.3, minimum=0.0, maximum=1.0, step=0.01,
                                          label="置信度阈值", info="低于此值的点会被过滤")
                num_points = gr.Slider(value=500000, minimum=10000, maximum=2000000, step=10000,
                                      label="显示点数上限")

        run_btn.click(fn=run_inference,
                     inputs=[svo_input, scale, frame_skip, max_frames,
                            min_depth, max_depth, depth_edge_threshold],
                     outputs=[model_view, status_text])

        conf_threshold.release(fn=updateVisualization,
                             inputs=[conf_threshold, num_points],
                             outputs=model_view)

        num_points.release(fn=updateVisualization,
                         inputs=[conf_threshold, num_points],
                         outputs=model_view)

    return demo


def main():
    parser = argparse.ArgumentParser(description='Fast-FoundationStereo Confidence Filter App')
    parser.add_argument('--svo', type=str, default=None,
                       help='SVO file path (optional, can also upload in UI)')
    parser.add_argument('--model_dir', type=str,
                       default=os.path.join(code_dir, 'weights/model_best_bp2_serialize.pth'),
                       help='FFS model path')
    parser.add_argument('--server_port', type=int, default=7860,
                       help='Gradio server port')
    parser.add_argument('--share', action='store_true',
                       help='Create public link')
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix='ffs_conf_')
    logging.info(f"Using temp dir: {tmp_dir}")

    demo = main_demo(args.model_dir, tmp_dir)
    demo.launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
