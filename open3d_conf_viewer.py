# -*- coding: utf-8 -*-
"""
Open3D-based Point Cloud Viewer with Confidence Slider

Usage:
    python open3d_conf_viewer.py --ply "path/to/pointcloud.ply"
    python open3d_conf_viewer.py --ply "path/to/pointcloud.ply" --conf "path/to/confidence.npy"
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import time

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, code_dir)


class ConfidenceSliderApp:
    def __init__(self, ply_path, conf_path=None, window_name="Confidence Filter Viewer"):
        self.window_name = window_name
        self.ply_path = ply_path
        self.conf_path = conf_path

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_name, width=1280, height=720)

        self.pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Loaded {len(self.pcd.points)} points from {ply_path}")

        self.points = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if len(self.pcd.colors) > 0 else None

        if conf_path and os.path.exists(conf_path):
            self.confidence = np.load(conf_path)
        elif hasattr(self.pcd, 'point_attributes') and 'confidence' in self.pcd.point_attributes:
            self.confidence = np.asarray(self.pcd.point_attributes['confidence'])
        else:
            self.confidence = np.ones(len(self.points))
            print("No confidence data found, using uniform confidence")

        if len(self.confidence) != len(self.points):
            print(f"Warning: confidence size ({len(self.confidence)}) != points size ({len(self.points)})")
            if len(self.confidence) < len(self.points):
                self.confidence = np.ones(len(self.points))
            else:
                self.confidence = self.confidence[:len(self.points)]

        self.conf_min = self.confidence.min()
        self.conf_max = self.confidence.max()
        print(f"Confidence range: [{self.conf_min:.4f}, {self.conf_max:.4f}]")

        self.current_threshold = 0.0
        self.display_pcd = o3d.geometry.PointCloud()

        self._setup_ui()

        self.vis.add_geometry(self.display_pcd)
        self._update_display()

        self.vis.register_key_callback(ord('Q'), self.close)

    def _setup_ui(self):
        self.param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

    def _update_display(self, threshold=None):
        if threshold is None:
            threshold = self.current_threshold
        else:
            self.current_threshold = threshold

        valid_mask = self.confidence > threshold
        filtered_points = self.points[valid_mask]
        filtered_colors = self.colors[valid_mask] if self.colors is not None else None

        self.display_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        if filtered_colors is not None:
            self.display_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        else:
            self.display_pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(filtered_points), 3))
            )

        self.vis.update_geometry(self.display_pcd)

    def close(self, vis=None):
        self.vis.destroy_window()
        return False

    def run(self):
        print(f"\n{'='*60}")
        print("Open3D Confidence Filter Viewer")
        print(f"{'='*60}")
        print(f"Points: {len(self.pcd.points)}")
        print(f"Confidence range: [{self.conf_min:.4f}, {self.conf_max:.4f}]")
        print(f"\nControls:")
        print("  Q - Quit")
        print("  R - Reset view")
        print("  +/- - Adjust threshold")
        print("  UP/DOWN - Increase/decrease threshold by 0.05")
        print(f"\nInitial threshold: {self.current_threshold:.4f}")
        print(f"{'='*60}\n")

        def increase_threshold(vis):
            self.current_threshold = min(self.conf_max, self.current_threshold + 0.05)
            print(f"Threshold: {self.current_threshold:.4f} ({self._get_valid_ratio()*100:.1f}% points)")
            self._update_display()
            return False

        def decrease_threshold(vis):
            self.current_threshold = max(self.conf_min, self.current_threshold - 0.05)
            print(f"Threshold: {self.current_threshold:.4f} ({self._get_valid_ratio()*100:.1f}% points)")
            self._update_display()
            return False

        def reset_view(vis):
            self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.param)
            return False

        def quit_app(vis):
            self.close()
            return False

        self.vis.register_key_callback(ord('R'), reset_view)
        self.vis.register_key_callback(ord('+'), increase_threshold)
        self.vis.register_key_callback(ord('='), increase_threshold)
        self.vis.register_key_callback(ord('-'), decrease_threshold)
        self.vis.register_key_callback(ord('_'), decrease_threshold)
        self.vis.register_key_callback(ord('Q'), quit_app)

        self.vis.run()


class InteractiveConfidenceApp:
    def __init__(self, ply_path, conf_path=None):
        self.ply_path = ply_path
        self.conf_path = conf_path

        self.pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Loaded {len(self.pcd.points)} points")

        self.points = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if len(self.pcd.colors) > 0 else None

        if conf_path and os.path.exists(conf_path):
            self.confidence = np.load(conf_path)
        else:
            self.confidence = np.ones(len(self.points))

        if len(self.confidence) != len(self.points):
            print(f"Warning: resizing confidence array")
            self.confidence = np.resize(self.confidence, len(self.points))

        self.conf_min = float(self.confidence.min())
        self.conf_max = float(self.confidence.max())
        self.current_threshold = self.conf_min

        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window(
            "Confidence Filter - Open3D",
            width=1280, height=720
        )

        self.pcd_vis = o3d.visualization.O3DVisualizer()
        self.pcd_vis.set_background((0.1, 0.1, 0.1, 1.0))
        self._update_geometry()

        self.window.add_child(self.pcd_vis)

        em = self.window.theme.font_size

        slider_widget = o3d.visualization.gui.Slider(o3d.visualization.gui.Slider.DOUBLE)
        slider_widget.set_limits(self.conf_min, self.conf_max)
        slider_widget.set_value(self.conf_min)
        slider_widget.set_on_value_changed(self._on_slider_changed)

        label = o3d.visualization.gui.Label(f"Confidence Threshold: {self.current_threshold:.4f}")

        panel = o3d.visualization.gui.Vert(0, o3d.visualization.gui.Margins(em, em))
        panel.add_child(label)
        panel.add_child(slider_widget)

        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.slider_widget = slider_widget
        self.label_widget = label

    def _on_layout(self, ctx):
        panel_width = 300
        self.window.set_child_rect(
            self.pcd_vis,
            o3d.visualization.gui.Rect(0, 0, ctx.size.width - panel_width, ctx.size.height)
        )

    def _on_slider_changed(self, new_val):
        self.current_threshold = new_val
        self.label_widget.text = f"Confidence Threshold: {self.current_threshold:.4f}"
        self._update_geometry()

    def _update_geometry(self):
        valid_mask = self.confidence > self.current_threshold
        n_valid = valid_mask.sum()
        print(f"Threshold: {self.current_threshold:.4f} -> {n_valid}/{len(self.points)} points ({n_valid/len(self.points)*100:.1f}%)")

        filtered_points = self.points[valid_mask]
        filtered_colors = self.colors[valid_mask] if self.colors is not None else None

        self.pcd_vis.clear_geometry()
        if len(filtered_points) > 0:
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            if filtered_colors is not None:
                temp_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
            else:
                temp_pcd.colors = o3d.utility.Vector3dVector(np.ones((len(filtered_points), 3)))
            self.pcd_vis.add_geometry(temp_pcd)
            self.pcd_vis.reset_camera_to_default()

    def _on_close(self):
        self.app.quit()
        return True

    def run(self):
        print(f"\n{'='*60}")
        print("Open3D Confidence Slider App")
        print(f"{'='*60}")
        print(f"Use the slider to adjust confidence threshold")
        print(f"Close window to quit")
        print(f"{'='*60}\n")
        self.app.run()


def main():
    parser = argparse.ArgumentParser(description='Open3D Confidence Point Cloud Viewer')
    parser.add_argument('--ply', type=str, required=True, help='Path to PLY file')
    parser.add_argument('--conf', type=str, default=None,
                       help='Path to confidence numpy file (.npy)')
    parser.add_argument('--use_new_api', action='store_true',
                       help='Use new Open3D visualizer API')
    args = parser.parse_args()

    if not os.path.exists(args.ply):
        print(f"Error: PLY file not found: {args.ply}")
        return

    if args.use_new_api:
        app = InteractiveConfidenceApp(args.ply, args.conf)
        app.run()
    else:
        app = ConfidenceSliderApp(args.ply, args.conf)
        app.run()


if __name__ == "__main__":
    main()
