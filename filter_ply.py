# -*- coding: utf-8 -*-
"""
Apply filtering to existing PLY files (for iterative refinement)

Recommended Workflow:
    1. Run svo_to_ply.py with recommended parameters to generate _02_temporal.ply
    2. Apply various filters using this script to find best parameters
    3. Use best filter configuration as new default

Usage:
    python filter_ply.py "input.ply"
    python filter_ply.py "input.ply" --stat_std 0.2
"""
import argparse
import logging
import numpy as np
import open3d as o3d
import os
import time

def set_logging_format():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def toOpen3dCloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        if colors.max() > 1:
            colors = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    set_logging_format()
    parser = argparse.ArgumentParser(description='Apply filters to existing PLY')
    parser.add_argument('input', help='Input PLY file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--stat_k', type=int, default=100, help='Statistical outlier k')
    parser.add_argument('--stat_std', type=float, default=0.5, help='Statistical outlier std')
    parser.add_argument('--radius_nb', type=int, default=16, help='Radius outlier nb_points')
    parser.add_argument('--radius_r', type=float, default=0.03, help='Radius outlier radius')
    parser.add_argument('--dbscan_eps', type=float, default=0.05, help='DBSCAN eps')
    parser.add_argument('--dbscan_min', type=int, default=10, help='DBSCAN min_pts')
    args = parser.parse_args()

    input_path = args.input
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(input_path))[0]

    logging.info(f"Loading {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255 if len(pcd.colors) > 0 else None
    logging.info(f"Loaded {len(pts)} points")

    def save_pcd(pcd, suffix):
        out = os.path.join(output_dir, f"{name}_{suffix}.ply")
        o3d.io.write_point_cloud(out, pcd)
        logging.info(f"Saved ({len(np.asarray(pcd.points))} pts): {out}")

    pcd_base = pcd
    for std in [0.2]:
        pcd_in, _ = pcd_base.remove_statistical_outlier(
            nb_neighbors=args.stat_k, std_ratio=std)
        n = len(np.asarray(pcd_in.points))
        suffix = f"stat_k{args.stat_k}_std{std}"
        save_pcd(pcd_in, suffix)
        logging.info(f"  Statistical (k={args.stat_k}, std={std}): {n} pts")

    pcd_stat = pcd_in
    for r, nb in [(0.02, 16), (0.03, 16), (0.03, 10), (0.03, 20)]:
        pcd_in, _ = pcd_stat.remove_radius_outlier(nb_points=nb, radius=r)
        n = len(np.asarray(pcd_in.points))
        suffix = f"stat_k{args.stat_k}_std0.2_radius_r{r}_nb{nb}"
        save_pcd(pcd_in, suffix)
        logging.info(f"  Radius (r={r}, nb={nb}): {n} pts")

    for eps, min_pts in [(0.05, 10), (0.08, 15), (0.1, 20), (0.05, 5)]:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            labels = np.array(pcd_stat.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
        if len(labels) > 0 and labels.max() >= 0:
            max_label = labels.max()
            cluster_sizes = [(labels == i).sum() for i in range(max_label + 1)]
            valid_mask = labels >= 0
            for i, sz in enumerate(cluster_sizes):
                if sz < min_pts:
                    valid_mask[labels == i] = False
            pcd_filtered = pcd_stat.select_by_index(np.where(valid_mask)[0])
            n = len(np.asarray(pcd_filtered.points))
            suffix = f"stat_k{args.stat_k}_std0.2_dbscan_eps{eps}_min{min_pts}"
            save_pcd(pcd_filtered, suffix)
            logging.info(f"  DBSCAN (eps={eps}, min={min_pts}): {n} pts, {max_label+1} clusters")

if __name__ == '__main__':
    main()
