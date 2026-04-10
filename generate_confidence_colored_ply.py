# -*- coding: utf-8 -*-
"""
Generate a colored point cloud based on confidence values.
White = high confidence, Black = low confidence

Usage:
    python generate_confidence_colored_ply.py --ply "path/to/pointcloud.ply" --conf "path/to/confidence.npy"
    python generate_confidence_colored_ply.py --ply "path/to/pointcloud.ply" --conf "path/to/confidence.npy" --output "output.ply"
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d

code_dir = os.path.dirname(os.path.realpath(__file__))


def confidence_to_rgb(confidence, invert=False):
    """Convert confidence values to RGB colors (white=high, gray=low).

    Args:
        confidence: (N,) array of confidence values
        invert: If True, white=low confidence, gray=high confidence

    Returns:
        (N, 3) array of RGB colors in range [0, 1]
        - confidence=0 -> gray (0.5, 0.5, 0.5)
        - confidence=1 -> white (1.0, 1.0, 1.0)
    """
    conf_min = confidence.min()
    conf_max = confidence.max()

    if conf_max - conf_min < 1e-6:
        normalized = np.ones_like(confidence)
    else:
        normalized = (confidence - conf_min) / (conf_max - conf_min)

    if invert:
        normalized = 1.0 - normalized

    gray_to_white = 0.5 + 0.5 * normalized
    rgb = np.stack([gray_to_white, gray_to_white, gray_to_white], axis=1)
    return rgb.astype(np.float64)


def generate_confidence_colored_ply(ply_path, conf_path=None, output_path=None, invert=False):
    """Generate a new PLY with confidence-based coloring.

    Args:
        ply_path: Path to input PLY file
        conf_path: Path to confidence numpy file (default: *_confidence.npy in same dir)
        output_path: Path to output PLY file (default: adds '_confcolored' suffix)
        invert: If True, white=low confidence, black=high confidence

    Returns:
        Path to generated PLY file
    """
    if conf_path is None:
        conf_path = ply_path.replace('.ply', '_confidence.npy')

    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found: {ply_path}")
        return None

    if not os.path.exists(conf_path):
        print(f"Error: Confidence file not found: {conf_path}")
        return None

    print(f"Loading PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points")

    print(f"Loading confidence: {conf_path}")
    confidence = np.load(conf_path)
    print(f"Loaded confidence array with {len(confidence)} values")

    if len(confidence) != len(points):
        print(f"Warning: confidence size ({len(confidence)}) != points size ({len(points)})")
        if len(confidence) > len(points):
            confidence = confidence[:len(points)]
        else:
            confidence = np.pad(confidence, (0, len(points) - len(confidence)))
            print(f"Warning: padded confidence to {len(confidence)}")

    print(f"Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")

    print("Generating RGB colors (white=high, black=low)...")
    colors = confidence_to_rgb(confidence, invert=invert)
    print(f"Color range: [{colors.min():.4f}, {colors.max():.4f}]")

    pcd.colors = o3d.utility.Vector3dVector(colors)

    if output_path is None:
        name = os.path.splitext(os.path.basename(ply_path))[0]
        dir_name = os.path.dirname(ply_path) or '.'
        if invert:
            output_path = os.path.join(dir_name, f"{name}_confcolored_inverted.ply")
        else:
            output_path = os.path.join(dir_name, f"{name}_confcolored.ply")

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate confidence-colored PLY')
    parser.add_argument('--ply', type=str, required=True, help='Input PLY file')
    parser.add_argument('--conf', type=str, default=None, help='Confidence numpy file')
    parser.add_argument('--output', type=str, default=None, help='Output PLY file')
    parser.add_argument('--invert', action='store_true', help='Invert colors (white=low, black=high)')
    args = parser.parse_args()

    result = generate_confidence_colored_ply(args.ply, args.conf, args.output, args.invert)
    if result:
        print(f"\nDone! Generated: {result}")


if __name__ == "__main__":
    main()
