"""
Visualize CALVIN dataset from Zarr file.

This script loads the converted Zarr file and visualizes:
- RGB static crops (160x160)
- Downsampled point clouds (2048 points, XYZRGB)
- Optionally loads PLY files if they exist
"""

import os
import sys
import zarr
import numpy as np
import argparse
from pathlib import Path
from termcolor import cprint

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.visualize_point_clouds import visualize_point_clouds
except ImportError as e:
    cprint(f"Error: Could not import visualize_point_clouds from utils: {e}", "red")
    cprint("Make sure you're running from the project root or utils/visualize_point_clouds.py exists", "red")
    raise


def load_ply_file(ply_path):
    """
    Load a PLY file and return point cloud and colors.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        points: (N, 3) array of XYZ coordinates
        colors: (N, 3) array of RGB colors in [0, 1]
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        return points, colors
    except ImportError:
        cprint("Open3D not available. Cannot load PLY files.", "yellow")
        return None, None
    except Exception as e:
        cprint(f"Error loading PLY file {ply_path}: {e}", "red")
        return None, None


def visualize_frame_from_zarr(zarr_path, frame_idx, ply_dir=None):
    """
    Visualize a single frame from the Zarr dataset.
    
    Args:
        zarr_path: Path to Zarr file
        frame_idx: Index of frame to visualize
        ply_dir: Optional directory containing PLY files to load instead
    """
    cprint(f'Loading frame {frame_idx} from Zarr: {zarr_path}', 'green')
    zarr_root = zarr.open(zarr_path, mode='r')
    zarr_data = zarr_root['data']
    
    total_frames = zarr_data['img'].shape[0]
    if frame_idx >= total_frames:
        cprint(f'Error: Frame {frame_idx} exceeds total frames ({total_frames})', 'red')
        return
    
    # Load data for this frame
    img = zarr_data['img'][frame_idx]  # (160, 160, 3) uint8
    point_cloud = zarr_data['point_cloud'][frame_idx]  # (2048, 6) float64 [XYZRGB]
    depth = zarr_data['depth'][frame_idx]  # (160, 160) float64
    
    # Extract point cloud components
    points_xyz = point_cloud[:, :3]  # (2048, 3)
    points_rgb = point_cloud[:, 3:]  # (2048, 3) in [0, 1]
    
    # Split into static and gripper (1024 points each)
    static_pts = points_xyz[:1024]
    gripper_pts = points_xyz[1024:]
    static_rgb = points_rgb[:1024]
    gripper_rgb = points_rgb[1024:]
    
    # Check if PLY files exist and should be loaded
    if ply_dir and os.path.exists(ply_dir):
        static_ply_path = os.path.join(ply_dir, f'frame_{frame_idx:06d}_static_after_downsampling.ply')
        gripper_ply_path = os.path.join(ply_dir, f'frame_{frame_idx:06d}_gripper_after_downsampling.ply')
        
        if os.path.exists(static_ply_path) and os.path.exists(gripper_ply_path):
            cprint(f'Loading PLY files from {ply_dir}', 'cyan')
            static_pts_ply, static_rgb_ply = load_ply_file(static_ply_path)
            gripper_pts_ply, gripper_rgb_ply = load_ply_file(gripper_ply_path)
            
            if static_pts_ply is not None:
                static_pts = static_pts_ply
                static_rgb = static_rgb_ply
            if gripper_pts_ply is not None:
                gripper_pts = gripper_pts_ply
                gripper_rgb = gripper_rgb_ply
    
    # Visualize using the utility function
    cprint(f'Visualizing frame {frame_idx}...', 'cyan')
    
    # Visualize point clouds with their RGB colors
    visualize_point_clouds(
        static_pts,
        gripper_pts,
        static_rgb=static_rgb,  # Point cloud colors (1024, 3) for 3D visualization
        gripper_rgb=gripper_rgb,  # Point cloud colors (1024, 3) for 3D visualization
        static_depth=depth,
        gripper_depth=None,
        frame_idx=frame_idx,
        title="CALVIN Point Cloud (From Zarr)",
        save_path=None,  # Display interactively or save based on environment
        stage="from_zarr"
    )
    
    # Also display the RGB image separately using matplotlib
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        
        img_normalized = img.astype(np.float32) / 255.0
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Static RGB crop
        axes[0].imshow(img_normalized)
        axes[0].set_title(f'Static RGB Crop (Frame {frame_idx})')
        axes[0].axis('off')
        
        # Depth
        im = axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title(f'Depth (Frame {frame_idx})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.suptitle(f'Frame {frame_idx} - RGB and Depth from Zarr', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        cprint("Matplotlib not available. Skipping RGB image display.", "yellow")


def visualize_multiple_frames(zarr_path, frame_indices, ply_dir=None, save_dir=None):
    """
    Visualize multiple frames from the Zarr dataset.
    
    Args:
        zarr_path: Path to Zarr file
        frame_indices: List of frame indices to visualize
        ply_dir: Optional directory containing PLY files
        save_dir: Optional directory to save visualizations
    """
    zarr_root = zarr.open(zarr_path, mode='r')
    zarr_data = zarr_root['data']
    total_frames = zarr_data['img'].shape[0]
    
    cprint(f'Total frames in dataset: {total_frames}', 'cyan')
    
    for frame_idx in frame_indices:
        if frame_idx >= total_frames:
            cprint(f'Warning: Frame {frame_idx} exceeds total frames ({total_frames}), skipping', 'yellow')
            continue
        
        visualize_frame_from_zarr(zarr_path, frame_idx, ply_dir=ply_dir)
        
        if save_dir:
            # The visualize_point_clouds function will save if save_path is provided
            visualize_frame_from_zarr_with_save(zarr_path, frame_idx, save_dir, ply_dir=ply_dir)


def visualize_frame_from_zarr_with_save(zarr_path, frame_idx, save_dir, ply_dir=None):
    """
    Visualize and save a frame from Zarr.
    """
    zarr_root = zarr.open(zarr_path, mode='r')
    zarr_data = zarr_root['data']
    
    # Load data
    img = zarr_data['img'][frame_idx]
    point_cloud = zarr_data['point_cloud'][frame_idx]
    depth = zarr_data['depth'][frame_idx]
    
    # Extract point cloud components
    points_xyz = point_cloud[:, :3]
    points_rgb = point_cloud[:, 3:]
    
    # Split into static and gripper
    static_pts = points_xyz[:1024]
    gripper_pts = points_xyz[1024:]
    static_rgb = points_rgb[:1024]
    gripper_rgb = points_rgb[1024:]
    
    # Check for PLY files
    if ply_dir and os.path.exists(ply_dir):
        static_ply_path = os.path.join(ply_dir, f'frame_{frame_idx:06d}_static_after_downsampling.ply')
        gripper_ply_path = os.path.join(ply_dir, f'frame_{frame_idx:06d}_gripper_after_downsampling.ply')
        
        if os.path.exists(static_ply_path) and os.path.exists(gripper_ply_path):
            static_pts_ply, static_rgb_ply = load_ply_file(static_ply_path)
            gripper_pts_ply, gripper_rgb_ply = load_ply_file(gripper_ply_path)
            
            if static_pts_ply is not None:
                static_pts = static_pts_ply
                static_rgb = static_rgb_ply
            if gripper_pts_ply is not None:
                gripper_pts = gripper_pts_ply
                gripper_rgb = gripper_rgb_ply
    
    # Visualize and save
    visualize_point_clouds(
        static_pts,
        gripper_pts,
        static_rgb=static_rgb,
        gripper_rgb=gripper_rgb,
        static_depth=depth,
        gripper_depth=None,
        frame_idx=frame_idx,
        title="CALVIN Point Cloud (From Zarr)",
        save_path=save_dir,
        stage="from_zarr"
    )
    
    # Also save RGB image separately
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        img_normalized = img.astype(np.float32) / 255.0
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Static RGB crop
        axes[0].imshow(img_normalized)
        axes[0].set_title(f'Static RGB Crop (Frame {frame_idx})')
        axes[0].axis('off')
        
        # Depth
        im = axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title(f'Depth (Frame {frame_idx})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.suptitle(f'Frame {frame_idx} - RGB and Depth from Zarr', fontsize=14)
        plt.tight_layout()
        
        img_path = os.path.join(save_dir, f'frame_{frame_idx:06d}_rgb_depth.png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        cprint(f"Saved RGB/Depth image to {img_path}", "green")
        plt.close()
        
    except ImportError:
        cprint("Matplotlib not available. Skipping RGB image save.", "yellow")


def main():
    parser = argparse.ArgumentParser(description='Visualize CALVIN Zarr dataset')
    parser.add_argument('zarr_path', type=str,
                        help='Path to the Zarr file')
    parser.add_argument('--frame_indices', type=int, nargs='+', default=[0],
                        help='Frame indices to visualize (default: [0])')
    parser.add_argument('--ply_dir', type=str, default=None,
                        help='Directory containing PLY files to load (optional)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations (if None, displays interactively)')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to visualize (starting from first frame index)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame index (used with --num_frames)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        cprint(f'Error: Zarr file not found: {args.zarr_path}', 'red')
        return
    
    # Determine which frames to visualize
    if args.num_frames is not None:
        frame_indices = list(range(args.start_frame, args.start_frame + args.num_frames))
    else:
        frame_indices = args.frame_indices
    
    cprint(f'Visualizing frames: {frame_indices}', 'green')
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        cprint(f'Saving visualizations to: {args.save_dir}', 'green')
    
    visualize_multiple_frames(
        args.zarr_path,
        frame_indices,
        ply_dir=args.ply_dir,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
