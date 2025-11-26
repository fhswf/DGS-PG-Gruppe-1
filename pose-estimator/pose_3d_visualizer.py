"""
3D Pose Visualizer: Visualize 3D pose estimation results

This script provides functions to plot 3D poses from the Pose3DConverter output.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
from typing import Union, Optional, List, Tuple


# COCO WholeBody skeleton connections for visualization
BODY_CONNECTIONS = [
    # Body
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (0, 5), (0, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hip
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (15,19), (19,17), (19,18), # Right Foot
    (16,22), (22,20), (22,21) # Left Foot
]

# Face connections (simplified)
FACE_CONNECTIONS = [
    (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37),
    (37, 38), (38, 39)
]

# Hand connections (simplified - just fingers)
LEFT_HAND_CONNECTIONS = [
    (91,92),(92,93),(93,94),(94,95), # Thumb
    (91, 96), (96, 97), (97, 98), (98,99),  # Index
    (91, 100), (100, 101), (101, 102), (102,103),  # Middle
    (91, 104), (104, 105), (105, 106), (106,107),  # Ring
    (91, 108), (108, 109), (109, 110), (110,111),  # Pinky
]

RIGHT_HAND_CONNECTIONS = [
    (112, 113), (113, 114), (114, 115), (115,116),  # Thumb
    (112, 117), (117, 118), (118, 119), (119,120),  # Index
    (112, 121), (121, 122), (122, 123), (123,124),  # Middle
    (112, 125), (125, 126), (126, 127), (127,128),  # Ring
    (112, 129), (129, 130), (130, 131), (131,132),  # Pinky
]


def plot_3d_pose_from_json(
    json_path: Union[str, Path],
    frame_idx: int = 0,
    view: str = 'combined',
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    confidence_threshold: float = 0.3,
    figsize: Tuple[int, int] = (12, 10),
    z_scale: float = 5.0  # Skalierungsfaktor für Z-Koordinaten
):
    """
    Plot 3D pose from JSON file.
    
    Args:
        json_path: Path to 3D pose JSON file
        frame_idx: Frame index to plot
        view: Which view to plot ('left_3d', 'right_3d', or 'combined')
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        confidence_threshold: Minimum confidence for displaying keypoints
        figsize: Figure size (width, height)
        z_scale: Scaling factor for Z coordinates (default: 100.0)
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if frame_idx >= len(data):
        raise ValueError(f"Frame {frame_idx} not found. Only {len(data)} frames available.")
    
    frame_data = data[frame_idx]
    view_data = frame_data.get(view, frame_data.get('combined_3d'))
    
    if view_data is None:
        raise ValueError(f"View '{view}' not found in frame data")
    
    # Extract 3D keypoints and scores
    keypoints_3d = np.array(view_data['keypoints_3d'])
    scores_3d = np.array(view_data['scores_3d'])
    
    # WICHTIG: Z-Koordinaten skalieren für bessere Sichtbarkeit
    keypoints_3d_scaled = keypoints_3d.copy()
    keypoints_3d_scaled[:, :, 2] *= z_scale
    
    # Debug: Zeige Z-Koordinaten-Bereich
    z_coords = keypoints_3d[:, :, 2].flatten()
    z_coords_valid = z_coords[~np.isnan(z_coords)]
    if len(z_coords_valid) > 0:
        print(f"Z-Koordinaten Bereich: min={z_coords_valid.min():.4f}, max={z_coords_valid.max():.4f}")
        print(f"Nach Skalierung ({z_scale}x): min={z_coords_valid.min()*z_scale:.2f}, max={z_coords_valid.max()*z_scale:.2f}")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each person
    for person_idx in range(len(keypoints_3d_scaled)):
        kpts = keypoints_3d_scaled[person_idx]
        scores = scores_3d[person_idx]
        
        # Filter by confidence
        valid_mask = scores > confidence_threshold
        
        # Plot body skeleton
        _plot_skeleton_3d(ax, kpts, scores, BODY_CONNECTIONS, 
                         color='blue', linewidth=2, label='Body' if person_idx == 0 else None)
        
        # Plot face (optional, lighter color)
        _plot_skeleton_3d(ax, kpts, scores, FACE_CONNECTIONS, 
                         color='green', linewidth=1, alpha=0.5)
        
        # Plot hands (optional)
        _plot_skeleton_3d(ax, kpts, scores, LEFT_HAND_CONNECTIONS, 
                         color='red', linewidth=1, alpha=0.7, label='Left Hand' if person_idx == 0 else None)
        _plot_skeleton_3d(ax, kpts, scores, RIGHT_HAND_CONNECTIONS, 
                         color='orange', linewidth=1, alpha=0.7, label='Right Hand' if person_idx == 0 else None)
        
        # Plot keypoints
        valid_kpts = kpts[valid_mask]
        if len(valid_kpts) > 0:
            ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                      c='black', marker='o', s=20, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel(f'Z (Depth, scaled {z_scale}x)')
    ax.set_title(f'3D Pose Estimation - Frame {frame_idx} ({view})')
    
    # WICHTIG: Invertiere Y-Achse (Bild-Koordinaten haben Y nach unten)
    ax.invert_yaxis()
    
    # Set aspect ratio - NICHT equal für bessere Z-Sichtbarkeit
    # _set_axes_equal(ax)  # Auskommentiert, damit Z sichtbar bleibt
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D pose plot to: {output_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def _plot_skeleton_3d(ax, keypoints, scores, connections, color='blue', 
                      linewidth=2, alpha=1.0, label=None):
    """Helper function to plot skeleton connections in 3D"""
    for i, (start_idx, end_idx) in enumerate(connections):
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
                start = keypoints[start_idx]
                end = keypoints[end_idx]
                
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]], 
                       color=color, linewidth=linewidth, alpha=alpha,
                       label=label if i == 0 else None)


def _set_axes_equal(ax):
    """Set 3D plot axes to equal scale"""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def plot_multiple_views(
    json_path: Union[str, Path],
    frame_idx: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    z_scale: float = 100.0  # Skalierungsfaktor für Z-Koordinaten
):
    """
    Plot all three views (left, right, combined) side by side.
    
    Args:
        json_path: Path to 3D pose JSON file
        frame_idx: Frame index to plot
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        z_scale: Scaling factor for Z coordinates
    """
    fig = plt.figure(figsize=(18, 6))
    
    views = ['left_3d', 'right_3d', 'combined_3d']
    titles = ['Left View', 'Right View', 'Combined View']
    
    # Load data once
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frame_data = data[frame_idx]
    
    for idx, (view, title) in enumerate(zip(views, titles)):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        view_data = frame_data.get(view)
        if view_data is None:
            continue
        
        keypoints_3d = np.array(view_data['keypoints_3d'])
        scores_3d = np.array(view_data['scores_3d'])
        
        # Z-Koordinaten skalieren
        keypoints_3d[:, :, 2] *= z_scale
        
        # Plot each person
        for person_idx in range(len(keypoints_3d)):
            kpts = keypoints_3d[person_idx]
            scores = scores_3d[person_idx]
            
            # Plot body skeleton
            _plot_skeleton_3d(ax, kpts, scores, BODY_CONNECTIONS, color='blue', linewidth=2)
            
            # Plot keypoints
            valid_mask = scores > 0.3
            valid_kpts = kpts[valid_mask]
            if len(valid_kpts) > 0:
                ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2], 
                          c='black', marker='o', s=20, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(f'Z (scaled {z_scale}x)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        # _set_axes_equal(ax)  # Auskommentiert für bessere Z-Sichtbarkeit
    
    plt.suptitle(f'3D Pose Estimation - Frame {frame_idx}', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-view plot to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_3d_animation_frames(
    json_path: Union[str, Path],
    output_dir: Union[str, Path],
    view: str = 'combined',
    max_frames: Optional[int] = None,
    z_scale: float = 100.0
):
    """
    Create individual frames for 3D pose animation.
    
    Args:
        json_path: Path to 3D pose JSON file
        output_dir: Directory to save frames
        view: Which view to plot
        max_frames: Maximum number of frames to process
        z_scale: Scaling factor for Z coordinates
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    num_frames = min(len(data), max_frames) if max_frames else len(data)
    
    print(f"Creating {num_frames} animation frames...")
    
    for frame_idx in range(num_frames):
        output_path = output_dir / f"frame_{frame_idx:05d}.png"
        plot_3d_pose_from_json(
            json_path, 
            frame_idx=frame_idx,
            view=view,
            output_path=output_path,
            show_plot=False,
            z_scale=z_scale
        )
        
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}/{num_frames}")
    
    print(f"All frames saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Plot single frame mit Z-Skalierung
    plot_3d_pose_from_json(
        "poses_3d.json",
        frame_idx=0,
        view='combined',
        output_path="pose_3d_plot.png",
        show_plot=True,
        z_scale=100.0  # Passe diesen Wert an für bessere Tiefendarstellung
    )
    
    # Plot all three views
    plot_multiple_views(
        "poses_3d.json",
        frame_idx=0,
        output_path="pose_3d_multi_view.png",
        show_plot=True,
        z_scale=100.0
    )