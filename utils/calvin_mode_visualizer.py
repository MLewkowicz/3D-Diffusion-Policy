"""
Visualize CALVIN trajectories per task as interactive 3D plots.

Loads the CALVIN debug dataset (raw or converted zarr), extracts end-effector
trajectories for each task, and saves interactive HTML plots (Plotly) so you
can rotate/zoom to inspect what modes are captured.

Supports:
  - Raw CALVIN root dir (training/ or validation/ with lang_annotations):
    one plot per task with all trajectories for that task.
  - Converted DP3 zarr: one plot with all episodes (no task labels unless
    you provide a task mapping).
"""

import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from termcolor import cprint

# Optional: zarr for converted dataset
try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

# Optional: Plotly for interactive HTML
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_trajectories_from_calvin_root(root_dir: str, split: str = "training"):
    """
    Load EE (end-effector) trajectories from raw CALVIN dataset by task.

    Uses lang_annotations/auto_lang_ann.npy for (start_id, end_id) and task names.
    Each trajectory is the concatenated robot_obs[:, :3] (x,y,z) across
    episode_XXXXXXX.npz from start_id to end_id.

    Returns:
        dict: task_name -> list of np.ndarray, each (T, 3) in world coordinates.
    """
    root_dir = Path(root_dir)
    ann_path = root_dir / split / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")

    annotations = np.load(ann_path, allow_pickle=True).item()
    indx = annotations["info"]["indx"]  # list of (start_id, end_id)
    tasks = annotations["language"]["task"]  # list of task names

    trajectories_by_task = defaultdict(list)

    for i, ((start_id, end_id), task_name) in enumerate(zip(indx, tasks)):
        trajectory_frames = []
        for ep_id in range(start_id, end_id + 1):
            ep_path = root_dir / split / f"episode_{ep_id:07d}.npz"
            if not ep_path.exists():
                cprint(f"Missing episode: {ep_path}", "yellow")
                continue
            data = np.load(ep_path)
            robot_obs = data["robot_obs"]  # (T, 15) or (15,) for single timestep
            if robot_obs.ndim == 1:
                robot_obs = robot_obs[np.newaxis, :]  # (1, 15)
            ee_pos = robot_obs[:, :3].astype(np.float64)
            trajectory_frames.append(ee_pos)
        if trajectory_frames:
            traj = np.concatenate(trajectory_frames, axis=0)
            trajectories_by_task[task_name].append(traj)
        else:
            cprint(f"Empty trajectory for task '{task_name}' (start={start_id}, end={end_id})", "yellow")

    return dict(trajectories_by_task)


def load_trajectories_from_zarr(zarr_path: str):
    """
    Load EE trajectories from converted DP3 zarr (by episode).

    state is (T, 15); we use state[:, :3] as EE position.
    Returns dict: "episode" -> list of (T, 3) arrays (one per episode).
    """
    if not HAS_ZARR:
        raise ImportError("zarr is required for zarr_path. pip install zarr")

    root = zarr.open(zarr_path, mode="r")
    data = root["data"]
    meta = root["meta"]
    state = data["state"]  # (N, 15) or similar
    episode_ends = meta["episode_ends"][:]

    trajectories = []
    start = 0
    for end in episode_ends:
        ee = np.asarray(state[start:end, :3], dtype=np.float64)
        if len(ee) > 0:
            trajectories.append(ee)
        start = end

    return {"episode": trajectories}


def _plotly_trajectories_to_figure(
    trajectories_by_task,
    title="CALVIN trajectories",
    color_per_task=True,
    max_trajectories_per_task=None,
):
    """Build a single Plotly figure with 3D lines per trajectory."""
    fig = go.Figure()
    # Use a consistent color cycle per task
    default_colors = [
        "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)",
        "rgb(214,39,40)", "rgb(148,103,189)", "rgb(23,190,207)",
        "rgb(255,187,120)", "rgb(174,199,232)", "rgb(255,152,150)",
    ]
    color_idx = 0
    for task_name, trajs in trajectories_by_task.items():
        if max_trajectories_per_task is not None:
            trajs = trajs[: max_trajectories_per_task]
        color = default_colors[color_idx % len(default_colors)]
        color_idx += 1
        for j, traj in enumerate(trajs):
            if traj is None or len(traj) < 2:
                continue
            traj = np.asarray(traj)
            fig.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    name=f"{task_name}" if len(trajs) == 1 else f"{task_name} #{j+1}",
                    line=dict(color=color, width=2),
                    showlegend=True,
                )
            )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def save_trajectory_plots(
    trajectories_by_task,
    output_dir: str,
    one_per_task: bool = True,
    combined: bool = True,
    max_trajectories_per_task=None,
):
    """
    Save interactive HTML plots (Plotly) for trajectory inspection.

    - If one_per_task: save one HTML per task (e.g. open_drawer.html).
    - If combined: save one all_tasks.html with all tasks in one scene.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. pip install plotly")

    os.makedirs(output_dir, exist_ok=True)

    # Sanitize filenames for task names
    def safe_name(name):
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_") or "task"

    if one_per_task:
        for task_name, trajs in trajectories_by_task.items():
            if not trajs:
                continue
            fig = _plotly_trajectories_to_figure(
                {task_name: trajs},
                title=f"Task: {task_name} ({len(trajs)} trajectories)",
                max_trajectories_per_task=max_trajectories_per_task,
            )
            path = os.path.join(output_dir, f"{safe_name(task_name)}.html")
            fig.write_html(path)
            cprint(f"Saved: {path}", "green")

    if combined and len(trajectories_by_task) > 0:
        fig = _plotly_trajectories_to_figure(
            trajectories_by_task,
            title="CALVIN trajectories (all tasks)",
            max_trajectories_per_task=max_trajectories_per_task,
        )
        path = os.path.join(output_dir, "all_tasks.html")
        fig.write_html(path)
        cprint(f"Saved: {path}", "green")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CALVIN trajectories per task as interactive 3D HTML (Plotly)."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to raw CALVIN dataset root (e.g. calvin_debug_dataset) or to converted .zarr",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="calvin_trajectory_plots",
        help="Directory to save HTML files (default: calvin_trajectory_plots)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "validation"],
        help="Split to use when input is raw CALVIN root (default: training)",
    )
    parser.add_argument(
        "--no_per_task",
        action="store_true",
        help="Do not save one HTML per task (only combined if --combined)",
    )
    parser.add_argument(
        "--no_combined",
        action="store_true",
        help="Do not save combined all_tasks.html",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Max trajectories per task to plot (default: all)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        cprint(f"Path does not exist: {input_path}", "red")
        return

    # Detect format: zarr if path ends with .zarr or is a dir with meta/
    is_zarr = False
    if input_path.is_dir():
        is_zarr = str(input_path).endswith(".zarr") or (input_path / "meta").exists()
    if is_zarr and HAS_ZARR:
        try:
            root = zarr.open(str(input_path), mode="r")
            if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
                is_zarr = False
        except Exception:
            is_zarr = False

    if is_zarr:
        cprint("Loading trajectories from converted zarr...", "cyan")
        trajectories_by_task = load_trajectories_from_zarr(str(input_path))
        cprint(f"Loaded {sum(len(t) for t in trajectories_by_task.values())} episodes.", "green")
    else:
        cprint(f"Loading trajectories from raw CALVIN root (split={args.split})...", "cyan")
        trajectories_by_task = load_trajectories_from_calvin_root(str(input_path), split=args.split)
        total = sum(len(t) for t in trajectories_by_task.values())
        cprint(f"Loaded {total} trajectories across {len(trajectories_by_task)} tasks.", "green")

    if not trajectories_by_task:
        cprint("No trajectories found.", "red")
        return

    if not HAS_PLOTLY:
        cprint("Install plotly to save HTML: pip install plotly", "red")
        return

    save_trajectory_plots(
        trajectories_by_task,
        args.output_dir,
        one_per_task=not args.no_per_task,
        combined=not args.no_combined,
        max_trajectories_per_task=args.max_trajectories,
    )
    cprint("Done. Open the HTML files in a browser to rotate/zoom the 3D scene.", "green")


if __name__ == "__main__":
    main()
