import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from diffusers import DDPMScheduler
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import os

# --- CONFIGURATION ---
N_SAMPLES = 9000  # 1000 per mode (9 modes)
BATCH_SIZE = 256
N_EPOCHS = 200    # High epochs to ensure unconditioned matches training dist
TRAJ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GUIDANCE_SCALE = 20.0

# Checkpoint: set via --load / --checkpoint or edit defaults
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_CHECKPOINT_PATH = os.path.join(DEFAULT_CHECKPOINT_DIR, "trajectory_steering.pt")

# --- 1. DATA GENERATION (9 Modes) ---
def generate_dataset(n_samples):
    data, labels = [], []
    samples_per_mode = n_samples // 9
    print(f"Generating {n_samples} trajectories (9 modes)...")
    
    # Obstacles are at x = -2 and x = 2
    # Targets: Left (-4), Center (0), Right (4)
    targets = [-4.0, 0.0, 4.0] 
    
    for mid_idx in [0, 1, 2]:      
        for goal_idx in [0, 1, 2]: 
            for _ in range(samples_per_mode):
                p_start = np.array([0.0, 0.0, 0.0])
                
                # Independent noise for mid and goal
                noise_mid = np.random.normal(0, 0.3, 3)
                noise_end = np.random.normal(0, 0.3, 3)
                
                # Midpoint at Y=5
                mid_z = 2.5 if mid_idx == 1 else 0.0 
                p_mid = np.array([targets[mid_idx], 5.0, mid_z]) + noise_mid
                
                # Endpoint at Y=10
                p_end = np.array([targets[goal_idx], 10.0, 0.0]) + noise_end
                
                # Use 'natural' BC to prevent endpoint whipping
                cs = CubicSpline([0, 5, 10], np.vstack([p_start, p_mid, p_end]), axis=0, bc_type='natural')
                traj = cs(np.linspace(0, 10, TRAJ_LEN))
                
                data.append(traj)
                labels.append(mid_idx * 3 + goal_idx)

    data = torch.tensor(np.array(data), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    
    # Shuffle
    indices = torch.randperm(data.size(0))
    data = data[indices]
    labels = labels[indices]
    
    # Per-Timestep Stats for Normalization
    stats_mean = data.mean(dim=0, keepdim=True)
    stats_std  = data.std(dim=0, keepdim=True).clamp(min=1e-5)
    
    return data, labels, stats_mean.to(DEVICE), stats_std.to(DEVICE)


def normalize(x): return (x - STATS_MEAN.to(x.device)) / STATS_STD.to(x.device)
def unnormalize(x): return x * STATS_STD.to(x.device) + STATS_MEAN.to(x.device)


def save_checkpoint(path, policy, forecaster, stats_mean, stats_std, traj_len, training_subset=None):
    """training_subset: optional numpy (N, TRAJ_LEN, 3) for visualization."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "policy": policy.state_dict(),
        "forecaster": forecaster.state_dict(),
        "stats_mean": stats_mean.cpu(),
        "stats_std": stats_std.cpu(),
        "traj_len": traj_len,
    }
    if training_subset is not None:
        state["training_subset"] = training_subset  # numpy
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, policy, forecaster):
    """Load checkpoint. Returns (STATS_MEAN, STATS_STD, training_subset or None)."""
    ckpt = torch.load(path, map_location=DEVICE)
    policy.load_state_dict(ckpt["policy"])
    forecaster.load_state_dict(ckpt["forecaster"])
    stats_mean = ckpt["stats_mean"].to(DEVICE)
    stats_std = ckpt["stats_std"].to(DEVICE)
    training_subset = ckpt.get("training_subset")  # numpy or None
    if ckpt.get("traj_len") != TRAJ_LEN:
        print(f"Warning: checkpoint TRAJ_LEN={ckpt.get('traj_len')} != current TRAJ_LEN={TRAJ_LEN}")
    print(f"Checkpoint loaded from {path}")
    return stats_mean, stats_std, training_subset


# --- 2. ARCHITECTURE ---
class ResidualBlock1D(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, dim)
        self.time_proj = nn.Linear(dim, dim)
        self.act = nn.Mish()

    def forward(self, x, t_emb):
        h = self.act(self.gn1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.act(self.gn2(self.conv2(h)))
        return x + h

class TemporalResNet(nn.Module):
    def __init__(self, input_dim=3, dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim))
        self.in_conv = nn.Conv1d(input_dim, dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock1D(dim, 1), ResidualBlock1D(dim, 2),
            ResidualBlock1D(dim, 4), ResidualBlock1D(dim, 8),
            ResidualBlock1D(dim, 1), ResidualBlock1D(dim, 1)
        ])
        self.out_conv = nn.Conv1d(dim, input_dim, 3, padding=1)

    def forward(self, x, t):
        x = x.permute(0, 2, 1)
        t_emb = self.time_mlp(t.float().view(-1, 1) / 1000.0)
        h = self.in_conv(x)
        for block in self.blocks: h = block(h, t_emb)
        return self.out_conv(h).permute(0, 2, 1)

policy = TemporalResNet().to(DEVICE)
forecaster = TemporalResNet().to(DEVICE)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
opt_p = optim.Adam(policy.parameters(), lr=1e-4)
opt_f = optim.Adam(forecaster.parameters(), lr=1e-4)

# Global stats (set by train_or_load before run_experiment)
STATS_MEAN = None
STATS_STD = None


TRAINING_VIZ_SIZE = 200  # number of training samples to save for visualization


def train_and_save(checkpoint_path):
    """Generate data, train, save checkpoint. Sets global STATS_MEAN, STATS_STD. Returns training_subset for viz."""
    global STATS_MEAN, STATS_STD
    dataset_raw, labels_raw, STATS_MEAN, STATS_STD = generate_dataset(N_SAMPLES)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(normalize(dataset_raw), labels_raw),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    print("Training...")
    for epoch in tqdm(range(N_EPOCHS)):
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(DEVICE)
            noise = torch.randn_like(x_batch)
            t = torch.randint(0, 1000, (x_batch.shape[0],), device=DEVICE).long()
            x_noisy = noise_scheduler.add_noise(x_batch, noise, t)
            loss_p = F.mse_loss(policy(x_noisy, t), noise)
            opt_p.zero_grad(); loss_p.backward(); opt_p.step()
            loss_f = F.mse_loss(forecaster(x_noisy, t), x_batch)
            opt_f.zero_grad(); loss_f.backward(); opt_f.step()
    training_subset = dataset_raw[:TRAINING_VIZ_SIZE].cpu().numpy()
    save_checkpoint(
        checkpoint_path, policy, forecaster, STATS_MEAN, STATS_STD, TRAJ_LEN,
        training_subset=training_subset,
    )
    return training_subset


def load_models_and_stats(checkpoint_path):
    """Load checkpoint. Sets global STATS_MEAN, STATS_STD. Returns training_subset for viz (or None)."""
    global STATS_MEAN, STATS_STD
    STATS_MEAN, STATS_STD, training_subset = load_checkpoint(checkpoint_path, policy, forecaster)
    return training_subset


# --- 4. FLEXIBLE STEERING ---
def temporal_smooth(traj, kernel_size=5):
    """Smoother helper (Smoothed Tweedie)"""
    if kernel_size <= 1: return traj
    t = traj.permute(0, 2, 1)
    pad = kernel_size // 2
    t = F.pad(t, (pad, pad), mode="replicate")
    t = F.avg_pool1d(t, kernel_size, stride=1)
    return t.permute(0, 2, 1)

def get_loss(traj, mid_targets, goal_targets):
    """
    Computes Min-Distance Loss to a list of allowed targets.
    mid_targets: List of allowed X values for midpoint (e.g., [-4, 0])
    goal_targets: List of allowed X values for endpoint
    """
    traj_real = unnormalize(traj)
    loss = 0.0
    
    def min_dist_loss(segment, targets):
        if targets is None or len(targets) == 0: return 0.0
        # Compute dist to ALL targets
        dists = [(segment - t)**2 for t in targets]
        # Stack: (B, T, N_targets)
        dists_stack = torch.stack(dists, dim=-1)
        # Take MIN across targets (Winner-Takes-All)
        min_dists, _ = torch.min(dists_stack, dim=-1)
        return torch.mean(min_dists)

    # Midpoint Loss (Indices 12-20)
    loss += min_dist_loss(traj_real[:, 12:20, 0], mid_targets)
    # Goal Loss (Indices 28-32)
    loss += min_dist_loss(traj_real[:, 28:, 0], goal_targets)
        
    return loss

def _sample_chunk(method, mid_targets, goal_targets, n_chunk, desc_prefix=""):
    """Sample one chunk (reduces peak GPU memory). Uncond uses no_grad to save memory."""
    policy.eval()
    forecaster.eval()
    if method == "uncond":
        with torch.no_grad():
            x = torch.randn(n_chunk, TRAJ_LEN, 3).to(DEVICE)
            for t in noise_scheduler.timesteps:
                t_batch = torch.full((n_chunk,), t, device=DEVICE, dtype=torch.long)
                noise_pred = policy(x, t_batch)
                x = noise_scheduler.step(noise_pred, t, x).prev_sample
            out = unnormalize(x).cpu().numpy()
    else:
        x = torch.randn(n_chunk, TRAJ_LEN, 3).to(DEVICE)
        for t in noise_scheduler.timesteps:
            t_batch = torch.full((n_chunk,), t, device=DEVICE, dtype=torch.long)
            x = x.detach().requires_grad_(True)
            noise_pred = policy(x, t_batch)
            alpha_t = noise_scheduler.alphas_cumprod[t].to(DEVICE).view(1, 1, 1)
            est = None
            if method == "explicit":
                est = forecaster(x, t_batch)
            elif method == "tweedie":
                est = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            elif method == "smooth":
                # raw_est = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                est = temporal_smooth(x)
            if est is not None:
                cost = get_loss(est, mid_targets, goal_targets)
                grad = torch.autograd.grad(cost, x)[0]
                noise_pred = noise_pred + GUIDANCE_SCALE * grad
            x = noise_scheduler.step(noise_pred, t, x).prev_sample
        out = unnormalize(x).detach().cpu().numpy()
    return out


def run_experiment(mid_targets, goal_targets, n_samples=200, sample_batch_size=24):
    """
    Runs Unconditioned + 3 Steering Methods for the given config.
    Uses chunked sampling to reduce GPU memory (sample_batch_size trajectories at a time).
    """
    results = {}
    if sample_batch_size is None or sample_batch_size >= n_samples:
        sample_batch_size = n_samples
    n_chunks = (n_samples + sample_batch_size - 1) // sample_batch_size

    # 1. Unconditioned Baseline (chunked)
    chunks = []
    for c in tqdm(range(n_chunks), desc="Uncond", leave=False):
        start = c * sample_batch_size
        size = min(sample_batch_size, n_samples - start)
        out = _sample_chunk("uncond", mid_targets, goal_targets, size)
        chunks.append(out)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results["Unconditioned"] = np.concatenate(chunks, axis=0)

    # 2. Steering Methods (chunked)
    for method in ["explicit", "tweedie", "smooth"]:
        chunks = []
        for c in tqdm(range(n_chunks), desc=f"Steer {method}", leave=False):
            start = c * sample_batch_size
            size = min(sample_batch_size, n_samples - start)
            out = _sample_chunk(method, mid_targets, goal_targets, size)
            chunks.append(out)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results[method] = np.concatenate(chunks, axis=0)

    return results

# --- 5. VISUALIZATION ---
def visualize_results(results, title_suffix):
    """
    Visualizes Training Data (if present), Uncond, Explicit, Tweedie, Smooth.
    Columns: Top-Down Trajectories (around cylinders) | Scatter Plot (Mid vs End)
    """
    methods = ["Unconditioned", "explicit", "tweedie", "smooth"]
    colors = ["gray", "green", "blue", "red"]
    if "Training Data" in results:
        methods = ["Training Data"] + methods
        colors = ["black"] + colors

    n_rows = len(methods)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle(f"Steering Configuration: {title_suffix}", fontsize=16)

    mid_idx = 16
    end_idx = -1

    for i, method in enumerate(methods):
        data = results[method]
        color = colors[i]
        label = "Training Data" if method == "Training Data" else method.capitalize()

        # --- Col 1: Top-Down View (trajectories around cylinders) ---
        ax_traj = axes[i, 0]
        for traj in data:
            ax_traj.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.15)
        ax_traj.set_title(f"{label} Trajectories")
        ax_traj.set_xlim(-6, 6)
        ax_traj.set_ylim(0, 11)
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")
        # Obstacles (cylinders at x=-2, x=2, y=5)
        ax_traj.add_patch(plt.Circle((-2, 5), 0.5, color="k", alpha=0.5))
        ax_traj.add_patch(plt.Circle((2, 5), 0.5, color="k", alpha=0.5))
        ax_traj.axhline(5.0, color="k", linestyle=":", alpha=0.2)
        ax_traj.axhline(10.0, color="k", linestyle=":", alpha=0.2)

        # --- Col 2: Scatter Plot (Midpoint X vs Endpoint X) ---
        ax_scat = axes[i, 1]
        mid_x = data[:, mid_idx, 0]
        end_x = data[:, end_idx, 0]
        ax_scat.scatter(mid_x, end_x, color=color, alpha=0.4, s=15)
        ax_scat.set_title(f"{label} Distribution")
        ax_scat.set_xlabel("Midpoint X (t=16)")
        ax_scat.set_ylabel("Endpoint X (t=32)")
        ax_scat.set_xlim(-6, 6)
        ax_scat.set_ylim(-6, 6)
        for tick in [-4, 0, 4]:
            ax_scat.axvline(tick, color="k", linestyle="--", alpha=0.2)
            ax_scat.axhline(tick, color="k", linestyle="--", alpha=0.2)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load trajectory steering models and run experiments.")
    parser.add_argument("--load", action="store_true", help="Load checkpoint and skip training (run experiments only).")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help=f"Path to save/load checkpoint (default: {DEFAULT_CHECKPOINT_PATH})")
    parser.add_argument("--sample-batch", type=int, default=24,
                        help="Samples per chunk (lower = less GPU memory; try 16 or 8 if OOM, default: 24)")
    args = parser.parse_args()

    if args.load:
        training_subset = load_models_and_stats(args.checkpoint)
    else:
        training_subset = train_and_save(args.checkpoint)

    # --- USER CONFIGURATION: change TARGET_CONFIG to try different steering ---
    # Targets are X coordinates: Left=-4.0, Center=0.0, Right=4.0
    #
    # EXAMPLE 1: "Split" - Allow Left OR Right, but NO Center.
    # TARGET_CONFIG = { 'mid': [-4.0, 4.0], 'goal': [-4.0, 4.0] }
    #
    # EXAMPLE 2: "S-Curve" - Must go Left Midpoint -> Center Goal
    # TARGET_CONFIG = { 'mid': [-4.0], 'goal': [0.0] }
    #
    # EXAMPLE 3: "Any Valid" - Allow all 3 lanes (Should match Unconditioned)
    TARGET_CONFIG = { 'mid': [0.0, 4.0], 'goal': [-4.0, 0.0] }

    # TARGET_CONFIG = {
    #     "mid": [-4.0, 4.0],
    #     "goal": [-4.0, 4.0],
    # }

    print(f"Running Experiment with Config: {TARGET_CONFIG}")
    results = run_experiment(
        mid_targets=TARGET_CONFIG["mid"],
        goal_targets=TARGET_CONFIG["goal"],
        sample_batch_size=args.sample_batch,
    )
    if training_subset is not None:
        results["Training Data"] = training_subset
    visualize_results(results, str(TARGET_CONFIG))