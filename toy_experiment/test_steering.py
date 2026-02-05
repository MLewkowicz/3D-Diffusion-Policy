import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from diffusers import DDPMScheduler
from tqdm import tqdm
import torch.nn.functional as F

# --- CONFIGURATION ---
N_SAMPLES = 6000 # More data for better density
BATCH_SIZE = 256
N_EPOCHS = 150
TRAJ_LEN = 32  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GUIDANCE_SCALE = 20.0

# --- 1. ROBUST DATA GENERATION ---
def generate_dataset(n_samples):
    data, labels = [], []
    samples_per_mode = n_samples // 3
    print(f"Generating {n_samples} trajectories...")
    
    for mode in [0, 1, 2]:
        for _ in range(samples_per_mode):
            p_start = np.array([0.0, 0.0, 0.0])
            p_end   = np.array([0.0, 10.0, 0.0])
            
            noise = np.random.normal(0, 0.4, 3) 
            
            if mode == 0:   p_mid = np.array([-4.0, 5.0, 0.0]) + noise
            elif mode == 1: p_mid = np.array([0.0, 5.0, 3.0]) + noise
            else:           p_mid = np.array([4.0, 5.0, 0.0]) + noise
            
            # Use 'clamped' boundary conditions to prevent massive overshooting to -7.5
            cs = CubicSpline([0, 5, 10], np.vstack([p_start, p_mid, p_end]), axis=0, bc_type='clamped')
            traj = cs(np.linspace(0, 10, TRAJ_LEN)) 
            
            data.append(traj)
            labels.append(mode)
            
    data = torch.tensor(np.array(data), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    
    # SHUFFLE
    indices = torch.randperm(data.size(0))
    data = data[indices]
    labels = labels[indices]
    
    # --- CRITICAL FIX: PER-TIMESTEP NORMALIZATION ---
    # Normalize each timestep independently (T, 3) instead of globally (3)
    # This prevents the zero-variance start/end points from skewing the stats
    stats_mean = data.mean(dim=0, keepdim=True) # (1, T, 3)
    stats_std  = data.std(dim=0, keepdim=True)  # (1, T, 3)
    
    # Avoid div by zero
    stats_std = torch.clip(stats_std, min=1e-5)
    
    print(f"Data Range X: [{data[:,:,0].min():.2f}, {data[:,:,0].max():.2f}]")
    
    return data, labels, stats_mean.to(DEVICE), stats_std.to(DEVICE)

# Prepare Data
dataset_raw, labels_raw, STATS_MEAN, STATS_STD = generate_dataset(N_SAMPLES)

def normalize(x): return (x - STATS_MEAN.to(x.device)) / STATS_STD.to(x.device)
def unnormalize(x): return x * STATS_STD.to(x.device) + STATS_MEAN.to(x.device) 

dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(normalize(dataset_raw), labels_raw), 
    batch_size=BATCH_SIZE, shuffle=True
)

# --- 2. ARCHITECTURE: ResNet with Strong Time Conditioning ---
class ResidualBlock1D(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, dim)
        
        # FiLM-like Time Embedding Projection for this block
        self.time_proj = nn.Linear(dim, dim) 
        self.act = nn.Mish()

    def forward(self, x, t_emb):
        # Inject Time Embedding into the block
        h = self.act(self.gn1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1) # Add time here!
        h = self.act(self.gn2(self.conv2(h)))
        return x + h

class TemporalResNet(nn.Module):
    def __init__(self, input_dim=3, dim=256): # Increased dim to 256
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
        for block in self.blocks:
            h = block(h, t_emb) # Pass time to every block
            
        return self.out_conv(h).permute(0, 2, 1)

policy = TemporalResNet().to(DEVICE)
forecaster = TemporalResNet().to(DEVICE)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# --- 3. TRAINING ---
opt_p = optim.Adam(policy.parameters(), lr=1e-4)
opt_f = optim.Adam(forecaster.parameters(), lr=1e-4)

print("Training...")
for epoch in tqdm(range(N_EPOCHS)):
    for x_batch, _ in dataloader:
        x_batch = x_batch.to(DEVICE)
        noise = torch.randn_like(x_batch)
        t = torch.randint(0, 1000, (x_batch.shape[0],), device=DEVICE).long()
        x_noisy = noise_scheduler.add_noise(x_batch, noise, t)
        
        # Policy
        loss_p = F.mse_loss(policy(x_noisy, t), noise)
        opt_p.zero_grad(); loss_p.backward(); opt_p.step()
        
        # Forecaster
        loss_f = F.mse_loss(forecaster(x_noisy, t), x_batch)
        opt_f.zero_grad(); loss_f.backward(); opt_f.step()

# --- 4. STEERING ---
def temporal_smooth(traj, kernel_size=5):
    """Smooth trajectory along time with a 1D average filter (preserves length)."""
    if kernel_size <= 1:
        return traj
    t = traj.permute(0, 2, 1)  # (N, 3, T)
    pad = kernel_size // 2
    t = F.pad(t, (pad, pad), mode="replicate")
    t = F.avg_pool1d(t, kernel_size, stride=1)  # stride=1 keeps same length
    return t.permute(0, 2, 1)


# target_config: 0=left (-4), 1=center (0), 2=right (4), 3=center+right (2, avoid left)
def get_loss(traj, target_config):
    traj_real = unnormalize(traj)
    mid_x = traj_real[:, 12:20, 0]
    if target_config == 0:   target = -4.0   # Steer Left
    elif target_config == 1: target = 0.0    # Steer Center
    elif target_config == 2: target = 4.0    # Steer Right
    else:                    target = 2.0    # Steer Center+Right (between 0 and 4, avoid left)
    return torch.mean((mid_x - target) ** 2)


def generate_comparison():
    print("Generating Samples...")
    N = 60
    training_subset = unnormalize(normalize(dataset_raw[:N])).cpu().numpy()
    uncond = None  # computed once and reused

    def sample(method, target_config=None):
        nonlocal uncond
        x = torch.randn(N, TRAJ_LEN, 3).to(DEVICE)
        policy.eval()
        forecaster.eval()
        desc = f"{method} (cfg {target_config})" if target_config is not None else method

        for t in tqdm(noise_scheduler.timesteps, leave=False, desc=desc):
            t_batch = torch.full((N,), t, device=DEVICE, dtype=torch.long)
            x = x.detach().requires_grad_(True)
            noise_pred = policy(x, t_batch)

            if method != "uncond":
                est = None
                alpha_t = noise_scheduler.alphas_cumprod[t].to(DEVICE).view(1, 1, 1)
                if method == "explicit":
                    est = forecaster(x, t_batch)
                elif method == "tweedie":
                    est = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                elif method == "smooth":
                    # est = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    est = temporal_smooth(x)
                if est is not None:
                    cost = get_loss(est, target_config)
                    grad = torch.autograd.grad(cost, x)[0]
                    noise_pred = noise_pred + GUIDANCE_SCALE * grad

            x = noise_scheduler.step(noise_pred, t, x).prev_sample

        return unnormalize(x).detach().cpu().numpy()

    # Unconditioned once for all figures
    uncond = sample("uncond")

    # Three configurations: Steer Left, Steer Center+Right, Steer Right
    results_steer_left = {
        "Training Data": training_subset,
        "Unconditioned": uncond,
        "Explicit": sample("explicit", target_config=0),
        "Tweedie": sample("tweedie", target_config=0),
        "Smooth": sample("smooth", target_config=0),
    }
    results_steer_center_right = {
        "Training Data": training_subset,
        "Unconditioned": uncond,
        "Explicit": sample("explicit", target_config=3),
        "Tweedie": sample("tweedie", target_config=3),
        "Smooth": sample("smooth", target_config=3),
    }
    results_steer_right = {
        "Training Data": training_subset,
        "Unconditioned": uncond,
        "Explicit": sample("explicit", target_config=2),
        "Tweedie": sample("tweedie", target_config=2),
        "Smooth": sample("smooth", target_config=2),
    }

    return {
        "steer_left": results_steer_left,
        "steer_center_right": results_steer_center_right,
        "steer_right": results_steer_right,
    }

# --- 5. VISUALIZATION ---
def draw_one_figure(results_dict, title_suffix, n_rows=5):
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    plt.subplots_adjust(hspace=0.45)
    fig.suptitle(title_suffix, fontsize=14)
    mid_idx = min(16, TRAJ_LEN // 2)
    configs = [
        ("Training Data", "Training Data (Subset)", "black"),
        ("Unconditioned", "Unconditioned", "gray"),
        ("Explicit", "Explicit", "green"),
        ("Tweedie", "Tweedie", "blue"),
        ("Smooth", "Smooth", "red"),
    ]
    for i, (key, title, color) in enumerate(configs):
        data = results_dict[key]
        ax_traj = axes[i, 0]
        for traj in data:
            ax_traj.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.3)
        ax_traj.set_title(title)
        ax_traj.set_xlim(-6, 6)
        ax_traj.set_ylim(0, 11)
        ax_traj.add_patch(plt.Circle((-2, 5), 0.5, color="k", alpha=0.5))
        ax_traj.add_patch(plt.Circle((2, 5), 0.5, color="k", alpha=0.5))
        ax_traj.axvline(-4, color="b", linestyle="--", alpha=0.35)
        ax_traj.axvline(0, color="g", linestyle="--", alpha=0.35)
        ax_traj.axvline(4, color="r", linestyle="--", alpha=0.35)
        ax_hist = axes[i, 1]
        mid_x = data[:, mid_idx, 0]
        ax_hist.hist(mid_x, bins=np.linspace(-6, 6, 40), color=color, alpha=0.7)
        ax_hist.set_title("Midpoint X Distribution")
        ax_hist.axvline(-4, color="b", linestyle="--", alpha=0.4)
        ax_hist.axvline(0, color="g", linestyle="--", alpha=0.4)
        ax_hist.axvline(4, color="r", linestyle="--", alpha=0.4)
        ax_hist.axvline(-2, color="k", linestyle=":", alpha=0.3)
        ax_hist.axvline(2, color="k", linestyle=":", alpha=0.3)
        ax_3d = fig.add_subplot(n_rows, 3, i * 3 + 3, projection="3d")
        axes[i, 2].axis("off")
        for traj in data:
            ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.3)
        ax_3d.set_title("3D View")
        ax_3d.set_zlim(0, 4)
    return fig


def visualize():
    results = generate_comparison()
    draw_one_figure(results["steer_left"], "Steer LEFT — all methods steer left")
    draw_one_figure(results["steer_center_right"], "Steer CENTER + RIGHT — all methods avoid left, split center/right")
    draw_one_figure(results["steer_right"], "Steer RIGHT — all methods steer right")
    plt.show()

if __name__ == "__main__":
    visualize()