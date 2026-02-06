import torch
import numpy as np

def farthest_point_sampling(points: np.ndarray, num_points: int = 1024, use_cuda: bool = False) -> tuple:
    """
    Hybrid FPS: Automatically moves Numpy data to GPU for processing, then back.
    """
    # 1. GPU ACCELERATION PATH
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        # Move Numpy -> Tensor (GPU)
        pts_tensor = torch.from_numpy(points[:, :3]).float().to(device)
        N, D = pts_tensor.shape
        
        # Handle cases where we have fewer points than requested
        if N < num_points:
            indices = torch.arange(N, device=device)
            padding = torch.full((num_points - N,), N - 1, device=device, dtype=torch.long)
            indices = torch.cat([indices, padding])
            sampled_pts = pts_tensor[indices]
            return sampled_pts.cpu().numpy(), indices.cpu().numpy()

        # --- THE OPTIMIZED LOGIC (Same as your pasted code) ---
        with torch.no_grad():
            centroids = torch.zeros((num_points,), dtype=torch.long, device=device)
            distance = torch.ones((N,), dtype=torch.float32, device=device) * 1e10
            farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()
            
            # Batch indices not needed here since we process 1 frame at a time in conversion
            for i in range(num_points):
                centroids[i] = farthest
                centroid = pts_tensor[farthest, :].view(1, 3)
                dist = torch.sum((pts_tensor - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = torch.max(distance, -1)[1].item()
            
        sampled_pts = pts_tensor[centroids]
        # Move Tensor (GPU) -> Numpy (CPU)
        return sampled_pts.cpu().numpy(), centroids.cpu().numpy()

    # 2. CPU FALLBACK (Standard Numpy)
    # This runs if --no_cuda is passed or GPU is missing
    N, D = points.shape
    if N < num_points:
        padding = np.tile(points[-1:], (num_points - N, 1))
        sampled_points = np.vstack([points, padding])
        indices = np.concatenate([np.arange(N), np.full(num_points - N, N-1)])
        return sampled_points, indices

    xyz = points[:, :3]
    centroids = np.zeros((num_points,), dtype=np.int64)
    distance = np.ones((N,), dtype=np.float64) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_points):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=0)
    sampled_points = points[centroids]
    return sampled_points, centroids