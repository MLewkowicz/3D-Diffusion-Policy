"""
CALVIN Environment Wrapper for DP3.

Note: CALVIN is a real robot dataset, so this wrapper is primarily for
compatibility with the DP3 framework. For actual evaluation, you would
need to connect to a real CALVIN robot or use the CALVIN simulation environment.
"""

import gym
import numpy as np
from gym import spaces
from termcolor import cprint

try:
    from calvin_env.envs.play_table_env import get_env
except ImportError:
    cprint("Warning: calvin_env package not found. CALVIN environment wrapper may not work.", "yellow")


class CalvinEnv(gym.Env):
    """
    CALVIN Environment Wrapper for DP3.
    
    This wrapper provides a gym-compatible interface to the CALVIN environment.
    For training, you typically use the dataset directly. This wrapper is
    mainly for evaluation if you have access to the CALVIN simulation.
    """
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, 
                 dataset_path=None,
                 split='validation',
                 num_points=2048,
                 use_gui=False):
        """
        Initialize CALVIN environment.
        
        Args:
            dataset_path: Path to CALVIN dataset (for simulation)
            split: Dataset split ('training' or 'validation')
            num_points: Number of points in point cloud
            use_gui: Whether to show GUI
        """
        self.num_points = num_points
        self.dataset_path = dataset_path
        self.split = split
        
        # Initialize environment if dataset path is provided
        if dataset_path is not None:
            try:
                from pathlib import Path
                val_folder = Path(dataset_path) / f"{split}"
                self.env = get_env(val_folder, show_gui=use_gui)
                self._env_initialized = True
            except Exception as e:
                cprint(f"Warning: Could not initialize CALVIN environment: {e}", "yellow")
                cprint("This is okay if you're only using the dataset for training.", "yellow")
                self.env = None
                self._env_initialized = False
        else:
            self.env = None
            self._env_initialized = False
            cprint("No dataset path provided. Environment wrapper created for compatibility only.", "yellow")

        # Action space: 7-DOF relative actions (x, y, z, euler_x, euler_y, euler_z, gripper)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(15,),  # robot_obs from CALVIN
                dtype=np.float32
            ),
        })

    def reset(self):
        """Reset the environment."""
        if not self._env_initialized:
            raise RuntimeError("Environment not initialized. Provide dataset_path in __init__.")
        
        # Reset CALVIN environment
        obs = self.env.reset()
        
        # Extract observations
        # Note: You'll need to process the CALVIN observations to match DP3 format
        # This is a placeholder - actual implementation depends on CALVIN env structure
        obs_dict = {
            'point_cloud': np.zeros((self.num_points, 3), dtype=np.float32),  # Placeholder
            'agent_pos': np.zeros((15,), dtype=np.float32),  # Placeholder
        }
        
        return obs_dict

    def step(self, action):
        """
        Step the environment.
        
        Args:
            action: 7-DOF relative action array
            
        Returns:
            obs: Observation dictionary
            reward: Reward (placeholder)
            done: Whether episode is done
            info: Additional info
        """
        if not self._env_initialized:
            raise RuntimeError("Environment not initialized. Provide dataset_path in __init__.")
        
        # Step CALVIN environment
        # CALVIN expects relative actions as 7-tuple: (x, y, z, euler_x, euler_y, euler_z, gripper)
        obs, reward, done, info = self.env.step(action)
        
        # Process observations
        obs_dict = {
            'point_cloud': np.zeros((self.num_points, 3), dtype=np.float32),  # Placeholder
            'agent_pos': np.zeros((15,), dtype=np.float32),  # Placeholder
        }
        
        return obs_dict, reward, done, info

    def seed(self, seed=None):
        """Set random seed."""
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self._seed = seed
        np.random.seed(seed)
        if self._env_initialized:
            self.env.seed(seed)

    def close(self):
        """Close the environment."""
        if self._env_initialized and hasattr(self.env, 'close'):
            self.env.close()
