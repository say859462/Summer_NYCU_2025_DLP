import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
from rainbowDQN import RainbowDQN


class DQN(nn.Module):
    def __init__(self, num_actions, env_name="CartPole-v1"):
        super(DQN, self).__init__()
        if env_name == "CartPole-v1":
            self.network = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )
        elif env_name == "ALE/Pong-v5":
            # The input is a stack of 4 frames, each frame is a grayscale image of size 84x84
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )

    def forward(self, x):
        if x.dim() == 4:
            x = x / 255.0  # Normalize the input for Atari games
        return self.network(x)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque(
            [frame for _ in range(self.frame_stack)], maxlen=self.frame_stack
        )
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor() if args.env_name.startswith("ALE/") else None
    num_actions = env.action_space.n

    if args.model_type == "dqn":
        model = DQN(num_actions=num_actions, env_name=args.env_name).to(device)
    elif args.model_type == "rainbow":
        model = RainbowDQN(
            num_actions=num_actions,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            env_name=args.env_name,
        ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs) if args.env_name.startswith("ALE/") else obs
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                if args.model_type == "rainbow":
                    # For Rainbow DQN, use the get_q_values method to get action
                    action = model.get_q_values(state_tensor).argmax().item()
                else:
                    # For standard DQN
                    action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = (
                preprocessor.step(next_obs)
                if args.env_name.startswith("ALE/")
                else next_obs
            )
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained .pt model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["dqn", "rainbow"],
        default="dqn",
        help="Type of model to evaluate",
    )
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=48763, help="Random seed for evaluation"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    # Parameters for Rainbow DQN
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=51,
        help="Number of atoms for Distributional RL (for rainbow)",
    )
    parser.add_argument(
        "--v-min",
        type=float,
        default=-10.0,
        help="Minimum value of support (for rainbow)",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=10.0,
        help="Maximum value of support (for rainbow)",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="ALE/Pong-v5",
        help="Name of the Gymnasium environment to use (e.g., 'ALE/Pong-v5', 'CartPole-v1')",
    )
    args = parser.parse_args()
    evaluate(args)
