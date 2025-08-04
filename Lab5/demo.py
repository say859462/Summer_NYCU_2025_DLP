import torch
import gymnasium as gym
import numpy as np
import argparse
import os
import imageio  # 匯入 imageio

# 從您的 dqn.py 檔案中匯入您定義的類別
from dqn import DQN, AtariPreprocessor


def run_demo_with_imageio(args):
    """
    載入訓練好的模型，手動收集畫面，並使用 imageio 產生影片。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = None
    if "ALE/" in args.env_name:
        preprocessor = AtariPreprocessor()

    # 1. 建立環境，render_mode 必須是 'rgb_array'
    env = gym.make(args.env_name, render_mode="rgb_array")

    # 初始化模型並載入權重
    num_actions = env.action_space.n
    model = DQN(num_actions, args.env_name).to(device)
    print(f"正在從 {args.model_path} 載入模型...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("模型載入成功！")

    # 2. 建立一個空的 list 用來存放畫面
    frames = []

    # 執行一個 episode
    obs, _ = env.reset()

    # 儲存第一幀畫面
    frames.append(env.render())

    state = preprocessor.reset(obs) if preprocessor else obs

    done = False
    total_reward = 0

    print("開始執行演示並收集畫面...")
    while not done:
        # 使用貪婪策略選擇動作
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        # 與環境互動
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 3. 收集每一幀的畫面
        frames.append(env.render())

        state = preprocessor.step(next_obs) if preprocessor else next_obs
        total_reward += reward

    env.close()

    # 4. 使用 imageio 將收集到的所有畫面儲存成影片
    video_dir = os.path.join(args.save_dir, "videos_imageio")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{args.run_name}.mp4")

    print(f"正在將 {len(frames)} 幀畫面儲存至 {video_path}...")
    imageio.mimsave(video_path, frames, fps=30)  # fps 參數可以控制影片的播放速度

    print("-" * 30)
    print(f"演示結束！總獎勵: {total_reward}")
    print(f"影片已成功儲存！")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Demo Script with ImageIO")
    parser.add_argument(
        "--env-name", type=str, default="CartPole-v1", help="Gym 環境名稱"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="已訓練好的模型權重檔案路徑 (.pt)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results/vedio", help="影片儲存的根目錄"
    )
    parser.add_argument(
        "--run-name", type=str, default="demo_imageio", help="影片檔案的名稱"
    )

    args = parser.parse_args()

    run_demo_with_imageio(args)
