# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import sumtree
from rainbowDQN import DuelingDQN, RainbowDQN, compute_distributional_loss

gym.register_envs(ale_py)


def init_weights(m):
    # initialize the weights of the neural network if the layer is Conv2d or Linear
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Q-Function Approximator
class DQN(nn.Module):
    """
    Design the architecture of your deep Q network
    - Input size is the same as the state dimension; the output size is the same as the number of actions
    - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
    - Feel free to add any member variables/functions whenever needed
    """

    def __init__(self, num_actions, env_name="CartPole-v1"):
        super(DQN, self).__init__()
        # An example:
        # self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        # )
        ########## YOUR CODE HERE (5~10 lines) ##########
        # state : (vcart, a_cart,𝜃_pole→cart, a_pole)
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
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )

        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if x.dim() == 4:
            x = x / 255.0
        return self.network(x)


# Preprocessing the image of Atari games into the accpectable input of DQN
class AtariPreprocessor:
    """
    Preprocesing the state input of DQN for Atari
    """

    def __init__(self, frame_stack=4):
        # Since the input image of Atrai games is high-resolution RGB image, we need to preprocess it to reduce the complexity
        # frame_stack: the number of frames to stack together
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Receive the observation from the environment and preprocess it
        # Convert the RGB image to grayscale, resize it to 84x84, and normalize
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        # The first frame of the episode is used to initialize the frame stack
        frame = self.preprocess(obs)
        self.frames = deque(
            [frame for _ in range(self.frame_stack)], maxlen=self.frame_stack
        )
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        # . Receive the next observation and add it to the frame stack
        frame = self.preprocess(obs)  # (84, 84)
        self.frames.append(frame)

        # return (self.frame_stack, 84, 84) shape of the stacked frames
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
    Prioritizing the samples in the replay memory by the Bellman error
    See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.tree = sumtree.SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = 0.01

    def add(self, transition, error=None):
        max_p = np.max(self.tree.tree[-self.tree.capacity :])

        # If there's no element in the tree, set max_p to 1.0
        if max_p == 0:
            max_p = 1.0

        self.tree.add(max_p, transition)

    def __len__(self):
        return self.tree.n_entries

    def sample(self, batch_size, beta=0.4):

        # Store the sample drawed from the replay memory
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_memory = []
        is_weights = np.empty((batch_size, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):

            # Sampling interval [a,b]
            a = priority_segment * i
            b = priority_segment * (i + 1)

            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)

            # P(i)
            sampling_prob = p / self.tree.total_priority()

            # calculate the importance sampling weight
            # w = ( 1/  (N*P(I)) ) ^ beta
            is_weights[i, 0] = np.power(self.tree.n_entries * sampling_prob, -beta)
            b_idx[i] = idx
            b_memory.append(data)

        # Normalize the importance sampling weights
        is_weights /= is_weights.max()
        return b_idx, b_memory, is_weights

    def update_priorities(self, tree_indices, abs_errors):
        
        errors = np.nan_to_num(abs_errors) + self.eps
        priorities = np.power(errors, self.alpha)

        for ti, p in zip(tree_indices, priorities):
            self.tree.update(ti, p)


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):

        # Initialize the DQN agent with the environment(like CartPole and artari games) and hyperparameters
        self.env = gym.make(
            env_name, render_mode="rgb_array"
        )
        self.test_env = gym.make(
            env_name, render_mode="rgb_array"
        )
        # There will be some differences on the environment settings for Atari games and CartPole
        self.num_actions = self.env.action_space.n
        self.env_name = env_name

        self.preprocessor = None
        if self.env_name == "ALE/Pong-v5":
            self.preprocessor = AtariPreprocessor()

        self.eval_scores = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # # DQN network
        # self.q_net = DQN(self.num_actions, env_name).to(self.device)
        # self.q_net.apply(init_weights)
        # # Target network for DQN
        # self.target_net = DQN(self.num_actions, env_name).to(self.device)
        # self.target_net.load_state_dict(self.q_net.state_dict())
        # self.target_net.eval()  # Set the target network to evaluation mode

        # Dueling Network Ver.
        # self.q_net = DuelingDQN(self.num_actions, env_name).to(self.device)
        # self.q_net.apply(init_weights)
        # # Target network for DQN
        # self.target_net = DuelingDQN(self.num_actions, env_name).to(self.device)
        # self.target_net.load_state_dict(self.q_net.state_dict())
        # self.target_net.eval()  # Set the target network to evaluation mode

        # Parameter for Double DQN,PER,multi-step return and Rainbow DQN
        self.use_ddqn = args.use_ddqn
        self.use_per = args.use_per
        self.n_steps = args.n_steps
        self.use_rainbow = args.use_rainbow

        # Distributional and Noisy are controlled by Rainbow DQN
        self.use_dist = self.use_rainbow
        self.use_noisy = self.use_rainbow

        if self.use_dist:
            self.n_atoms = args.n_atoms
            self.v_min = args.v_min
            self.v_max = args.v_max
            self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(
                self.device
            )
            self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        if self.use_rainbow:
            print("Using Rainbow Model (Dueling + Distributional + Noisy)")
            self.q_net = RainbowDQN(
                self.num_actions, self.n_atoms, self.v_min, self.v_max, env_name
            ).to(self.device)
            self.target_net = RainbowDQN(
                self.num_actions, self.n_atoms, self.v_min, self.v_max, env_name
            ).to(self.device)
        else:
            # DQN
            print("Using Standard DQN Model")
            self.q_net = DQN(self.num_actions, env_name).to(self.device)
            self.target_net = DQN(self.num_actions, env_name).to(self.device)

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        if self.use_per:
            # Initialize the prioritized replay buffer
            self.memory = PrioritizedReplayBuffer(
                capacity=args.memory_size, alpha=args.per_alpha, beta=args.per_beta
            )
        else:
            self.memory = deque(maxlen=args.memory_size)

        # For multi-step return, we need to keep a buffer of the last n steps
        if self.n_steps > 1:
            self.n_step_buffer = deque(maxlen=self.n_steps)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr,weight_decay=5e-4)
        self.scheduler = None
        if args.use_lr_scheduler:
            print(
                f"Using ExponentialLR scheduler with gamma={args.lr_decay_gamma} and step={args.lr_scheduler_step}"
            )
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=args.lr_decay_gamma
            )
            self.lr_scheduler_step = args.lr_scheduler_step
        self.per_beta_start = args.per_beta
        self.total_steps = args.total_steps
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0

        if env_name == "CartPole-v1":
            self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        elif env_name == "ALE/Pong-v5":
            self.best_reward = -21

        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir + f"\{env_name}_{args.wandb_run_name}"

        print(
            f"Running with settings: DDQN={self.use_ddqn}, PER={self.use_per}, N-steps={self.n_steps}"
        )
        if self.use_per:
            print(f"PER params: alpha={args.per_alpha}, beta={args.per_beta}")
        os.makedirs(self.save_dir, exist_ok=True)

    # Epsilon-greedy action selection
    def select_action(self, state):
        # If the Rainbow DQN is used, we will use the get_q_values method to get the Q-values
        if self.use_noisy:
            state_tensor = (
                torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                q_values = self.q_net.get_q_values(state_tensor)

            return q_values.argmax().item()

        # Epsilon-greedy action selection
        # random
        if random.random() < self.epsilon:
            # Select a random action from the action space [0, num_actions)
            return random.randint(0, self.num_actions - 1)

        # greedy
        state_tensor = (
            torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        )
        # Computer the Q-values for the current state for each action using the Q-network
        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        # Select the action with the highest Q-value
        return q_values.argmax().item()

    def _get_n_step_info(self):
        # Get the n-step return information from the n-step buffer
        reward = 0.0
        for i in range(len(self.n_step_buffer)):
            reward += (self.gamma**i) * self.n_step_buffer[i][2]

        # (s_t, a_t, R_t^{(n)}, s_{t+n}, done_{t+n})
        return (
            self.n_step_buffer[0][0],
            self.n_step_buffer[0][1],
            reward,
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4],
        )

    def run(self, episodes=1000):
        # episode, just like playing the whole game
        for ep in range(episodes):
            obs, _ = self.env.reset()
            # Since there might be differnet games like CartPole and Atari games
            # We dont need to preprocess the observation for CartPole
            state = self.preprocessor.reset(obs) if self.preprocessor else obs

            if self.n_steps > 1:
                self.n_step_buffer.clear()

            # Does the episode end flag
            done = False
            total_reward = 0
            step_count = 0
            # step, just like taking one step in the game
            while not done and step_count < self.max_episode_steps:

                if self.use_noisy:
                    self.q_net.reset_noise()
                    self.target_net.reset_noise()
                # select action using epsilon-greedy policy
                action = self.select_action(state)

                # Take the action in the environment and get the next state, reward, and done flag
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                """
                    CartPole-v1: reward is 1 if alive, and -1 if died, 
                    Pong-v5: reward is -1 for losing,and 1 for winning
                    
                    Ｐong,Terminated: The game is over due to the internal rule (e.g. the opponent wins in Pong)
                    Truncated: The game is over due to the time limit
                    
                    CartPole, Terminated: The game is over due to the pole falling down or the cart moving out of bounds
                    Truncated: The game is over due to exceeding the maximum number of steps
                """

                done = terminated or truncated

                # Take the next observation and preprocess it into the acceptable input of DQN

                next_state = (
                    self.preprocessor.step(next_obs) if self.preprocessor else next_obs
                )

                # Store the transition in the replay memory
                transition = (state, action, reward, next_state, done)

                if self.n_steps > 1:
                    self.n_step_buffer.append(transition)
                    if len(self.n_step_buffer) == self.n_steps:
                        # when buffer is full，calculate n-step return and store into memory
                        n_step_transition = self._get_n_step_info()
                        if self.use_per:
                            self.memory.add(n_step_transition, None)
                        else:
                            self.memory.append(n_step_transition)
                else:  # 1-step return
                    if self.use_per:

                        self.memory.add(transition, None)
                    else:
                        self.memory.append(transition)

                # Train the DQN network
                for _ in range(self.train_per_step):
                    self.train()

                if self.n_steps > 1 and done:
                    while len(self.n_step_buffer) > 0:
                        n_step_transition = self._get_n_step_info()
                        s_t, a_t, R_t_n, s_t_n, _ = n_step_transition
                        corrected_transition = (s_t, a_t, R_t_n, s_t_n, True)
                        if self.use_per:
                            self.memory.add(corrected_transition)
                        else:
                            self.memory.append(corrected_transition)

                        self.n_step_buffer.popleft()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 100000 == 0:
                    if self.env_name == "ALE/Pong-v5":
                        model_path = os.path.join(
                            self.save_dir, f"model_pong{self.env_count}.pt"
                        )
                    else:
                        model_path = os.path.join(
                            self.save_dir, f"model_CartPole{self.env_count}.pt"
                        )
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved model checkpoint to {model_path}")

                if self.env_count % 1000 == 0:
                    print(
                        f"[Collect] Ep: {ep} Step: {step_count} Env_count: {self.env_count} Train_count: {self.train_count} Eps: {self.epsilon:.4f}"
                    )
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "Episode": ep,
                            "Step Count": step_count,
                            "Env Step Count": self.env_count,
                            "Update Count": self.train_count,
                            "Epsilon": self.epsilon,
                            "Learning Rate": current_lr,
                        },
                        step=self.env_count,
                    )
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed

                    ########## END OF YOUR CODE ##########
            print(
                f"[Eval] Ep: {ep} Total Reward: {total_reward} Env_Count: {self.env_count} Train_Count: {self.train_count} Eps: {self.epsilon:.4f}"
            )
            wandb.log(
                {
                    "Episode": ep,
                    "Total Reward": total_reward,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                },
                step=self.env_count,
            )

            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed

            ########## END OF YOUR CODE ##########

            if ep % 5 == 0:
                eval_reward = self.evaluate()
                self.eval_scores.append(eval_reward)
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(
                        self.save_dir,
                        f"best_model_{self.env_count}_{self.best_reward}.pt",
                    )
                    torch.save(self.q_net.state_dict(), model_path)
                    print(
                        f"Saved new best model to {model_path} with reward {eval_reward}"
                    )
                print(
                    f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} Env_Count: {self.env_count} Train_Count: {self.train_count}"
                )
                wandb.log(
                    {
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Eval Reward": eval_reward,
                    },
                    step=self.env_count,
                )

        print("\n" + "=" * 40)
        print("           Training Finished           ")
        print("=" * 40)
        if self.eval_scores:
            avg_score = np.mean(self.eval_scores)
            max_score = np.max(self.eval_scores)
            print(f"Total Evaluations Performed: {len(self.eval_scores)}")
            print(f"Average Evaluation Score: {avg_score:.2f}")
            print(f"Maximum Evaluation Score: {max_score:.2f}")
            wandb.log(
                {"final_avg_eval_score": avg_score, "final_max_eval_score": max_score}
            )
        else:
            print("No evaluations were performed (total episodes < 20).")
        print("=" * 40)

    def evaluate(self):
        self.q_net.eval()
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs) if self.env_name == "ALE/Pong-v5" else obs

        done = False
        total_reward = 0

        while not done:
            state_tensor = (
                torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                if self.use_rainbow:
                    q_values = self.q_net.get_q_values(state_tensor)
                else:
                    q_values = self.q_net(state_tensor)
                action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward

            state = (
                self.preprocessor.step(next_obs)
                if self.env_name == "ALE/Pong-v5"
                else next_obs
            )
            
        self.q_net.train()
        return total_reward

    def train(self):
        
        if len(self.memory) < self.replay_start_size:
            return

        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_count += 1

        if self.use_per:
            beta = self.per_beta_start + (1.0 - self.per_beta_start) * (
                self.train_count / self.total_steps
            )
            beta = min(1.0, beta)  # Clamp beta to [0, 1]

            tree_indices, mini_sample, is_weights = self.memory.sample(
                self.batch_size, beta
            )
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            is_weights = torch.from_numpy(is_weights).to(self.device).float()
        else:
            mini_sample = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            is_weights = None

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(
            self.device
        )
        # size (batch_size,)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        if self.use_dist:
            # For rainbow DQN 
            gamma = self.gamma**self.n_steps
            loss, abs_errors = compute_distributional_loss(
                q_net=self.q_net,
                target_net=self.target_net,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                gamma=gamma,
                support=self.support,
                delta_z=self.delta_z,
                v_min=self.v_min,
                v_max=self.v_max,
                n_atoms=self.n_atoms,
                batch_size=self.batch_size,
                is_weights=is_weights,
            )

            if self.use_per:
                self.memory.update_priorities(tree_indices, abs_errors)
        else:
            ########## YOUR CODE HERE (~10 lines) ##########
            # Implement the loss function of DQN and the gradient updates
            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if self.use_ddqn:
                    # Select the actions for the next states using the main network
                    main_net_actions = self.q_net(next_states).argmax(1, keepdim=True)
                    # Use the target network to get the Q-values for the next states
                    next_q_values = (
                        self.target_net(next_states)
                        .gather(1, main_net_actions)
                        .squeeze(1)
                    )
                else:
                    # Compute the target Q-values using the target network
                    next_q_values = self.target_net(next_states).max(1)[0]
            gamma = self.gamma**self.n_steps
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
            # if the games is done, the target Q-value is just the reward (there is no future reward)
            # if the game is not done, the target Q-value is the reward + discounted future
            # if n_steps > 1, we need to consider the n-step return (gamma^n)
            # Calculate dqn loss
            if self.use_per:
                td_errors = target_q_values - q_values
                loss = (is_weights * (td_errors**2)).mean()
                abs_errors = td_errors.abs().detach().cpu().numpy()
                self.memory.update_priorities(tree_indices, abs_errors)
            else:
                loss = nn.functional.mse_loss(q_values, target_q_values)

        # Optimize Step
        self.optimizer.zero_grad()
        loss.backward()
        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        # Update the learning rate if a scheduler is used
        if (
            self.scheduler is not None
            and self.train_count % self.lr_scheduler_step == 0
        ):
            self.scheduler.step()

        # Update the model weight of target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            if self.use_dist:
                # Ouput Loss only for Distributional DQN
                print(
                    f"[Train #{self.train_count}] Loss: {loss.item():.4f} (Distributional Mode)"
                )
            else:
                # Output Loss and Q-value statistics for standard DQN
                print(
                    f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    # Start training after collecting enough samples
    parser.add_argument("--replay_start_size", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument(
        "--env-name",
        type=str,
        default="CartPole-v1",
        choices=["CartPole-v1", "ALE/Pong-v5"],
    )
    parser.add_argument("--episode", type=int, default=1000)
    # Enhancements Flags
    parser.add_argument("--use-ddqn", action="store_true", help="Use Double DQN (DDQN)")
    parser.add_argument(
        "--use-per", action="store_true", help="Use Prioritized Experience Replay (PER)"
    )

    # PER Hyperparameters
    parser.add_argument("--per-alpha", type=float, default=0.6, help="Alpha for PER")
    parser.add_argument("--per-beta", type=float, default=0.4, help="Beta for PER")

    # Rainbow DQN
    parser.add_argument(
        "--use-rainbow",
        action="store_true",
        help="Use Rainbow components (Dueling, Distributional, Noisy)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=1, help="N-step return for Rainbow DQN"
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=500000,
        help="Total training steps for beta annealing schedule",
    )

    # Distributinoal RL Parameters
    parser.add_argument(
        "--n-atoms", type=int, default=51, help="Number of atoms for Distributional RL"
    )
    parser.add_argument(
        "--v-min",
        type=float,
        default=-10.0,
        help="Minimum value of support for Distributional RL",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=10.0,
        help="Maximum value of support for Distributional RL",
    )

    # Learning Rate Scheduler
    parser.add_argument(
        "--use-lr-scheduler",
        action="store_true",
        help="Enable learning rate scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.999,
        help="Gamma for ExponentialLR scheduler",
    )
    parser.add_argument(
        "--lr-scheduler-step",
        type=int,
        default=5000,
        help="Step frequency for LR scheduler",
    )

    args = parser.parse_args()

    run_name = f"{args.env_name}"

    if args.use_rainbow:
        run_name += "-Rainbow"
        args.use_ddqn = True
        args.use_per = True
    else:
        run_name += "-VanillaDQN"
        if args.use_ddqn:
            run_name += "+DDQN"
        if args.use_per:
            run_name += "+PER"
        if args.n_steps > 1:
            run_name += f"+{args.n_steps}steps"

    run_name += f"-lr{args.lr}-bs{args.batch_size}-eps_decay{args.epsilon_decay}-ms{args.memory_size}-replay_start{args.replay_start_size}_target_update{args.target_update_frequency}_nstep_{args.n_steps}"

    if args.env_name == "ALE/Pong-v5":
        wandb.init(project=f"DLP-Lab5-DQN-PongV5", name=run_name, save_code=True)
    else:
        wandb.init(
            project=f"DLP-Lab5-DQN-{args.env_name}", name=run_name, save_code=True
        )
    args.wandb_run_name = run_name
    agent = DQNAgent(args=args, env_name=args.env_name)
    agent.run(episodes=args.episode)
