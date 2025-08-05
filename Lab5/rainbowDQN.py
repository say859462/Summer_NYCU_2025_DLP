from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


# Reference : https://github.com/Curt-Park/rainbow-is-all-you-need


class DuelingDQN(nn.Module):
    """
    Dueling Network Architecture for DQN
    """

    def __init__(self, num_actions, env_name="CartPole-v1"):
        super(DuelingDQN, self).__init__()
        self.num_actions = num_actions
        self.env_name = env_name

        if env_name == "CartPole-v1":

            self.feature_layer = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
            )

            self.value_stream = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, num_actions)
            )

        elif env_name == "ALE/Pong-v5":

            self.feature_layer = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

            self.value_stream = nn.Sequential(
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, 1)
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions)
            )

    def forward(self, x):
        if x.dim() == 4:
            x = x / 255.0

        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class RainbowDQN(nn.Module):

    def __init__(self, num_actions, n_atoms, v_min, v_max, env_name="ALE/Pong-v5"):
        super(RainbowDQN, self).__init__()
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        self.support = torch.linspace(v_min, v_max, n_atoms)

        if env_name == "ALE/Pong-v5":
            self.feature_layer = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            feature_size = 64 * 7 * 7
            noisy_layer_size = 512

            self.value_stream_fc = NoisyLinear(feature_size, noisy_layer_size)
            self.value_stream_out = NoisyLinear(noisy_layer_size, n_atoms)

            self.advantage_stream_fc = NoisyLinear(feature_size, noisy_layer_size)
            self.advantage_stream_out = NoisyLinear(
                noisy_layer_size, num_actions * n_atoms
            )

    def forward(self, x):
        if x.dim() == 4:
            x = x / 255.0
        batch_size = x.size(0)

        features = self.feature_layer(x)

        # Value Stream with Noisy Layers
        value_fc = F.relu(self.value_stream_fc(features))
        value_dist = self.value_stream_out(value_fc).view(batch_size, 1, self.n_atoms)

        # Advantage Stream with Noisy Layers
        advantage_fc = F.relu(self.advantage_stream_fc(features))
        advantage_dist = self.advantage_stream_out(advantage_fc).view(
            batch_size, self.num_actions, self.n_atoms
        )

        # Dueling & Distributional Combination
        dist_logits = value_dist + (
            advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        )
        log_probs = F.log_softmax(dist_logits, dim=2)

        return log_probs

    def get_q_values(self, x):
        log_probs = self.forward(x)
        probs = torch.exp(log_probs)
        support = self.support.to(x.device)
        q_values = (probs * support).sum(2)
        return q_values

    def reset_noise(self):
        # Rest NoisyLinear Layers noise
        self.value_stream_fc.reset_noise()
        self.value_stream_out.reset_noise()
        self.advantage_stream_fc.reset_noise()
        self.advantage_stream_out.reset_noise()


def compute_distributional_loss(
    q_net,
    target_net,
    states,
    actions,
    rewards,
    next_states,
    dones,
    gamma,
    support,
    delta_z,
    v_min,
    v_max,
    n_atoms,
    batch_size,
    is_weights=None,
):
    """
    compute_distributional_loss
    :return: (loss, abs_errors) a tuple containing the final loss and the absolute errors for PER
    """
    device = next(q_net.parameters()).device
    with torch.no_grad():

        next_q_values = q_net.get_q_values(next_states)
        next_actions = next_q_values.argmax(1)
        next_dist = torch.exp(target_net(next_states)[range(batch_size), next_actions])

        Tz = rewards.unsqueeze(1) + (
            1 - dones.unsqueeze(1)
        ) * gamma * support.unsqueeze(0)
        Tz = Tz.clamp(v_min, v_max)

        b = (Tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l[(u > 0) * (l == u)] -= 1
        u[(l < (n_atoms - 1)) * (l == u)] += 1

        m = torch.zeros(batch_size, n_atoms, device=device)
        offset = (
            torch.linspace(0, (batch_size - 1) * n_atoms, batch_size)
            .long()
            .unsqueeze(1)
            .to(device)
        )

        m.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        m.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

    log_pred_dist = q_net(states)[range(batch_size), actions]

    loss_per_sample = -(m * log_pred_dist).sum(1)

    abs_errors = loss_per_sample.detach().cpu().numpy()

    if is_weights is not None:
        loss = (is_weights.squeeze(1) * loss_per_sample).mean()
    else:
        loss = loss_per_sample.mean()

    return loss, abs_errors
