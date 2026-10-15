import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
import os
from collections import deque

# =============================================================================
#  1. REPLAY BUFFER FOR CONTINUOUS CONTROL
# =============================================================================
class ReplayBuffer:
    """
    Stores transitions (state, action, reward, next_state, done).
    In continuous control, 'action' is a float vector [throttle, steering]
    instead of a single discrete integer.
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
#  2. CRITIC (TWIN Q-NETWORKS)
# =============================================================================
class Critic(nn.Module):
    """
    Clipped Double-Q Networks:
    Evaluates Q(s, a). SAC maintains TWO independent critics (Q1 and Q2)
    to combat Q-value overestimation bias. When computing target values,
    we take min(Q1_target, Q2_target).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()

        # Q1 Architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 Architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        # Q1 forward pass
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1_out(q1)

        # Q2 forward pass
        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2_out(q2)

        return q1, q2


# =============================================================================
#  3. ACTOR (GAUSSIAN POLICY WITH REPARAMETERIZATION)
# =============================================================================
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
    """
    Actor Network:
    Given a state 's', outputs the parameters (mean, log_std) of a Gaussian
    distribution. It then samples an action and squashes it through tanh()
    so actions are smoothly constrained to [-1.0, 1.0].
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamp log_std to prevent numerical instabilities / exploding gradients
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Reparameterization Trick:
        Action is sampled as a = tanh(mu + sigma * eps), where eps ~ N(0, 1).
        This allows gradients to backpropagate directly into the policy network!
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # rsample() uses the reparameterization trick: eps ~ N(0, 1)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        # Compute log probability with Jacobian correction for tanh squashing:
        # log pi(a|s) = log p(u|s) - sum(log(1 - tanh(u)^2 + epsilon))
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds derivative correction
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Deterministic evaluation mean action (for testing / greedy exploit)
        mean_action = torch.tanh(mean)

        return action, log_prob, mean_action


# =============================================================================
#  4. SOFT ACTOR-CRITIC (SAC) AGENT
# =============================================================================
class SACAgent:
    """
    Soft Actor-Critic Agent (Haarnoja et al., 2018)
    Key pillars:
      1. Actor-Critic framework for continuous action spaces.
      2. Clipped Double-Q trick to prevent Q-value overestimation.
      3. Maximum Entropy RL: maximizes E[Reward + alpha * Entropy].
      4. Automatic temperature (alpha) tuning to maintain optimal exploration.
      5. Soft target updates (Polyak averaging) for smooth stability.
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_entropy=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Critic and Target Critic Networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        # Initialize target networks with identical weights
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 2. Actor (Policy) Network
        self.actor = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # 3. Automatic Entropy Adjustment (alpha)
        # Target entropy is set to -dim(A) according to the SAC paper
        if self.auto_entropy:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # 4. Experience Replay
        self.memory = ReplayBuffer(capacity=100000)

    def select_action(self, state, evaluate=False):
        """
        Selects an action for the given state.
        If evaluate=True, uses the deterministic mean (greedy evaluation).
        If evaluate=False, samples stochastically with entropy exploration.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0]

    def train_step(self, batch_size):
        """
        Performs one gradient update step for Critic, Actor, and Entropy Alpha.
        """
        if len(self.memory) < batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ---------------------------------------------------------------------
        # 1. UPDATE CRITIC NETWORKS
        # ---------------------------------------------------------------------
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            # Take the minimum of the two target critics (Clipped Double-Q)
            min_target_q = torch.min(target_q1, target_q2)
            # Maximize Reward + Entropy: y = r + gamma * (min_Q - alpha * log_pi)
            target_q = rewards + (1.0 - dones) * self.gamma * (min_target_q - self.alpha * next_log_prob)

        # Current Q predictions
        current_q1, current_q2 = self.critic(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------------------------------------------------
        # 2. UPDATE ACTOR NETWORK
        # ---------------------------------------------------------------------
        # Actor aims to maximize E[min_Q(s, a) - alpha * log_pi(a|s)]
        # Equivalently minimize: E[alpha * log_pi(a|s) - min_Q(s, a)]
        new_actions, log_prob, _ = self.actor.sample(states)
        new_q1, new_q2 = self.critic(states, new_actions)
        min_new_q = torch.min(new_q1, new_q2)

        actor_loss = (self.alpha * log_prob - min_new_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------------------------------------------------
        # 3. UPDATE ENTROPY TEMPERATURE (ALPHA)
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.auto_entropy:
            # Objective: J(alpha) = -alpha * (log_pi + target_entropy)
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # ---------------------------------------------------------------------
        # 4. SOFT UPDATE TARGET CRITIC (POLYAK AVERAGING)
        # target_params = tau * params + (1 - tau) * target_params
        # ---------------------------------------------------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_val
        }

    def save(self, filepath, episode=0):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "episode": episode
        }
        if self.auto_entropy:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        if not os.path.exists(filepath):
            return 0
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if self.auto_entropy and "log_alpha" in checkpoint:
                self.log_alpha = checkpoint["log_alpha"].to(self.device)
                self.alpha = self.log_alpha.exp().item()
                if "alpha_optimizer" in checkpoint:
                    self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            saved_episode = checkpoint.get("episode", 0)
            return saved_episode
        except Exception as e:
            print(f"[Warning] Could not load SAC checkpoint from '{filepath}': {e}")
            return 0
