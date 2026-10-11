import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # 2 Hidden layers with 64 nodes each is sufficient for this complexity
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma # Discount factor for future rewards
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(100000)

    def act(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_size) # Explore
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item() # Exploit best known action

    def train_step(self, batch_size):
        """Samples a batch from memory and performs one backpropagation step."""
        if len(self.memory) < batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        curr_q = self.q_network(states).gather(1, actions)
        
        # Max Next Q values (Target)
        with torch.no_grad():
            max_next_q = self.q_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
            
        # Compute Huber loss (less sensitive to outliers than MSE)
        loss = F.smooth_l1_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath, episode=0, epsilon=1.0):
        checkpoint = {
            "model_state_dict": self.q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.q_network.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.q_network.train()
            saved_episode = checkpoint.get("episode", 0)
            saved_epsilon = checkpoint.get("epsilon", 0.2)
            return saved_episode, saved_epsilon
        else:
            self.q_network.load_state_dict(checkpoint)
            self.q_network.train()
            return 0, 0.2 