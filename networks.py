import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper-parameters

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-7  # for soft update of target parameters
LR = 1e-3  # learning rate
UPDATE_EVERY = 1  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(os.path.join(os.getcwd(), 'logits')):
    os.mkdir(os.path.join(os.getcwd(), 'logits'))
logits_path = os.path.join(os.getcwd(), 'logits')

if not os.path.exists(os.path.join(os.getcwd(), 'logits', 'action_values')):
    os.mkdir(os.path.join(os.getcwd(), 'logits', 'action_values'))
action_saving_path = os.path.join(os.getcwd(), 'logits', 'action_values')


class QNetwork(nn.Module):
    def __init__(self, state_size=12, hidden_dim=128, action_size=10, seed=0):
        super(QNetwork, self).__init__()
        fc1_units = state_size
        fc2_units = hidden_dim

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc2_units)
        self.fc4 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.seed = seed

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

