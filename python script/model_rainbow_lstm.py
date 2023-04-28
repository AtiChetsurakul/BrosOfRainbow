import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NoisyLinear

class Network_lstm(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network_lstm, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

# Accroding to paper
        self.feature_layer = nn.Sequential(
        nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=512, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 128), # Output shape is [64, out_dim] 
        nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128, std_init=0.5)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size, std_init=0.5)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128, std_init=0.5)
        self.value_layer = NoisyLinear(128, atom_size, std_init=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2) # [64, 7]
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        feature = self.lstm(feature)
        
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        # [64, 7, 51]
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()