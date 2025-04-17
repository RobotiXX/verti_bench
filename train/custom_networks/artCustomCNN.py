import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=32):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)
        
        input_vector_size = observation_space.shape[0]
        
        # Define a simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(input_vector_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Process observations through the network
        return self.network(observations)
