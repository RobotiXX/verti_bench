import os
import random
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from typing import List  # For List
from torch import nn, Tensor  # For nn and Tensor
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from torch import distributions as dist
from torch import nn
from abc import abstractmethod
from typing import Callable, List, Any, Optional, Sequence, Type

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class SWAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 reg_weight: int = 100,
                 wasserstein_deg: float= 2.,
                 num_projections: int = 200,
                 projection_dist: str = 'normal',
                    **kwargs) -> None:
        super(SWAE, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.p = wasserstein_deg
        self.num_projections = num_projections
        self.proj_dist = projection_dist

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        return z

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return  [self.decode(z), input, z]

    def loss_function(self, recons, input, z) -> dict:
        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recons_loss_l2 = F.mse_loss(recons, input)
        recons_loss_l1 = F.l1_loss(recons, input)

        swd_loss = self.compute_swd(z, self.p, reg_weight)

        loss = recons_loss_l2 + recons_loss_l1 + swd_loss
        return loss

    def get_random_projections(self, latent_dim: int, num_samples: int) -> Tensor:
        """
        Returns random samples from latent distribution's (Gaussian)
        unit sphere for projecting the encoded samples and the
        distribution samples.

        :param latent_dim: (Int) Dimensionality of the latent space (D)
        :param num_samples: (Int) Number of samples required (S)
        :return: Random projections from the latent unit sphere
        """
        if self.proj_dist == 'normal':
            rand_samples = torch.randn(num_samples, latent_dim)
        elif self.proj_dist == 'cauchy':
            rand_samples = dist.Cauchy(torch.tensor([0.0]),
                                       torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
        else:
            raise ValueError('Unknown projection distribution.')

        rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1,1)
        return rand_proj # [S x D]


    def compute_swd(self,
                    z: Tensor,
                    p: float,
                    reg_weight: float) -> Tensor:
        """
        Computes the Sliced Wasserstein Distance (SWD) - which consists of
        randomly projecting the encoded and prior vectors and computing
        their Wasserstein distance along those projections.

        :param z: Latent samples # [N  x D]
        :param p: Value for the p^th Wasserstein distance
        :param reg_weight:
        :return:
        """
        prior_z = torch.randn_like(z) # [N x D]
        device = z.device

        proj_matrix = self.get_random_projections(self.latent_dim,
                                                  num_samples=self.num_projections).transpose(0,1).to(device)

        latent_projections = z.matmul(proj_matrix) # [N x S]
        prior_projections = prior_z.matmul(proj_matrix) # [N x S]

        # The Wasserstein distance is computed by sorting the two projections
        # across the batches and computing their element-wise l2 distance
        w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
                 torch.sort(prior_projections.t(), dim=1)[0]
        w_dist = w_dist.pow(p)
        return reg_weight * w_dist.mean()
    
    def freeze_encoder(self):
        for param in list(self.encoder.parameters()) + list(self.fc_z.parameters()):
            param.requires_grad = False


class LatentSpaceMapper(nn.Module):
    """
    Not recommended to only use Fully Connected layers, 
    adding ReLU activations to introduce non-linearity.
    """
    def __init__(self, input_dim, output_dim):
        super(LatentSpaceMapper, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  
            nn.ReLU(),
            nn.Linear(128, 64),  
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh() # [-1, 1]
        )
        
    def forward(self, x):
        return self.network(x)

