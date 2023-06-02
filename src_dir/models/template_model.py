#!/usr/bin/env python3
# @brief:     Generic pytorch module for NN
# @author     Kaustab Pal    [kaustab21@gmail.com]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
from src_dir.models.base import BaseModel


class GenericModel(BaseModel):
    def __init__(self, cfg):
        """Init all layers needed for range image-based point cloud prediction"""
        super().__init__(cfg)

    def forward(self, x):
        """Forward range image-based point cloud prediction

        Args:
            x (torch.tensor): Input tensor of concatenated, unnormalize range images

        Returns:
            dict: Containing the predicted range tensor and mask logits
        """
        pass



class Normalization(nn.Module):
    """Custom Normalization layer to enable different normalization strategies"""

    def __init__(self, cfg, n_channels):
        """Init custom normalization layer"""
        super(Normalization, self).__init__()
        self.cfg = cfg
        self.norm_type = self.cfg["MODEL"]["NORM"]
        n_channels_per_group = self.cfg["MODEL"]["N_CHANNELS_PER_GROUP"]

        if self.norm_type == "batch":
            self.norm = nn.BatchNorm3d(n_channels)
        elif self.norm_type == "instance":
            self.norm = nn.InstanceNorm3d(n_channels)
        elif self.norm_type == "none":
            self.norm = nn.Identity()

    def forward(self, x):
        """Forward normalization pass

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        x = self.norm(x)
        return x


