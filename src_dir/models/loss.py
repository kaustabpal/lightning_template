import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math
import random


class Loss(nn.Module):
    """Combined loss for the task"""

    def __init__(self, cfg):
        """Init"""
        super().__init__()
        self.cfg = cfg

        self.loss_l1 = loss_l1(self.cfg)
        self.loss_bce = loss_bce(self.cfg)

    def forward(self, output, target, mode, epoch_number=40):
        """Forward pass with multiple loss components

        Args:
        output (dict): Predicted mask logits and ranges
        target (torch.tensor): Target range image
        mode (str): Mode (train,val,test)

        Returns:
        dict: Dict with loss components
        """

        # L1 Loss
        loss_l1 = self.loss_l1(output, target, epoch_number)

        # Binary-Cross Entropy Loss
        loss_bce = self.loss_bce(output, target, epoch_number)

        loss = ( # Modify as per your need
            self.loss_weight_l1 * loss_l1
            + self.loss_weight_bce * loss_bce
        )

        loss_dict = {
            "loss": loss,
            "loss_l1": loss_l1.detach(),
            "loss_bce": loss_bce.detach(),
        }
        return loss_dict


class loss_bce(nn.Module):
    """Binary cross entropy loss for prediction of valid mask"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, output, target, epoch_number):
        loss = self.loss(output, target)
        return loss


class loss_l1(nn.Module):
    """L1 loss for range image prediction"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, output, target, epoch_number):
        loss = self.loss(output, target)
        return loss
