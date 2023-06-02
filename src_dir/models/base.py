#!/usr/bin/env python3
# @brief:    Generic base class for lightning
# @author    Kaustab Pal    [kaustab21@gmail.com]

import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl
from src_dir.models.loss import Loss

class BaseModel(pl.LightningModule):
    """Pytorch Lightning base model"""

    def __init__(self, cfg):
        """Init base model

        Args:
            cfg (dict): Config parameters
        """
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.loss = Loss(self.cfg)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        """Optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["TRAIN"]["LR"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg["TRAIN"]["LR_EPOCH"],
            gamma=self.cfg["TRAIN"]["LR_DECAY"],
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning training step including logging

        Args:
            batch (dict): A dict with a batch of training samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        input_data = batch["input"]
        target_output = batch["target_output"]
        pred_output = self.forward(past)
        loss = self.loss(target_output, pred_output)
        self.log("train/loss", loss["loss"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch Lightning validation step including logging

        Args:
            batch (dict): A dict with a batch of validation samples
            batch_idx (int): Index of batch in dataset

        Returns:
            None
        """
        input_data = batch["input"]
        target_output = batch["target_output"]
        pred_output = self.forward(past)
        loss = self.loss(target_output, pred_output,"val", self.current_epoch)

        self.log("val/loss", loss["loss"], prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning test step including logging

        Args:
            batch (dict): A dict with a batch of test samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        input_data = batch["input"]
        target_output = batch["target_output"]

        batch_size, n_inputs, n_future_steps, H, W = input_data.shape

        start = time.time()
        pred_output = self.forward(past)
        inference_time = (time.time() - start) / batch_size
        self.log("test/inference_time", inference_time, on_epoch=True)

        loss = self.loss(pred_output, target_output, "test", self.current_epoch)

        self.log("test/loss", loss["loss"], prog_bar=True, on_epoch=True)

        return loss

    def test_epoch_end(self, outputs):
        pass
