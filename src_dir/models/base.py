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
        t1 = time.time()
        pred_output = self.forward(past)
        t2 = time.time()
        print("Time: ", t2-t1)
        loss = self.loss(target_output, pred_output)
        #self.log("train/loss", loss["loss"])
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

        self.log("val/loss", loss["loss"], on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning test step including logging

        Args:
            batch (dict): A dict with a batch of test samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        past = batch["past_data"]
        future = batch["fut_data"]

        batch_size, n_inputs, n_future_steps, H, W = past.shape

        start = time.time()
        output = self.forward(past)
        inference_time = (time.time() - start) / batch_size
        self.log("test/inference_time", inference_time, on_epoch=True)

        loss = self.loss(output, future, "test", self.current_epoch)

        #self.log("test/loss_range_view", loss["loss_range_view"], on_epoch=True)
        #self.log("test/loss_mask", loss["loss_mask"], on_epoch=True)

        for step, value in loss["chamfer_distance"].items():
            self.log("test/chamfer_distance_{:d}".format(step), value, on_epoch=True)

        self.log(
            "test/mean_chamfer_distance", loss["mean_chamfer_distance"], on_epoch=True
        )
        self.log(
            "test/final_chamfer_distance", loss["final_chamfer_distance"], on_epoch=True
        )

        self.chamfer_distances_tensor = torch.cat(
            (self.chamfer_distances_tensor, loss["chamfer_distances_tensor"]), dim=1
        )

        #print(self.cfg["TEST"]["SAVE_POINT_CLOUDS"])
        if self.cfg["TEST"]["SAVE_POINT_CLOUDS"]:
            #save_point_clouds(self.cfg, self.projection, batch, output)

            sequence_batch, frame_batch = batch["meta"]
            for sample_idx in range(frame_batch.shape[0]):
                sequence = sequence_batch[sample_idx].item()
                frame = frame_batch[sample_idx].item()
                save_range_and_mask(
                    self.cfg,
                    self.projection,
                    batch,
                    output,
                    sample_idx,
                    sequence,
                    frame,
                )

        return loss

    def test_epoch_end(self, outputs):
        # Remove first row since it was initialized with zero
        self.chamfer_distances_tensor = self.chamfer_distances_tensor[:, 1:]
        n_steps, _ = self.chamfer_distances_tensor.shape
        mean = torch.mean(self.chamfer_distances_tensor, dim=1)
        std = torch.std(self.chamfer_distances_tensor, dim=1)
        q = torch.tensor([0.25, 0.5, 0.75])
        quantile = torch.quantile(self.chamfer_distances_tensor, q, dim=1)

        chamfer_distances = []
        for s in range(n_steps):
            chamfer_distances.append(self.chamfer_distances_tensor[s, :].tolist())
        print("Final size of CD: ", self.chamfer_distances_tensor.shape)
        print("Mean :", mean)
        print("Std :", std)
        print("Quantile :", quantile)

        testdir = os.path.join(self.cfg["LOG_DIR"], "test")
        if not os.path.exists(testdir):
            os.makedirs(testdir)

        filename = os.path.join(
            testdir, "stats_" + time.strftime("%Y%m%d_%H%M%S") + ".yml"
        )

        log_to_save = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "quantile": quantile.tolist(),
            "chamfer_distances": chamfer_distances,
        }
        with open(filename, "w") as yaml_file:
            yaml.dump(log_to_save, yaml_file, default_flow_style=False)
