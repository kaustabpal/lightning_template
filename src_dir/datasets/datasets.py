#!/usr/bin/env python3
# @brief:    Pytorch Lightning module for datasets
# @author    Kaustab Pal    [kaustab21@gmail.com]
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

class TemplateDatasetModule(pl.LightningDataModule):
    """A Pytorch Lightning module for Datasets"""

    def __init__(self, cfg):
        """Method to initizalize the dataset class

        Args:
          cfg: config dict

        Returns:
          None
        """
        super(TemplateDatasetModule, self).__init__()
        self.cfg = cfg


    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        ########## Point dataset splits
        train_set = TemplatePytorchDataset(self.cfg, split="train")

        val_set = TemplatePytorchDataset(self.cfg, split="val")

        test_set = TemplatePytorchDataset(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,#self.cfg["DATA_CONFIG"]["DATALOADER"]["SHUFFLE"],
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=False
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=True
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
            persistent_workers=True
        )

        self.test_iter = iter(self.test_loader)

        # Optionally compute statistics of training data
        #if self.cfg["DATA_CONFIG"]["COMPUTE_MEAN_AND_STD"]:
        #    compute_mean_and_std(self.cfg, self.train_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class TemplatePytorchDataset(Dataset):
    """Dataset class for a dataset"""

    def __init__(self, cfg, split):
        """Read parameters and data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = os.environ.get("DATA_PROCESSED")
        self.dataset_size = 0
        if split == "train":
            pass
        elif split == "val":
            pass
        elif split == "test":
            pass
        else:
            raise Exception("Split must be train/val/test")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Generic function to get items from a dataset

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """

        item = {"input": input_data, "target_output": output_data}
        return item



if __name__ == "__main__":
    config_filename = # location of config file
    cfg = yaml.safe_load(open(config_filename))
    data = TemplatePytorchDataset(cfg, "train")
    print(len(data))
    data = TemplateDatasetModule(cfg)
    data.setup()
    train_loader = data.train_loader
    for i, batch in enumerate(train_loader):
        past = batch["input"]
        fut = batch["target_output"]
        print(past.shape)
        print(fut.shape)
        exit()


