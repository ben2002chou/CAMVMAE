import argparse
import os
import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.data import WeightedRandomSampler

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader_piano_roll as dataloader
import models
import numpy as np
from traintest_cavmae_piano_roll import train

sum_vals = 0.0
sum_square_vals = 0.0
num_elements = 0

data_train = "/home/ben2002chou/code/cav-mae/data/cocochorals/cocochorals_test.json"
te_data = "/home/ben2002chou/code/cav-mae/data/cocochorals/audioset_eval_cocochorals_valid.json"
label_csv = (
    "/home/ben2002chou/code/cav-mae/data/cocochorals/class_labels_indices_combined.csv"
)
audio_conf = {
    "num_mel_bins": 128,
    "target_length": 512,
    "freqm": 0,
    "timem": 0,
    "mixup": False,
    "dataset": "cocochorals",
    "mode": "train",
    "noise": False,
    "label_smooth": 0,
    "im_res": 224,
    "skip_norm": True,
}

train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(data_train, label_csv=label_csv, audio_conf=audio_conf),
    batch_size=128,
    num_workers=32,
    pin_memory=True,
    drop_last=True,
)

for (
    a1_input,
    a2_input,
    v_input,
    _,
) in train_loader:  # Replace 'data_loader' with your PyTorch DataLoader
    batch = a2_input
    batch = batch.float()  # Ensure the batch is a float tensor
    sum_vals += torch.sum(batch)
    sum_square_vals += torch.sum(batch**2)
    num_elements += torch.numel(batch)

mean = sum_vals / num_elements
std_dev = torch.sqrt((sum_square_vals / num_elements) - (mean**2))

print(mean)
print(std_dev)
