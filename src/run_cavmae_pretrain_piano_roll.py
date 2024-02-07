# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

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
import wandb
from lightning.fabric import Fabric  # Importing Fabric

# set the default precision to utilize tensorcores
torch.set_float32_matmul_precision("medium")
    
def parse_args():
    """
    Parse arguments given to the script.

    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-train", type=str, default="", help="training data json")
    parser.add_argument("--data-val", type=str, default="", help="validation data json")
    parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
    parser.add_argument("--label-csv", type=str, default="", help="csv with class labels")
    parser.add_argument("--n_class", type=int, default=527, help="number of classes")
    parser.add_argument("--model", type=str, default="ast", help="the model used")
    parser.add_argument(
        "--dataset",
        type=str,
        default="audioset",
        help="the dataset used",
        choices=[
            "audioset",
            "esc50",
            "speechcommands",
            "fsd50k",
            "vggsound",
            "epic",
            "k400",
            "msrvtt",
            "cocochorals",
        ],
    )
    parser.add_argument(
        "--dataset_mean",
        type=float,
        help="the dataset audio spec mean, used for input normalization",
    )
    parser.add_argument(
        "--dataset_std",
        type=float,
        help="the dataset audio spec std, used for input normalization",
    )
    parser.add_argument("--target_length", type=int, help="the input length in frames")
    parser.add_argument("--noise", help="if use balance sampling", type=ast.literal_eval)

    parser.add_argument(
        "--exp-dir", type=str, default="", help="directory to dump experiments"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="training optimizer",
        choices=["sgd", "adam"],
    )
    parser.add_argument(
        "-b", "--batch-size", default=12, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        default=24,  # 32
        type=int,
        metavar="NW",
        help="# of workers for dataloading (default: 32)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=1, help="number of maximum training epochs"
    )
    # not used in the formal experiments, only for preliminary experiments
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=2,
        help="how many epoch to wait to reduce lr if mAP doesn't improve",
    )
    parser.add_argument(
        "--lr_adapt", help="if use adaptive learning rate", type=ast.literal_eval
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mAP",
        help="the main evaluation metrics in finetuning",
        choices=["mAP", "acc"],
    )
    parser.add_argument(
        "--warmup",
        help="if use warmup learning rate scheduler",
        type=ast.literal_eval,
        default="True",
    )
    parser.add_argument(
        "--lrscheduler_start",
        default=10,
        type=int,
        help="when to start decay in finetuning",
    )
    parser.add_argument(
        "--lrscheduler_step",
        default=5,
        type=int,
        help="the number of step to decrease the learning rate in finetuning",
    )
    parser.add_argument(
        "--lrscheduler_decay",
        default=0.5,
        type=float,
        help="the learning rate decay ratio in finetuning",
    )
    parser.add_argument(
        "--n-print-steps", type=int, default=100, help="number of steps to print statistics"
    )
    parser.add_argument("--save_model", help="save the model or not", type=ast.literal_eval)

    parser.add_argument(
        "--mixup",
        type=float,
        default=0,
        help="how many (0-1) samples need to be mixup during training",
    )
    parser.add_argument(
        "--bal", type=str, default=None, help="use balanced sampling or not"
    )

    parser.add_argument(
        "--cont_model", help="previous pretrained model", type=str, default=None
    )
    parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
    parser.add_argument(
        "--norm_pix_loss", help="if use norm_pix_loss", type=ast.literal_eval, default=None
    )
    parser.add_argument(
        "--pretrain_path", type=str, default="None", help="pretrained model path"
    )
    parser.add_argument(
        "--contrast_loss_weight",
        type=float,
        default=0.01,
        help="weight for contrastive loss",
    )
    parser.add_argument(
        "--mae_loss_weight", type=float, default=3.0, help="weight for mae loss"
    )
    parser.add_argument(
        "--tr_pos",
        help="if use trainable positional embedding",
        type=ast.literal_eval,
        default=None,
    )
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="masking ratio")
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="unstructured",
        help="masking ratio",
        choices=["unstructured", "time", "freq", "tf"],
    )
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=[
            "32-true",
            "32",
            "16-mixed",
            "bf16-mixed",
            "transformer-engine",
            "16-true",
            "bf16-true",
            "64-true",
        ],
        default="bf16-mixed",
    )
    parser.add_argument("--log-all", type=bool, default=True, help="wandb logging for all gpus")
    

    args = parser.parse_args()
    return args

def setup_run(args):
    if args.log_all:
        run = wandb.init(
        project="CAVMAE",
        config={
            **vars(args),
        },
        group='DDP'
        
    )
    else:
        if args.local_rank == 0:
            run = wandb.init(
            project="CAVMAE",
            config={
            **vars(args),
        },

        )
        else:
            run = None

    return run

# pretrain cav-mae model
def main(args):
    # Initialize Fabric
    fabric = Fabric(
        accelerator="auto",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
    )
    fabric.launch()
    fabric.seed_everything(0)
    fabric.print(
        "I am process %s, running on %s: starting (%s)"
        % (os.getpid(), os.uname()[1], time.asctime())
    )
    
    run = setup_run(args)

    im_res = 224
    audio_conf = {
        "num_mel_bins": 128,
        "target_length": args.target_length,
        "freqm": 0,
        "timem": 0,
        "mixup": args.mixup,
        "dataset": args.dataset,
        "mode": "train",
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "noise": args.noise,
        "label_smooth": 0,
        "im_res": im_res,
    }
    val_audio_conf = {
        "num_mel_bins": 128,
        "target_length": args.target_length,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "dataset": args.dataset,
        "mode": "eval",
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "noise": False,
        "im_res": im_res,
    }

    fabric.print(
        "current mae loss {:.3f}, and contrastive loss {:.3f}".format(
            args.mae_loss_weight, args.contrast_loss_weight
        )
    )

    if args.bal == "bal":
        fabric.print("balanced sampler is being used")
        if args.weight_file == None:
            samples_weight = np.loadtxt(args.data_train[:-5] + "_weight.csv", delimiter=",")
        else:
            samples_weight = np.loadtxt(
                args.data_train[:-5] + "_" + args.weight_file + ".csv", delimiter=","
            )
        sampler = WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True
        )

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(
                args.data_train, label_csv=args.label_csv, audio_conf=audio_conf
            ),
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
        )
    else:
        fabric.print("balanced sampler is not used")
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(
                args.data_train, label_csv=args.label_csv, audio_conf=audio_conf
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    if args.data_eval != None:
        eval_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(
                args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    if args.model == "cav-mae":
        fabric.print(
            "pretrain a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers"
        )
        audio_model = models.CAVMAE(
            audio_length=args.target_length,
            norm_pix_loss=args.norm_pix_loss,
            modality_specific_depth=11,
            tr_pos=args.tr_pos,
        )
    else:
        raise ValueError("model not supported")
    # TODO: Check if we can add pretrained model without errors
    # Optimize model for training

    # initialized with a pretrained checkpoint (e.g., original vision-MAE checkpoint)
    if args.pretrain_path != "None":
        mdl_weight = torch.load(args.pretrain_path, map_location=torch.device("cpu")) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_model = audio_model.to(device)
        if not isinstance(audio_model, torch.nn.parallel.DistributedDataParallel):
            audio_model = torch.nn.parallel.DistributedDataParallel(audio_model)
        audio_model = audio_model.to(device)
        miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
        fabric.print("now load mae pretrained weights from ", args.pretrain_path)
        fabric.print(miss, unexpected)
    
    # if args.cont_model != None:
    #     print('now load pretrained weights from : ' + args.cont_model)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     sdA = fabric.load(args.cont_model, map_location=device)
    #     if isinstance(audio_model, torch.nn.DataParallel) == False:
    #         audio_model = torch.nn.DataParallel(audio_model)
    #     audio_model.load_state_dict(sdA, strict=True)

    fabric.print("\nCreating experiment directory: %s" % args.exp_dir)
    try:
        os.makedirs("%s/models" % args.exp_dir)
    except:
        pass
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)
    with open(args.exp_dir + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    fabric.print("Now starting training for {:d} epochs.".format(args.n_epochs))
    
    
    train(audio_model, train_loader, val_loader, args, fabric, run)
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)