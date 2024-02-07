# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# not rely on supervised feature

import sys
import os
import datetime

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
from torch.nn import CTCLoss
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
from lightning.fabric import Fabric  # Importing Fabric

def train(audio_model, train_loader, test_loader, args, fabric, run=None):
    device = fabric.device
    fabric.print("running on " + str(device))
    torch.set_grad_enabled(True)

    (
        batch_time,
        per_sample_time,
        data_time,
        per_sample_data_time,
        per_sample_dnn_time,
    ) = (AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter())
    loss_av_meter, loss_a1_meter, loss_a2_meter, loss_v_meter, loss_c_meter = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append(
            [epoch, global_step, best_epoch, best_loss, time.time() - start_time]
        )
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    audio_model = fabric.to_device(audio_model)
    if not isinstance(audio_model, nn.parallel.DistributedDataParallel):
        audio_model = nn.parallel.DistributedDataParallel(audio_model)

    audio_model = fabric.to_device(audio_model)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    fabric.print(
        "Total parameter number is : {:.3f} million".format(
            sum(p.numel() for p in audio_model.parameters()) / 1e6
        )
    )
    fabric.print(
        "Total trainable parameter number is : {:.3f} million".format(
            sum(p.numel() for p in trainables) / 1e6
        )
    )
    optimizer = torch.optim.AdamW(
        trainables, args.lr, weight_decay=5e-2
    )

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience, verbose=True
        )
        fabric.print("Override to use adaptive learning rate scheduler.")
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
            gamma=args.lrscheduler_decay,
        )
        fabric.print(
            "The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches".format(
                args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step
            )
        )

    fabric.print(
        "now training with {:s}, learning rate scheduler: {:s}".format(
            str(args.dataset), str(scheduler)
        )
    )

    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # fabric.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
    #epoch = 12 FOR DEBUGGING error saving model at 13th epoch
    epoch += 1
    # scaler = GradScaler()

    fabric.print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    fabric.print("start training...")
    result = np.zeros([args.n_epochs, 12])  # for each epoch, 12 metrics to record

    # Setup model and optimizer with Fabric
    audio_model, optimizer = fabric.setup(audio_model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)

    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        fabric.print("---------------")
        fabric.print(datetime.datetime.now())
        fabric.print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        fabric.print(
            "current masking ratio is {:.3f} for both modalities; audio mask mode {:s}".format(
                args.masking_ratio, args.mask_mode
            )
        )
        fabric.print("start dataloader")
        fabric.print("train loader length is %s" % len(train_loader))
        for i, (a1_input, a2_input, v_input, _) in enumerate(train_loader):
            
            batch_size = a1_input.size(0)
            a1_input = a1_input.to(device, non_blocking=True)
            a2_input = a2_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a1_input.shape[0])
            dnn_start_time = time.time()
            fabric.print("start forward")
           
            (
                loss,
                loss_mae,
                loss_mae_a1,
                loss_mae_a2,
                loss_mae_v,
                loss_c,
                mask_a1,
                mask_a2,
                mask_v,
                c_acc,
            ) = audio_model(
                a1_input,
                a2_input,
                v_input,
                args.masking_ratio,
                args.masking_ratio,
                mae_loss_weight=args.mae_loss_weight,
                contrast_loss_weight=args.contrast_loss_weight,
                mask_mode=args.mask_mode,
            )
            # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually TODO: Check if this is still the case
            # loss, loss_mae, loss_mae_a1, loss_mae_a2, loss_mae_v, loss_c, c_acc = (
            #     loss.sum(),
            #     loss_mae.sum(),
            #     loss_mae_a1.sum(),
            #     loss_mae_a2.sum(),
            #     loss_mae_v.sum(),
            #     loss_c.sum(),
            #     c_acc.mean(),
            # )

            optimizer.zero_grad(set_to_none=True)
            # scaler.scale(loss).backward()
            fabric.backward(loss)  # TODO: why does .sum twice work?
            optimizer.step()
            # scaler.update()

            # loss_av is the main loss
            loss_av_meter.update(loss.item(), batch_size)
            loss_a1_meter.update(loss_mae_a1.item(), batch_size)
            loss_a2_meter.update(loss_mae_a2.item(), batch_size)
            loss_v_meter.update(loss_mae_v.item(), batch_size)
            loss_c_meter.update(loss_c.item(), batch_size)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / a1_input.shape[0])
            per_sample_dnn_time.update(
                (time.time() - dnn_start_time) / a1_input.shape[0]
            )
            # TODO: Check if this is messing up the memory
            # del a1_input, a2_input, v_input, loss
            # torch.cuda.empty_cache()  # Use this sparingly

            print_step = global_step % args.n_print_steps == 0
            early_print_step = (
                epoch == 0 and global_step % (args.n_print_steps / 10) == 0
            )
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                fabric.print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Per Sample Total Time {per_sample_time.avg:.5f}\t"
                    "Per Sample Data Time {per_sample_data_time.avg:.5f}\t"
                    "Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t"
                    "Train Total Loss {loss_av_meter.val:.4f}\t"
                    "Train MAE Loss Audio {loss_a1_meter.val:.4f}\t"
                    "Train MAE Loss Midi Audio {loss_a2_meter.val:.4f}\t"
                    "Train MAE Loss Visual {loss_v_meter.val:.4f}\t"
                    "Train Contrastive Loss {loss_c_meter.val:.4f}\t"
                    "Train Contrastive Accuracy {c_acc:.4f}\t".format(
                        epoch,
                        i,
                        len(train_loader),
                        per_sample_time=per_sample_time,
                        per_sample_data_time=per_sample_data_time,
                        per_sample_dnn_time=per_sample_dnn_time,
                        loss_av_meter=loss_av_meter,
                        loss_a1_meter=loss_a1_meter,
                        loss_a2_meter=loss_a2_meter,
                        loss_v_meter=loss_v_meter,
                        loss_c_meter=loss_c_meter,
                        c_acc=c_acc,
                    ),
                    flush=True,
                )
                # Prepare the values you want to log
                values = {
                    "Loss/Total": loss_av_meter.val,
                    "Loss/MAE_Audio": loss_a1_meter.val,
                    "Loss/MAE_Midi_Audio": loss_a2_meter.val,
                    "Loss/MAE_Visual": loss_v_meter.val,
                    "Loss/Contrastive": loss_c_meter.val,
                    "Accuracy/Contrastive": c_acc,
                }
                # log metrics to wandb
                run.log(values)

                if np.isnan(loss_av_meter.avg):
                    fabric.print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        fabric.print("start validation")
        (
            eval_loss_av,
            eval_loss_mae,
            eval_loss_mae_a1,
            eval_loss_mae_a2,
            eval_loss_mae_v,
            eval_loss_c,
            eval_c_acc,
        ) = validate(audio_model, test_loader, args, fabric)

        fabric.print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a1))
        fabric.print("Eval MIDI Audio MAE Loss: {:.6f}".format(eval_loss_mae_a2))
        fabric.print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        fabric.print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
        fabric.print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        fabric.print("Eval Total Loss: {:.6f}".format(eval_loss_av))
        fabric.print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))
        val_values = {
                    "Eval Total Loss": eval_loss_av,
                    "Eval Audio MAE Loss": eval_loss_mae_a1,
                    "Eval MIDI Audio MAE Loss": eval_loss_mae_a2,
                    "Eval Visual MAE Loss": eval_loss_mae_v,
                    "Eval Contrastive Loss": eval_loss_c,
                    "Eval Contrastive Accuracy": eval_c_acc,
                }
        run.log(val_values)

        fabric.print("Train Audio MAE Loss: {:.6f}".format(loss_a1_meter.avg))
        fabric.print("Train MIDI Audio MAE Loss: {:.6f}".format(loss_a2_meter.avg))
        fabric.print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        fabric.print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        fabric.print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch - 1, :] = [
            loss_a1_meter.avg,
            loss_a2_meter.avg,
            loss_v_meter.avg,
            loss_c_meter.avg,
            loss_av_meter.avg,
            eval_loss_mae_a1,
            eval_loss_mae_a2,
            eval_loss_mae_v,
            eval_loss_c,
            eval_loss_av,
            eval_c_acc,
            optimizer.param_groups[0]["lr"],
        ]
        np.savetxt(exp_dir + "/result.csv", result, delimiter=",")
        fabric.print("validation finished")

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            fabric.save(
                audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir)
            )
            fabric.save(
                optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir)
            )

        if args.save_model == True:
            fabric.save(
                audio_model.state_dict(),
                "%s/models/audio_model.%d.pth" % (exp_dir, epoch),
            )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        fabric.print("Epoch-{0} lr: {1}".format(epoch, optimizer.param_groups[0]["lr"]))

        _save_progress()

        finish_time = time.time()
        fabric.print(
            "epoch {:d} training time: {:.3f}".format(epoch, finish_time - begin_time)
        )

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a1_meter.reset()
        loss_a2_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()


def validate(audio_model, val_loader, args, fabric):
    device = fabric.device
    batch_time = AverageMeter()
    audio_model = fabric.to_device(audio_model)
    if not isinstance(audio_model, nn.parallel.DistributedDataParallel):
        audio_model = nn.parallel.DistributedDataParallel(audio_model)
    audio_model = fabric.to_device(audio_model)

    audio_model = fabric.setup(audio_model)  # Setup model for validation
    # val_loader = fabric.setup_dataloaders(val_loader)

    audio_model.eval()

    end = time.time()
    (
        A_loss,
        A_loss_mae,
        A_loss_mae_a1,
        A_loss_mae_a2,
        A_loss_mae_v,
        A_loss_c,
        A_c_acc,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with torch.no_grad():
        for i, (a1_input, a2_input, v_input, _) in enumerate(val_loader):
            a1_input = fabric.to_device(a1_input)
            a2_input = fabric.to_device(a2_input)
            v_input = fabric.to_device(v_input)
            
                
            (loss,
            loss_mae,
            loss_mae_a1,
            loss_mae_a2,
            loss_mae_v,
            loss_c,
            mask_a1,
            mask_a2,
            mask_v,
            c_acc,) = audio_model(
            a1_input,
            a2_input,
            v_input,
            args.masking_ratio,
            args.masking_ratio,
            mae_loss_weight=args.mae_loss_weight,
            contrast_loss_weight=args.contrast_loss_weight,
            mask_mode=args.mask_mode,
            )

            # loss, loss_mae, loss_mae_a1, loss_mae_a2, loss_mae_v, loss_c, c_acc = (
            #     loss.sum(),
            #     loss_mae.sum(),
            #     loss_mae_a1.sum(),
            #     loss_mae_a2.sum(),
            #     loss_mae_v.sum(),
            #     loss_c.sum(),
            #     c_acc.mean(),
            # )

            A_loss.append(loss.to("cpu").detach())
            A_loss_mae.append(loss_mae.to("cpu").detach())
            A_loss_mae_a1.append(loss_mae_a1.to("cpu").detach())
            A_loss_mae_a2.append(loss_mae_a2.to("cpu").detach())
            A_loss_mae_v.append(loss_mae_v.to("cpu").detach())
            A_loss_c.append(loss_c.to("cpu").detach())
            A_c_acc.append(c_acc.to("cpu").detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a1 = np.mean(A_loss_mae_a1)
        loss_mae_a2 = np.mean(A_loss_mae_a2)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_mae, loss_mae_a1, loss_mae_a2, loss_mae_v, loss_c, c_acc

