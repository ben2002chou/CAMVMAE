# -*- coding: utf-8 -*-
# @Time    : 3/13/23 2:52 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : inpaint.py

# inpaint with cav-mae model
import os.path
import torch
from models import cav_mae_with_midi as models
import numpy as np
from matplotlib import pyplot as plt
import dataloader_piano_roll as dataloader

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=""):
    # image is [H, W, 3]
    if image.shape[2] == 3:
        plt.imshow(
            torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        )
    else:
        plt.imshow(image, origin="lower")
    plt.title(title, fontsize=12)
    plt.axis("off")
    return


def run_one_image(
    audio,
    midi,
    img,
    model,
    mask_ratio_a=0.75,
    mask_ratio_m=0.75,
    mask_ratio_v=0.75,
    mask_mode="unstructured",
):
    x = img

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    audio = audio.unsqueeze(dim=0)
    audio = torch.einsum("nhwc->nchw", audio)

    audio_input = audio.squeeze(0)

    midi = midi.unsqueeze(dim=0)
    midi = torch.einsum("nhwc->nchw", midi)

    midi_input = midi.squeeze(0)

    (
        y_a,
        y_m,
        y,
        mask_a,
        mask_m,
        mask,
        loss_a,
        loss_m,
        loss_v,
    ) = model.module.forward_inpaint(
        audio_input,
        midi_input,
        x.float(),
        mask_ratio_a1=mask_ratio_a,
        mask_ratio_a2=mask_ratio_m,
        mask_ratio_v=mask_ratio_v,
    )
    y = model.module.unpatchify(y, 3, 14, 14, 16)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    y_a = model.module.unpatchify(y_a, 1, 8, 64, 16)
    y_a = torch.einsum("nchw->nhwc", y_a).detach().cpu()

    y_m = model.module.unpatchify(y_m, 1, 8, 64, 16)
    y_m = torch.einsum("nchw->nhwc", y_m).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.module.patch_embed_v.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.module.unpatchify(mask, 3, 14, 14, 16)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    mask_a = mask_a.detach()
    mask_a = mask_a.unsqueeze(-1).repeat(
        1, 1, model.module.patch_embed_a1.patch_size[0] ** 2 * 1
    )  # (N, H*W, p*p*3)
    mask_a = model.module.unpatchify(
        mask_a, 1, 8, 64, 16
    )  # 1 is removing, 0 is keeping
    mask_a = torch.einsum("nchw->nhwc", mask_a).detach().cpu()

    mask_m = mask_m.detach()
    mask_m = mask_m.unsqueeze(-1).repeat(
        1, 1, model.module.patch_embed_a2.patch_size[0] ** 2 * 1
    )  # (N, H*W, p*p*3)
    mask_m = model.module.unpatchify(
        mask_m, 1, 8, 64, 16
    )  # 1 is removing, 0 is keeping
    mask_m = torch.einsum("nchw->nhwc", mask_m).detach().cpu()

    x = torch.einsum("nchw->nhwc", x)
    audio = torch.einsum("nchw->nwhc", audio)
    midi = torch.einsum("nchw->nwhc", midi)

    im_masked = x * (1 - mask)
    audio_masked = audio * (1 - mask_a)
    midi_masked = midi * (1 - mask_m)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    audio_paste = audio * (1 - mask_a) + y_a * mask_a
    midi_paste = midi * (1 - mask_m) + y_m * mask_m

    # make the plt figure larger
    plt.rcParams["figure.figsize"] = [24, 24]

    plt.subplot(3, 3, 1)
    show_image(x[0], "Original Image")
    #
    plt.subplot(3, 3, 4)
    show_image(audio[0], "Original Spectrogram")
    #
    plt.subplot(3, 3, 7)
    show_image(midi[0], "Original Piano Roll")
    #
    plt.subplot(3, 3, 2)
    show_image(im_masked[0], "Masked Image")
    #
    plt.subplot(3, 3, 5)
    show_image(audio_masked[0], "Masked Spectrogram")
    #
    plt.subplot(3, 3, 8)
    show_image(midi_masked[0], "Masked Piano Roll")
    #
    plt.subplot(3, 3, 3)
    show_image(im_paste[0], "Reconstructed Image")

    plt.subplot(3, 3, 6)
    show_image(audio_paste[0], "Reconstructed Spectrogram")

    plt.subplot(3, 3, 9)
    show_image(midi_paste[0], "Reconstructed Piano Roll")

    return loss_a.item(), loss_m.item(), loss_v.item()


device = "cpu"
mask_ratio_a, mask_ratio_m, mask_ratio_v = 0.75, 0.75, 0.75
# or 'time' or 'freq' or 'tf'
mask_mode = "unstructured"
# the model has to be trained without pixel normalization for inpaint purpose
# cav
model_path = "/home/ben2002chou/code/cav-mae/exp_midi/testmae02-audioset-cav-mae-balNone-lr5e-5-epoch25-bs48-normFalse-c0.01-p0.0-tpFalse-mr-unstructured-0.75/models/best_audio_model.pth"
# # mae
# model_path = "/home/ben2002chou/code/cav-mae/exp_midi/testmae02-audioset-cav-mae-balNone-lr5e-5-epoch25-bs48-normFalse-c0-p1.0-tpFalse-mr-unstructured-0.75/models/best_audio_model.pth"
# # cavmae
# model_path = "/home/ben2002chou/code/cav-mae/exp_midi/testmae02-audioset-cav-mae-balNone-lr5e-5-epoch25-bs48-normFalse-c0.01-p1.0-tpFalse-mr-unstructured-0.75/models/best_audio_model.pth"
A_loss_a, A_loss_m, A_loss_v = [], [], []
if os.path.exists("./sample_reconstruct_midi_c_urmp") == False:
    os.makedirs("./sample_reconstruct_midi_c_urmp")

mae_mdl = models.CAVMAE(modality_specific_depth=11)

pretrained_weights = torch.load(model_path, map_location=device)
mae_mdl = torch.nn.DataParallel(mae_mdl)
msg = mae_mdl.load_state_dict(pretrained_weights, strict=False)
print("Model Loaded.", msg)
mae_mdl = mae_mdl.to(device)
mae_mdl.eval()

val_audio_conf = {
    "num_mel_bins": 128,
    "target_length": 1024,
    "freqm": 0,
    "timem": 0,
    "mixup": 0,
    "dataset": "vggsound",
    "mode": "eval",
    "mean": -5.081,
    "std": 4.4849,
    "noise": False,
    "im_res": 224,
    "frame_use": 5,
}

# on vggsound, while the model is pretrained on AS-2M, so it is zero-shot.
val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(
        "/home/ben2002chou/code/cav-mae/data/cocochorals/urmp.json",
        label_csv="/home/ben2002chou/code/cav-mae/data/cocochorals/class_labels_indices_combined.csv",
        audio_conf=val_audio_conf,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

for i, (a_input, m_input, v_input, _) in enumerate(val_loader):
    a_input = a_input.to(device)
    m_input = m_input.to(device)
    v_input = v_input.to(device)
    v_input = v_input[0].permute(1, 2, 0)
    a_input = a_input.permute(1, 2, 0)
    m_input = m_input.permute(1, 2, 0)
    torch.manual_seed(2)
    fig = plt.figure()
    loss_a, loss_m, loss_v = run_one_image(
        a_input,
        m_input,
        v_input,
        mae_mdl,
        mask_ratio_a,
        mask_ratio_m,
        mask_ratio_v,
        mask_mode=mask_mode,
    )
    ###
    A_loss_a.append(loss_a)
    A_loss_m.append(loss_m)
    A_loss_v.append(loss_v)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.savefig(
        "./sample_reconstruct_midi_c_urmp/{:d}_{:.4f}_{:.4f}_{:.4f}.png".format(
            i, mask_ratio_a, mask_ratio_m, mask_ratio_v
        ),
        dpi=150,
    )
    plt.close()
    # show 4 samples, change if you want to see more
    if i >= 3:
        break
print(
    "loss a is {:.4f}, loss m is {:.4f}, loss v is {:.4f}".format(
        np.mean(A_loss_a), np.mean(A_loss_m), np.mean(A_loss_v)
    )
)
