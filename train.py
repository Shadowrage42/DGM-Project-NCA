import os
import time
from data.data import load_emoji
import imageio

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython.display import clear_output

from model.CAModel import CAModel
from lib.utils_vis import SamplePool, get_living_mask, make_seed, make_circle_masks
from utils.visualize import plot_loss, visualize_batch, to_alpha, to_rgb
from medmnist import DermaMNIST
from medmnist.info import INFO

from torch.utils.data import DataLoader

USE_WANDB = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "model/checkpoints/remaster_1.pth"

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40

lr = 2e-3
lr_gamma = 0.9999
betas = (0.5, 0.5)
n_epoch = 80000

BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

TARGET_EMOJI = 0 #@param "🦎"

EXPERIMENT_TYPE = "Persistent"
EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DERMAMNIST_CLASSES = INFO["dermamnist"]["label"]

def load_dermaMNIST(split, download, as_rgb, size):
    train_mnist_dataset = DermaMNIST(split="train", download=True, as_rgb=True, size=28)
    return train_mnist_dataset
dermaMnist_dataset = load_dermaMNIST(split="train", download=True, as_rgb=True, size=28)


melanoma_samples = []
for i, sample in enumerate(dermaMnist_dataset):
    if sample[1][0] == 4:
        img_rgba = sample[0].convert("RGBA")
        melanoma_samples.append(np.array(img_rgba, dtype=np.float32))
print(f"dataset length {len(melanoma_samples)}")

target_img = melanoma_samples

# plt.figure(figsize=(4,4))
# plt.imshow(target_img)
# plt.show()


p = TARGET_PADDING
pad_target = np.pad(target_img, [(0,0), (p, p), (p, p), (0, 0)])
h, w = pad_target.shape[1:3]
# pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

train_dataloader = DataLoader(pad_target, batch_size=BATCH_SIZE, shuffle=True)

seed = make_seed((h, w), CHANNEL_N)
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))
batch = pool.sample(BATCH_SIZE).x

ca = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
ca.load_state_dict(torch.load(model_path, map_location=device))

optimizer = optim.Adam(ca.parameters(), lr=lr, betas=betas)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

if USE_WANDB:
    import wandb
    import secret

    os.environ["WANDB_API_KEY"] = secret.key
    wandb.init(project="GrowingNCA")
    wandb.watch(ca, log='gradients', log_freq=BATCH_SIZE)
    wandb.watch(ca, log='parameters', log_freq=BATCH_SIZE)

loss_log = []

def train(x, target, steps, optimizer, scheduler):
    x = ca(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :4], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

def loss_f(x, target):
    return torch.mean(torch.pow(x[..., :4]-target, 2), [-2,-3,-1])

for i in range(n_epoch+1):
    if USE_PATTERN_POOL:
        batch = pool.sample(BATCH_SIZE)
        x0 = torch.from_numpy(batch.x.astype(np.float32)).to(device)
        loss_rank = loss_f(x0, pad_target).detach().cpu().numpy().argsort()[::-1]
        x0 = batch.x[loss_rank]
        x0[:1] = seed

    else:
        x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    x0 = torch.from_numpy(x0.astype(np.float32)).to(device)

    x, loss = train(x0, pad_target, np.random.randint(64,96), optimizer, scheduler)

    if USE_WANDB:
        wandb.log({'model_loss': loss.item()})
    
    if USE_PATTERN_POOL:
        batch.x[:] = x.detach().cpu().numpy()
        batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.item())
    
    if step_i%100 == 0:
        # clear_output()
        print(step_i, "loss =", loss.item())
        # visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy())
        # plot_loss(loss_log)
        torch.save(ca.state_dict(), model_path)

if USE_WANDB:
    wandb.finish()
    