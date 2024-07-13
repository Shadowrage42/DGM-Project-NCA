import os
import pygame
import torch
import numpy as np

from lib.displayer import displayer
from lib.utils import mat_distance
from model.CAModel import CAModel
from utils.visualize import to_rgb, make_seed

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

eraser_radius = 3
pix_size = 8
_map_shape = (64,64)
CHANNEL_N = 16
CELL_FIRE_RATE = 0.5
model_path = "remaster_1.pth_100"
device = torch.device("cpu")

_rows = np.arange(_map_shape[0]).repeat(_map_shape[1]).reshape([_map_shape[0],_map_shape[1]])
_cols = np.arange(_map_shape[1]).reshape([1,-1]).repeat(_map_shape[0],axis=0)

x0 = np.array(torch.rand(1, 64, 64, 16))
x0[:, :, :, :4] = 0
x0[:, :, :, 3] = 1

model = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
output = model(torch.from_numpy(x0.reshape([1,_map_shape[0],_map_shape[1],CHANNEL_N]).astype(np.float32)), 96).squeeze(0)[:, :, :4]

plt.imshow(output.detach().numpy())
plt.show()
