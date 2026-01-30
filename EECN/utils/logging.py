
import os
import shutil
from typing import Optional

import torch
import numpy as np
from PIL import Image


def _tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """
    Accepts:
      - (B,C,H,W) or (C,H,W) tensor in [0,1] or [-1,1]
    Returns HWC uint8 in RGB.
    """
    if img.dim() == 4:
        img = img[0]
    img = img.detach().cpu().float()

    # if in [-1,1], map to [0,1]
    if img.min() < 0:
        img = (img + 1.0) / 2.0

    img = img.clamp(0, 1)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    arr = (img.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return arr


def save_image(img: torch.Tensor, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    arr = _tensor_to_uint8(img)
    Image.fromarray(arr).save(file_path)


def save_checkpoint(state, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path: str, device: Optional[str]):
    if device is None:
        return torch.load(path, map_location='cpu')
    return torch.load(path, map_location=device)


def copytree(src, dst, symlinks=False, ignore=None):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks, ignore)
