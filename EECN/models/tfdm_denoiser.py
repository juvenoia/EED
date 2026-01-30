"""
TFDM denoiser UNet wrapper.

This wrapper keeps naming consistent with the EECN paper (TFDM),
while reusing the existing UNet implementation (originally named DiffusionUNet).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet import DiffusionUNet as _LegacyDiffusionUNet


class TFDMDenoiserUNet(_LegacyDiffusionUNet):
    """
    Alias of the existing UNet, but renamed to match the paper terminology.

    Forward signature stays the same:
        y = model(x_cat, t_float)
    where x_cat = concat(condition, noisy_target) in channel dimension.
    """
    pass
