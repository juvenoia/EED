"""
EECN Stage-1 (coarse restoration + frequency prior) implementation wrapper.

This file intentionally avoids legacy/baseline naming (e.g., "decom", "CTDN").
It exposes a single Stage-1 network consistent with the EECN paper:
- MaskPredictor (Eq.5)
- PSFI (Eq.6)
- DFCM (Eq.7-8)

Actual implementation lives in `eecn.py` to keep components reusable.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .eecn_modules import EECNStage1 as _EECNStage1


class EECNStage1(nn.Module):
    """
    Stage-1 network used by the overall EECN framework.

    Inputs:
        x_in:  (B,3,H,W) improperly-exposed input
        x_ref: (B,3,H,W) optional reference (during training you can pass GT)

    Outputs:
        dict with:
            - x_coarse: (B,3,H,W) coarse restored image
            - cond_img: (B,3,H,W) condition image for TFDM (we use x_coarse)
            - gammas:   (B,N)     mask ratio sequence
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.core = _EECNStage1(*args, **kwargs)

    def forward(self, x_in: torch.Tensor, x_ref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.core(x_in, x_ref)
        # For diffusion conditioning, the most stable drop-in is the coarse image itself.
        return {
            "x_coarse": out["x_coarse"],
            "cond_img": out["x_coarse"],
            "gammas": out["gammas"],
            # internal features for Eq.(12)
            "z_feat": out.get("z_feat"),
            "feat_in": out.get("feat_in"),
            "feat_ref": out.get("feat_ref"),
        }
