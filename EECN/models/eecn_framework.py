"""
EECN overall framework (Stage-1 + TFDM diffusion refinement).

Goal: keep naming consistent with the paper and avoid legacy/baseline names.

This module assumes the existing codebase provides:
- `utils.data_transform` / `utils.inverse_data_transform`
- a config object with `config.diffusion.*` and `config.model.*`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import utils  # existing project dependency

from .eecn_stage1 import EECNStage1
from .tfdm_denoiser import TFDMDenoiserUNet
from .eecn_modules import time_aware_condition


def get_beta_schedule(beta_schedule: str, *, beta_start: float, beta_end: float, num_diffusion_timesteps: int):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class EECN(nn.Module):
    """
    Training-time input convention (kept compatible with your current pipeline):
        inputs: (B,6,H,W) = concat([x_in, x_gt], dim=1)

    Inference-time:
        inputs: (B,3,H,W) = x_in only

    Outputs (training):
        dict with:
            - noise_pred, noise_gt
            - pred_x (final refined prediction in image space)
            - target_x (GT image)
            - coarse_x (Stage-1 coarse output)
            - gammas (mask ratio sequence)
    """
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        # TFDM denoiser
        self.tfdm = TFDMDenoiserUNet(config)

        # Stage-1
        self.stage1 = EECNStage1()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = int(self.betas.shape[0])

    @staticmethod
    def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def _ddim_sample(self, cond_img_norm: torch.Tensor, b: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling (used to produce pred_x during training for SCC-like regularization or logging).
        """
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = list(range(0, self.config.diffusion.num_diffusion_timesteps, skip))
        n, c, h, w = cond_img_norm.shape
        seq_next = [-1] + seq[:-1]

        x = torch.randn(n, c, h, w, device=self.device)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            next_t = torch.full((n,), j, device=x.device, dtype=torch.long)

            at = self.compute_alpha(b, t)
            at_next = self.compute_alpha(b, next_t)

            # time-aware conditioning (paper Sec.3.4)
            cond_t = time_aware_condition(cond_img_norm, t=t, T=self.num_timesteps)

            et = self.tfdm(torch.cat([cond_t, x], dim=1), t.float())
            x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            x = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

        return x

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = self.betas.to(inputs.device)

        if self.training:
            assert inputs.shape[1] == 6, "training expects (B,6,H,W) = [x_in, x_gt]"
            x_in = inputs[:, :3]
            x_gt = inputs[:, 3:]

            # Stage-1 (use GT as ref during training if you want; can also pass None)
            s1 = self.stage1(x_in, x_ref=x_gt)
            coarse_x = s1["x_coarse"]
            gammas = s1["gammas"]

            # condition image for TFDM
            cond_img = s1["cond_img"]
            cond_img_norm = utils.data_transform(cond_img)

            # diffusion target
            target_norm = utils.data_transform(x_gt)

            # sample timesteps
            t = torch.randint(low=0, high=self.num_timesteps, size=(target_norm.shape[0] // 2 + 1,),
                              device=inputs.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:target_norm.shape[0]]

            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            noise = torch.randn_like(target_norm)
            x_noisy = target_norm * a.sqrt() + noise * (1.0 - a).sqrt()

            # time-aware condition (paper Sec.3.4)
            cond_t = time_aware_condition(cond_img_norm, t=t.long(), T=self.num_timesteps)

            noise_pred = self.tfdm(torch.cat([cond_t, x_noisy], dim=1), t.float())

            # optional: get a denoised image via DDIM sampling (for logging / feature constraints)
            pred_norm = self._ddim_sample(cond_img_norm, b)
            pred_x = utils.inverse_data_transform(pred_norm)

            return {
                "noise_pred": noise_pred,
                "noise_gt": noise,
                "pred_x": pred_x,
                "target_x": x_gt,
                "coarse_x": coarse_x,
                "gammas": gammas,
            }

        # inference
        assert inputs.shape[1] == 3, "inference expects (B,3,H,W) = x_in"
        x_in = inputs
        s1 = self.stage1(x_in, x_ref=None)
        cond_img = s1["cond_img"]
        cond_img_norm = utils.data_transform(cond_img)
        pred_norm = self._ddim_sample(cond_img_norm, b)
        pred_x = utils.inverse_data_transform(pred_norm)
        return {"pred_x": pred_x, "coarse_x": s1["x_coarse"], "gammas": s1["gammas"]}
