"""EECN (ACM MM 2025) core network modules.

This file implements the main architectural blocks described in the paper:

 - Mask-aware Fourier prior + learnable mask proportion predictor (Eq. 5)
 - Partial Spatial-Frequency Interaction (PSFI) block (Eq. 6)
 - Dual Frequency Cross-attention Module (DFCM) (Eq. 7-8)
 - Helper for timestep-guided frequency conditioning used in TFDM (Sec. 3.4)

The diffusion process (q/p, training loop, sampling) is intentionally kept
outside this file, because different codebases wire DDPM/DDIM differently.
You can directly reuse your existing diffusion wrapper and only replace its
condition with `time_aware_condition()`.

All ops are pure PyTorch and support autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# FFT helpers (per-channel)
# -------------------------


def _fftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D fftshift over the last two dims."""
    return torch.fft.fftshift(x, dim=(-2, -1))


def _ifftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D ifftshift over the last two dims."""
    return torch.fft.ifftshift(x, dim=(-2, -1))


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered FFT2 (complex output)."""
    x = x.to(torch.float32)
    X = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    return _fftshift2(X)


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    """Centered iFFT2 (real output)."""
    X = _ifftshift2(X)
    x = torch.fft.ifft2(X, dim=(-2, -1), norm="ortho")
    # keep real component (inputs are real-valued feature maps)
    return x.real


def amp_phase(X: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return amplitude and phase of a complex tensor."""
    A = torch.abs(X)
    P = torch.atan2(X.imag, X.real + eps)
    return A, P


def from_amp_phase(A: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Reconstruct complex spectrum from amplitude and phase."""
    return torch.polar(A, P)


# -------------------------
# Mask generation
# -------------------------


def build_radial_mask(
    gamma: torch.Tensor,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a circular low-frequency mask with area ratio controlled by gamma.

    gamma ∈ [0, 1]. Larger gamma keeps more frequencies (larger radius).
    Output shape: [B, 1, H, W] with values in {0,1}.
    """
    # Create normalized radius map in [0, 1]
    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(X * X + Y * Y)  # [H, W], max ~= sqrt(2)
    r = r / (r.max() + 1e-12)

    # radius threshold per batch
    thr = gamma.clamp(0.0, 1.0).view(-1, 1, 1, 1)
    mask = (r.view(1, 1, h, w) <= thr).to(dtype)
    return mask


# -------------------------
# Core blocks
# -------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: bool = True):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=pad)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class MaskPredictor(nn.Module):
    """Predict monotonic mask proportions (Eq. 5).

    We output N gammas in ascending order in [0, 1].
    """

    def __init__(self, in_ch: int = 3, n_levels: int = 3, hidden: int = 64):
        super().__init__()
        self.n_levels = n_levels
        self.c1 = ConvBlock(in_ch, hidden, k=3, act=True)
        self.c2 = ConvBlock(hidden, hidden, k=3, act=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, n_levels),
            nn.Softplus(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Return gammas: [B, N] in ascending order."""
        b = img.shape[0]
        h = self.c2(self.c1(img))
        h = self.pool(h).view(b, -1)
        d = self.mlp(h)  # [B, N], positive
        delta = d / (d.sum(dim=1, keepdim=True) + 1e-12)
        gamma = torch.cumsum(delta, dim=1).clamp(0.0, 1.0)
        return gamma


class PSFI(nn.Module):
    """Partial Spatial-Frequency Interaction (Eq. 6)."""

    def __init__(self, ch: int):
        super().__init__()
        # spatial branch: Conv3 -> LReLU -> Conv3 -> LReLU
        self.s1 = ConvBlock(ch, ch, k=3, act=True)
        self.s2 = ConvBlock(ch, ch, k=3, act=True)

        # frequency branch: Conv1 -> LReLU (for amp & phase respectively)
        self.fa = ConvBlock(ch, ch, k=1, act=True)
        self.fp = ConvBlock(ch, ch, k=1, act=True)

        # fusion
        self.fuse = nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1)

    def forward(self, r_i: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:
        r_i: [B, C, H, W]
        mask: [B, 1, H, W] (0/1)
        """
        # spatial branch
        r_s = self.s2(self.s1(r_i))

        # frequency branch
        X = fft2c(r_i)
        A, P = amp_phase(X)
        A = A * mask
        P = P * (1.0 - mask)
        A = self.fa(A)
        P = self.fp(P)
        X_new = from_amp_phase(A, P)
        r_f = ifft2c(X_new)

        r_o = self.fuse(torch.cat([r_f, r_s], dim=1)) + r_i
        return r_o


class ChannelWiseCrossAttention(nn.Module):
    """Channel-wise cross attention (CCA) used in DFCM (Eq. 8).

    We compute attention in the channel dimension:
      Q from input, K/V from reference.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.q = nn.Conv2d(ch, ch, kernel_size=1)
        self.k = nn.Conv2d(ch, ch, kernel_size=1)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x_in: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_in.shape
        q = self.q(x_in).view(b, c, h * w)
        k = self.k(x_ref).view(b, c, h * w)
        v = self.v(x_ref).view(b, c, h * w)

        # channel attention: [B, C, C]
        attn = torch.bmm(q, k.transpose(1, 2)) / (h * w) ** 0.5
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).view(b, c, h, w)
        return self.proj(out)


class DFCM(nn.Module):
    """Dual Frequency Cross-attention Module (Eq. 7-8)."""

    def __init__(self, ch: int):
        super().__init__()
        # shared convs for amplitude and phase learning (Eq. 7)
        self.a_conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=1),
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=1),
        )
        self.cca_a = ChannelWiseCrossAttention(ch)
        self.cca_p = ChannelWiseCrossAttention(ch)

    def forward(self, r_input: torch.Tensor, r_ref: torch.Tensor) -> torch.Tensor:
        X_in = fft2c(r_input)
        X_ref = fft2c(r_ref)
        A_in, P_in = amp_phase(X_in)
        A_ref, P_ref = amp_phase(X_ref)

        A_in_hat = self.a_conv(A_in)
        A_ref_hat = self.a_conv(A_ref)
        P_in_hat = self.p_conv(P_in)
        P_ref_hat = self.p_conv(P_ref)

        A_star = self.cca_a(A_in_hat, A_ref_hat)
        P_star = self.cca_p(P_in_hat, P_ref_hat)

        X_star = from_amp_phase(A_star, P_star)
        r_out = ifft2c(X_star)
        return r_out


# -------------------------
# Encoder / Decoder
# -------------------------


@dataclass
class EECNConfig:
    in_ch: int = 3
    base_ch: int = 64
    n_psfi: int = 3  # paper uses 3 PSFI layers


class EECNEncoder(nn.Module):
    """Shared encoder E(·): Conv + (PSFI + Down)*"""

    def __init__(self, cfg: EECNConfig):
        super().__init__()
        self.cfg = cfg
        self.in_conv = ConvBlock(cfg.in_ch, cfg.base_ch, k=3, act=True)
        self.psfi = nn.ModuleList([PSFI(cfg.base_ch) for _ in range(cfg.n_psfi)])
        self.down = nn.ModuleList([
            nn.Conv2d(cfg.base_ch, cfg.base_ch, kernel_size=3, stride=2, padding=1)
            for _ in range(cfg.n_psfi - 1)
        ])

    def forward(self, x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        """Return the deepest feature map."""
        h = self.in_conv(x)
        for i in range(self.cfg.n_psfi):
            h = self.psfi[i](h, masks[i])
            if i < self.cfg.n_psfi - 1:
                h = self.down[i](h)
        return h


class SimpleDecoder(nn.Module):
    """A lightweight decoder D(·) to map features back to RGB."""

    def __init__(self, ch: int = 64, out_ch: int = 3, n_up: int = 2):
        super().__init__()
        self.n_up = n_up
        self.body = nn.Sequential(
            ConvBlock(ch, ch, k=3, act=True),
            ConvBlock(ch, ch, k=3, act=True),
        )
        self.out_conv = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.n_up):
            h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        h = self.body(h)
        return self.out_conv(h)


# -------------------------
# TFDM helper
# -------------------------


def time_aware_condition(x_tilde: torch.Tensor, t: torch.Tensor, T: int) -> torch.Tensor:
    """Build the time-aware condition \tilde{x}_t in Sec. 3.4.

    Paper: generate a mask M with ratio gamma=t/T and apply to \tilde{x}
    to obtain \tilde{x}_t.

    Here we implement a mask-aware Fourier filtering on \tilde{x}:
      - keep low-frequency amplitude inside mask
      - keep high-frequency phase outside mask

    Args:
        x_tilde: [B, C, H, W] coarse features / image used as condition
        t: [B] int timesteps
        T: total timesteps
    """
    assert t.dim() == 1, "t must be [B]"
    gamma = (t.float() / float(T)).clamp(0.0, 1.0)
    b, _, h, w = x_tilde.shape
    mask = build_radial_mask(gamma, h, w, device=x_tilde.device, dtype=x_tilde.dtype)  # [B,1,H,W]

    X = fft2c(x_tilde)
    A, P = amp_phase(X)
    A = A * mask
    P = P * (1.0 - mask)
    return ifft2c(from_amp_phase(A, P))


# -------------------------
# Full coarse network (stage-1)
# -------------------------


class EECNStage1(nn.Module):
    """Stage-1 of EECN: PSFI encoder + DFCM + decoder.

    It takes an ill-exposed image and a well-exposed reference image
    (unpaired) and outputs a coarse restoration and the conditioning
    feature for TFDM.
    """

    def __init__(self, cfg: EECNConfig = EECNConfig()):
        super().__init__()
        self.cfg = cfg
        self.mask_pred = MaskPredictor(in_ch=cfg.in_ch, n_levels=cfg.n_psfi, hidden=cfg.base_ch)
        self.encoder = EECNEncoder(cfg)
        self.dfcm = DFCM(cfg.base_ch)
        # if n_psfi=3, we downsample twice -> upsample twice
        self.decoder = SimpleDecoder(ch=cfg.base_ch, out_ch=cfg.in_ch, n_up=cfg.n_psfi - 1)

    def forward(self, x_in: torch.Tensor, x_ref: torch.Tensor):
        """Returns:
        - x_coarse: coarse RGB prediction
        - cond_feat: feature used as condition (R*_o)
        - gammas: [B, N]
        """
        gammas = self.mask_pred(x_in)  # [B,N]
        # build masks at each encoder resolution
        masks: List[torch.Tensor] = []
        h, w = x_in.shape[-2:]
        for i in range(self.cfg.n_psfi):
            hi = h // (2 ** i)
            wi = w // (2 ** i)
            masks.append(build_radial_mask(gammas[:, i], hi, wi, device=x_in.device, dtype=x_in.dtype))

        r_in = self.encoder(x_in, masks)
        r_ref = self.encoder(x_ref, masks)
        r_star = self.dfcm(r_in, r_ref)
        x_coarse = self.decoder(r_star)
        return {"x_coarse": x_coarse, "cond_feat": r_star, "gammas": gammas, "z_feat": r_star, "feat_in": r_in, "feat_ref": r_ref}
