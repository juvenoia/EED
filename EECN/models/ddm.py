
import os
import time
import math
from collections import OrderedDict
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
from .eecn_stage1 import EECNStage1
from .tfdm_denoiser import TFDMDenoiserUNet
from .eecn_modules import amp_phase, time_aware_condition


# -------------------------
# EMA helper (unchanged)
# -------------------------
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
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


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict

class EECNConfig:
    in_ch: int = 3
    base_ch: int = 64
    n_psfi: int = 3  # paper uses 3 PSFI layers

# -------------------------
# EECN Framework (Stage1 + TFDM)
# -------------------------
class EECNModel(nn.Module):
    """
    Strictly follows the paper structure:
      Stage-1: PSFI encoder E + DFCM + decoder D (coarse restoration)
      Stage-2: TFDM diffusion (U-Net noise predictor) with time-aware masked condition
    """
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        # Stage-1
        stage1_cfg = EECNConfig()
        stage1_cfg.in_ch = config.data.channels
        stage1_cfg.base_ch = config.eecn.stage1.base_channels
        stage1_cfg.n_psfi = config.eecn.stage1.psfi_layers

        self.stage1 = EECNStage1(stage1_cfg)

        # Stage-2 denoiser (TFDM)
        self.denoiser = TFDMDenoiserUNet(config)

        # diffusion schedule
        betas = get_beta_schedule(
            config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.num_timesteps = self.betas.shape[0]

    # ---- diffusion helpers ----
    def compute_alpha(self, t: torch.Tensor) -> torch.Tensor:
        b = self.betas
        # cumprod(1-beta)
        a = (1 - b).cumprod(dim=0)
        return a.index_select(0, t).view(-1, 1, 1, 1)

    @torch.no_grad()
    def ddim_sample(self, cond: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling (fast) to obtain x0 prediction for L_con or inference.
        cond is in *normalized* space (utils.data_transform output).
        """
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = list(range(0, self.config.diffusion.num_diffusion_timesteps, skip))
        seq_next = [-1] + list(seq[:-1])

        n, c, h, w = cond.shape
        x = torch.randn(n, c, h, w, device=cond.device)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n, device=cond.device) * i).long()
            next_t = (torch.ones(n, device=cond.device) * j).long()

            at = self.compute_alpha(t)
            at_next = self.compute_alpha(next_t.clamp(min=0))

            # time-aware condition (paper Sec. 3.4)
            cond_t = time_aware_condition(cond, t.float(), float(self.num_timesteps))

            et = self.denoiser(torch.cat([cond_t, x], dim=1), t.float())
            x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()

            if j < 0:
                x = x0_t
                continue

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            x = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

        return x

    # ---- forward paths ----
    def forward_stage1(self, x_in: torch.Tensor, x_ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.stage1(x_in, x_ref)

    def forward_stage2_training(self, x_in: torch.Tensor, x_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns components for Eq.(13):
          L_df: noise prediction MSE
          L_con: L1(x0_pred, x_gt) where x0_pred is reverse output conditioned on x~t
        """
        # stage-1 provides coarse condition
        with torch.no_grad():
            s1 = self.stage1(x_in, x_gt)
            x_coarse = s1["cond_img"]  # (B,3,H,W) in image space

        cond = utils.data_transform(x_coarse)  # normalize to [-1,1]
        gt = utils.data_transform(x_gt)

        # sample timestep
        bsz = cond.shape[0]
        t = torch.randint(low=0, high=self.num_timesteps, size=(bsz // 2 + 1,), device=cond.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:bsz]

        a = self.compute_alpha(t)
        e = torch.randn_like(gt)
        x_t = gt * a.sqrt() + e * (1.0 - a).sqrt()

        cond_t = time_aware_condition(cond, t.float(), float(self.num_timesteps))
        noise_pred = self.denoiser(torch.cat([cond_t, x_t], dim=1), t.float())

        # for L_con, run fast DDIM once (few steps)
        x0_pred = self.ddim_sample(cond, eta=0.0)
        x0_pred = utils.inverse_data_transform(x0_pred)

        return {
            "noise_pred": noise_pred,
            "noise_gt": e,
            "x0_pred": x0_pred,
            "x_gt": x_gt,
        }

    @torch.no_grad()
    def infer(self, x_in: torch.Tensor, x_ref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Inference: run stage-1 coarse, then TFDM sampling to refine. If x_ref is None,
        fall back to using x_in as reference (works but not ideal).
        """
        if x_ref is None:
            x_ref = x_in
        s1 = self.stage1(x_in, x_ref)
        x_coarse = s1["cond_img"]
        cond = utils.data_transform(x_coarse)
        x0 = self.ddim_sample(cond, eta=0.0)
        x0 = utils.inverse_data_transform(x0)
        return {"x_coarse": x_coarse, "x_refined": x0}


# -------------------------
# Trainer wrapper (kept name for compatibility with train.py)
# -------------------------
class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        # stage selection
        self.stage = getattr(args, "stage", None) or getattr(config.training, "stage", "stage2")
        assert self.stage in ["stage1", "stage2"], "stage must be 'stage1' or 'stage2'"

        self.model = EECNModel(args, config).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2 = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        # loss weights per paper (Sec 4.1)
        self.lambda1 = getattr(config.training, "lambda1", 1.0)
        self.lambda2 = getattr(config.training, "lambda2", 0.1)
        self.lambda3 = getattr(config.training, "lambda3", 0.1)

    def load_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.step = checkpoint.get("step", 0)
        if ema and "ema_helper" in checkpoint:
            self.ema_helper.load_state_dict(checkpoint["ema_helper"])
            self.ema_helper.ema(self.model)
        print(f"=> loaded checkpoint {load_path} step {self.step}")

    def load_ddm_ckpt(self, load_path, ema=False):
        # backward-compatible alias
        return self.load_ckpt(load_path, ema=ema)

    def _set_trainable(self):
        # stage1: only optimize stage1
        # stage2: freeze stage1, optimize denoiser
        for name, p in self.model.named_parameters():
            if self.stage == "stage1":
                p.requires_grad = ("stage1" in name)
            else:
                p.requires_grad = ("denoiser" in name)

        # rebuild optimizer with correct params
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = utils.optimize.get_optimizer(self.config, params)

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if getattr(self.args, "resume", "") and os.path.isfile(self.args.resume):
            self.load_ckpt(self.args.resume, ema=False)

        self._set_trainable()
        self.model.train()

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x_in = x[:, :3, :, :]
                x_gt = x[:, 3:, :, :]

                if self.stage == "stage1":
                    out = self.model.module.forward_stage1(x_in, x_gt)
                    x_coarse = out["x_coarse"]
                    # L_rec Eq.(11): ||Igt - D(E(Iin))||2 + ||Igt - D(E(Igt))||2
                    # Our stage1 already uses DFCM for coarse, but we keep a faithful variant:
                    # - coarse from Iin with ref Igt
                    # - self-recon from Igt with ref Igt
                    out_self = self.model.module.forward_stage1(x_gt, x_gt)
                    l_rec = self.l2(x_coarse, x_gt) + self.l2(out_self["x_coarse"], x_gt)

                    # L_f Eq.(12): ||A(Z)-A(E(Igt))||2 + ||P(Z)-P(E(Iin))||2
                    z = out["z_feat"]
                    e_gt = out["feat_ref"]
                    e_in = out["feat_in"]
                    a_z, p_z = amp_phase(z)
                    a_gt, _ = amp_phase(e_gt)
                    _, p_in = amp_phase(e_in)
                    l_f = self.l2(a_z, a_gt) + self.l2(p_z, p_in)

                    loss = self.lambda1 * l_rec + self.lambda2 * l_f

                    log_items = {"l_rec": l_rec.item(), "l_f": l_f.item()}
                else:
                    out = self.model.module.forward_stage2_training(x_in, x_gt)
                    l_df = self.l2(out["noise_pred"], out["noise_gt"])
                    l_con = self.l1(out["x0_pred"], out["x_gt"])
                    loss = l_df + self.lambda3 * l_con
                    log_items = {"l_df": l_df.item(), "l_con": l_con.item()}

                data_time += time.time() - data_start

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                self.step += 1
                data_start = time.time()

                if self.step % 10 == 0:
                    msg = f"stage:{self.stage} step:{self.step} loss:{loss.item():.5f} time:{data_time/(i+1):.4f}"
                    for k,v in log_items.items():
                        msg += f" {k}:{v:.5f}"
                    print(msg)

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    print("evaluation.")
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)
                    utils.logging.save_checkpoint(
                        {'step': self.step,
                         'epoch': epoch + 1,
                         'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'ema_helper': self.ema_helper.state_dict(),
                         'params': self.args,
                         'config': self.config},
                        filename=os.path.join(self.config.data.ckpt_dir, f'model_epoch_{epoch+1}')
                    )
                    self.model.train()

    @torch.no_grad()
    def sample_validation_patches(self, val_loader, step: int):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        os.makedirs(os.path.join(image_folder, str(step)), exist_ok=True)
        self.model.eval()

        print(f'Performing validation at step: {step}')
        psnr_list = []

        for i, (x, y) in enumerate(val_loader):
            b, _, img_h, img_w = x.shape

            img_h_64 = int(64 * np.ceil(img_h / 64.0))
            img_w_64 = int(64 * np.ceil(img_w / 64.0))
            x_pad = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect').to(self.device)

            x_in = x_pad[:, :3]
            x_gt = x_pad[:, 3:]

            # use GT as reference during validation (paired setting)
            pred = self.model.module.infer(x_in, x_gt)["x_refined"][:, :, :img_h, :img_w]
            gt = x_gt[:, :, :img_h, :img_w].to(pred.device)

            utils.logging.save_image(pred, os.path.join(image_folder, str(step), f'predict_{y[0]}'))
            utils.logging.save_image(gt, os.path.join(image_folder, str(step), f'gt_{y[0]}'))

            psnr_list.append(calculate_psnr(tensor2img(pred.cpu()), tensor2img(gt.cpu())))

        mean_psnr = sum(psnr_list) / max(len(psnr_list), 1)
        print(f"Validation: mean_psnr: {mean_psnr:.5f}")


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def tensor2img(tensor):
    img_np = tensor.numpy()[0]
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    a, b = np.min(img_np), np.max(img_np)
    img_np = (img_np - a) / (b - a + 1e-12)
    return (img_np * 255.0).round().astype(np.uint8)
