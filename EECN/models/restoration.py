
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import utils


class DiffusiveRestoration:
    """
    Evaluation / inference helper.
    Expects validation loader to provide:
      x: (B,6,H,W) where x[:,:3]=input and x[:,3:]=GT (paired val)
    """
    def __init__(self, diffusion, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained model path is missing!')

    @torch.no_grad()
    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        os.makedirs(image_folder, exist_ok=True)

        for i, (x, y) in enumerate(val_loader):
            x_in = x[:, :3, :, :].to(self.diffusion.device)
            x_gt = x[:, 3:, :, :].to(self.diffusion.device)

            b, c, h, w = x_in.shape
            img_h_64 = int(64 * np.ceil(h / 64.0))
            img_w_64 = int(64 * np.ceil(w / 64.0))
            x_in_pad = F.pad(x_in, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')
            x_gt_pad = F.pad(x_gt, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')

            t1 = time.time()
            out = self.diffusion.model.module.infer(x_in_pad, x_gt_pad)
            pred = out["x_refined"][:, :, :h, :w]
            t2 = time.time()

            utils.logging.save_image(pred, os.path.join(image_folder, f"predict_{y[0]}"))
            utils.logging.save_image(x_gt, os.path.join(image_folder, f"gt_{y[0]}"))

            print(f"{i}: {y[0]}  time:{t2 - t1:.3f}s")
