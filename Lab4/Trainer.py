import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import (
    Generator,
    Gaussian_Predictor,
    Decoder_Fusion,
    Label_Encoder,
    RGB_Encoder,
    Learned_Prior_Predictor,
)

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack
from torch import Tensor
from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log10


def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor >1 Batch is available"""
    # Flatten per image: (B, C, H, W) → (B, -1)
    imgs1 = imgs1.view(imgs1.size(0), -1)
    imgs2 = imgs2.view(imgs2.size(0), -1)

    mse = torch.mean((imgs1 - imgs2) ** 2, dim=1)  # Per image MSE
    psnr = 20 * torch.log10(
        torch.tensor(data_range, device=mse.device)
    ) - 10 * torch.log10(mse)

    return psnr.mean()  # Return average PSNR over batch


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD

def kl_criterion_lp(posterior_mu, posterior_logvar, prior_mu, prior_logvar, batch_size):
    """
    Calculates the KL divergence between two diagonal Gaussians: D_KL(q || p).
    q is the posterior distribution.
    p is the prior distribution.
    """
    # 獲取變異數
    posterior_var = posterior_logvar.exp()
    prior_var = prior_logvar.exp()

    # 計算 KL 散度
    # 公式: 0.5 * sum( (var_q + (mu_q - mu_p)^2) / var_p - 1 + log(var_p) - log(var_q) )
    kl_div = 0.5 * torch.sum(
        prior_logvar - posterior_logvar +
        (posterior_var + (posterior_mu - prior_mu).pow(2)) / prior_var -
        1
    )
    
    # 返回每個樣本的平均 KL 散度
    return kl_div / batch_size

class kl_annealing:
    def __init__(self, args, current_epoch=0):

        self.annealing_type = args.kl_anneal_type

        assert self.annealing_type in [
            "Cyclical",
            "Monotonic",
            "Without",
            "Cosine",
            "Constant",
        ]

        self.iter = current_epoch + 1

        if self.annealing_type == "Cyclical":
            self.L = self.frange_cycle_linear(
                n_iter=args.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=args.kl_anneal_cycle,
                ratio=args.kl_anneal_ratio,
            )
        elif self.annealing_type == "Monotonic":
            self.L = self.frange_cycle_linear(
                n_iter=args.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=1,
                ratio=args.kl_anneal_ratio,
            )
        elif self.annealing_type == "Cosine":
            self.L = self.frange_cosine(
                n_iter=args.num_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=args.kl_anneal_cycle,
                ratio=args.kl_anneal_ratio,
            )
        elif self.annealing_type == "Constant":
            self.L = np.ones(args.num_epoch + 1) * 1e-4
        else:
            self.L = np.ones(args.num_epoch + 1)

    def update(self):
        self.iter += 1

    def get_beta(self):
        return self.L[self.iter]

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):

        # Ref : https://github.com/haofuml/cyclical_annealing
        L = np.ones(n_iter + 1)
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):
            v, i = start, 0

            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1

        return L

    def frange_cosine(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1.0):
        L = np.ones(n_iter + 1)
        period = n_iter / n_cycle
        for c in range(n_cycle):
            for i in range(int(period)):
                t = i / (period * ratio)
                v = start + (stop - start) * 0.5 * (1 - np.cos(np.pi * t))
                L[int(i + c * period)] = v
            L[int((c + 1) * period) :] = stop
        return L


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        self.Learned_Prior = Learned_Prior_Predictor(
            args.F_dim + args.L_dim, args.N_dim
        )
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim
        )
        decoder_fusion_in_dim = args.F_dim + args.L_dim + args.N_dim + args.F_dim
        self.Decoder_Fusion = Decoder_Fusion(
            decoder_fusion_in_dim, args.D_out_dim
        )

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        # Add weight decay to avoid overfitting and gradient explosion
        self.optim = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-5)
        self.kl_annealing = kl_annealing(args, current_epoch=0)

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim,
            milestones=[
                2,
            ],
            gamma=0.1,
        )
        self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        self.model_prefix = f"kl-type_{args.kl_anneal_type}_tfr_{args.tfr}_teacher-decay_{args.tfr_d_step}"
        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "psnr": [],
            "beta": [],
            "tfr": [],
        }

    def forward(self, img, label):
        pass

    def training_stage(self):

        min_val_loss = np.inf
        max_psnr = 0.0

        for i in range(self.current_epoch, self.args.num_epoch):
            train_loader = self.train_dataloader()
            # tfr is the probability that should the model adapt teacher forcing technique
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            train_loss = []
            for img, label in (pbar := tqdm(train_loader, ncols=170)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)

                loss = self.training_one_step(img, label, adapt_TeacherForcing)

                train_loss.append(loss.detach().cpu())

                beta = self.kl_annealing.get_beta()

                if adapt_TeacherForcing:
                    self.tqdm_bar(
                        "train [TeacherForcing: ON, {:.1f}], beta: {}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )
                else:
                    self.tqdm_bar(
                        "train [TeacherForcing: OFF, {:.1f}], beta: {}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )

            # if self.current_epoch % self.args.per_save == 0:
            #     self.save(
            #         os.path.join(
            #             self.args.save_root, f"epoch={self.current_epoch}.ckpt"
            #         )
            #     )

            val_loss, avg_psnr, _ = self.eval()
            tqdm.write(
                f"Validation - Avg Loss: {val_loss:.6f}, Avg PSNR: {avg_psnr:.6f}"
            )

            # We starting to save the model weight until over 10 epochs
            if (
                avg_psnr > max_psnr or val_loss < min_val_loss
            ) and self.current_epoch >= 5:
                max_psnr = max(avg_psnr, max_psnr)
                min_val_loss = min(val_loss, min_val_loss)
                self.save(
                    os.path.join(
                        self.args.save_root,
                        f"{self.model_prefix}_psnr_{avg_psnr:.6f}_val_loss_{val_loss:.6f}.ckpt",
                    )
                )

            # Save the history for plotting
            self.history["train_loss"].append(loss.item())
            self.history["val_loss"].append(val_loss)
            self.history["psnr"].append(avg_psnr)
            self.history["beta"].append(beta)
            self.history["tfr"].append(self.tfr)

            # Save the plot at last epoch
            if self.current_epoch == self.args.num_epoch - 1:
                self.plot_history()

            self.current_epoch += 1
            # update learning rate based on avg psnr
            self.scheduler.step()
            self.scheduler2.step(avg_psnr)

            # update kl annealing and teacher forcing ratio
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

    def plot_history(self):
        os.makedirs(f"plots/{self.model_prefix}", exist_ok=True)

        def save_multiple(metric_dict, title, filename):
            plt.figure()
            color_map = cm.get_cmap("tab10")

            for i, (label, values) in enumerate(metric_dict.items()):
                color = color_map(i % 10)
                plt.plot(values, label=label, color=color)

                values_array = np.array(values)
                valid_values = values_array[np.isfinite(values_array)]

                max_val = np.max(valid_values)
                min_val = np.min(valid_values)
                avg_val = np.mean(valid_values)
                max_idx = np.where(values_array == max_val)[0][0]
                min_idx = np.where(values_array == min_val)[0][0]

                plt.scatter(
                    max_idx,
                    max_val,
                    marker="o",
                    color=color,
                    label=f"{label} max: {max_val:.6f}",
                )
                plt.scatter(
                    min_idx,
                    min_val,
                    marker="x",
                    color=color,
                    label=f"{label} min: {min_val:.6f}",
                )
                plt.hlines(
                    avg_val,
                    0,
                    len(values) - 1,
                    linestyles="--",
                    colors=color,
                    label=f"{label} avg: {avg_val:.6f}",
                )

            plt.xlabel("Epoch")
            plt.ylabel(title + " (log scale)")
            plt.yscale("log")
            plt.title(f"{title} over Epochs")
            plt.grid(True)
            plt.legend(loc="best", fontsize="small")
            plt.tight_layout()
            plt.savefig(f"plots/{self.model_prefix}/{filename}.png")
            plt.close()

        def save_plot(metric, ylabel):
            values = self.history[metric]
            plt.figure()
            plt.plot(values, label=metric)

            values_array = np.array(values)
            valid_values = values_array[np.isfinite(values_array)]

            max_val = np.max(valid_values)
            min_val = np.min(valid_values)
            avg_val = np.mean(valid_values)
            max_idx = np.where(values_array == max_val)[0][0]
            min_idx = np.where(values_array == min_val)[0][0]
            plt.scatter(max_idx, max_val, marker="o", label=f"Max: {max_val:.6f}")
            plt.scatter(min_idx, min_val, marker="x", label=f"Min: {min_val:.6f}")
            plt.hlines(
                avg_val,
                0,
                len(values) - 1,
                linestyles="--",
                label=f"Avg: {avg_val:.6f}",
            )

            log = False
            if metric == "train_loss" or metric == "val_loss":
                plt.yscale("log")
                log = True

            plt.xlabel("Epoch")
            plt.ylabel(ylabel + " (log scale)" if log else ylabel)
            plt.title(f"{ylabel} over Epochs")
            plt.grid(True)
            plt.legend(loc="best", fontsize="small")
            plt.tight_layout()
            plt.savefig(f"plots/{self.model_prefix}/{metric}.png")
            plt.close()

        save_multiple(
            {"train": self.history["train_loss"], "val": self.history["val_loss"]},
            "Loss",
            "loss_compare",
        )
        save_plot("train_loss", "Train Loss")
        save_plot("val_loss", "Validation Loss")
        save_plot("psnr", "PSNR")
        save_plot("beta", "KL Beta")
        save_plot("tfr", "Teacher Forcing Ratio")

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        val_loss = []

        for img, label in (pbar := tqdm(val_loader, ncols=170)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, avg_psnr, psnr_list = self.val_one_step(img, label)
            val_loss.append(loss)
            self.tqdm_bar("val", pbar, loss, lr=self.scheduler.get_last_lr()[0])

        # This is correct since batch size is 1 in validation
        return loss, avg_psnr, psnr_list

    def training_one_step(self, imgs, labels, adapt_TeacherForcing):

        # img : torch.Size([2, 16, 3, 32, 64]) (Batch_Size,video length,channel,height,width)
        pred_img = imgs[:, 0]
        loss = torch.zeros(1, device=self.args.device)
        beta = self.kl_annealing.get_beta()

        with torch.no_grad(): # No need to track gradients for the skip feature source
            skip_feat = self.frame_transformation(imgs[:, 0])
        
        for i in range(1, self.train_vi_len):
            prev_img = imgs[:, i - 1] if adapt_TeacherForcing else pred_img

            gt_img_feat = self.frame_transformation(imgs[:, i])
            prev_img_feat = self.frame_transformation(prev_img)

            label_feat = self.label_transformation(labels[:, i])

            posterior_z, posterior_mu, posterior_logvar = self.Gaussian_Predictor(gt_img_feat, label_feat)
            prior_mu, prior_logvar = self.Learned_Prior(prev_img_feat, label_feat)

            decorder_fusion_out = self.Decoder_Fusion(prev_img_feat, label_feat, posterior_z, skip_feat)
            pred_img = self.Generator(decorder_fusion_out)

            mse_loss = self.mse_criterion(pred_img, imgs[:, i])
            # The KL loss is now between the posterior and the learned prior
            kld_loss = kl_criterion_lp(posterior_mu, posterior_logvar, prior_mu, prior_logvar, self.batch_size)
            
            loss += mse_loss + beta * kld_loss

        # Divide (self.train_vi_len - 1), since we skip the prediction of first frame
        loss = loss / (self.train_vi_len - 1)

        # Check for NaN or Inf before backward
        if torch.isnan(loss) or torch.isinf(loss):
            tqdm.write("Warning: Loss is NaN or Inf. Skipping this batch.")
            return torch.tensor(np.inf, device=self.args.device)

        self.optim.zero_grad()
        loss.backward()

        # Clip gradient to prevent explosion
        # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer_step()

        return loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def val_one_step(self, imgs, labels):
        # imgs shape: torch.Size([1, 630, 3, 32, 64])
        pred_img = imgs[:, 0]
        loss = torch.zeros(1) # Loss is on CPU now
        beta = self.kl_annealing.get_beta()
        psnr = []
        with torch.no_grad():
            skip_feat = self.frame_transformation(imgs[:, 0])
        # Keep track of the previous frame's feature
        prev_img_feat = self.frame_transformation(pred_img)

        for i in range(1, self.val_vi_len):
            # --- CORRECTED INFERENCE LOGIC ---
            # DO NOT use ground truth future frame here.
            
            # Feature from the current pose label
            label_feat = self.label_transformation(labels[:, i])

            # 1. Get the PRIOR distribution from the Learned Prior Network
            prior_mu, prior_logvar = self.Learned_Prior(prev_img_feat, label_feat)

            # 2. Sample z from this PRIOR distribution for generation
            z = self.reparameterize(prior_mu, prior_logvar)

            # 3. Generate the next frame
            decorder_fusion_out = self.Decoder_Fusion(prev_img_feat, label_feat, z, skip_feat)
            pred_img = self.Generator(decorder_fusion_out)
            
            # For validation loss, we can still compare to ground truth
            # We can use the posterior from ground truth to calculate a consistent validation loss
            gt_img_feat = self.frame_transformation(imgs[:, i])
            _, posterior_mu, posterior_logvar = self.Gaussian_Predictor(gt_img_feat, label_feat)
            
            mse_loss = self.mse_criterion(pred_img, imgs[:, i])
            kld_loss = kl_criterion_lp(posterior_mu, posterior_logvar, prior_mu, prior_logvar, 1) # Batch size is 1 for val
            
            loss += mse_loss.cpu() + (beta * kld_loss).cpu()
            psnr.append(Generate_PSNR(pred_img, imgs[:, i]).cpu().item())

            # Update prev_img_feat for the next iteration using the newly generated frame
            prev_img_feat = self.frame_transformation(pred_img)


        # Divide (self.val_vi_len - 1), since we skip the prediction of first frame
        loss = loss / (self.val_vi_len - 1)

        return loss.item(), np.mean(psnr), psnr

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(
            img_name,
            format="GIF",
            append_images=new_list,
            save_all=True,
            duration=40,
            loop=0,
        )

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )

        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="train",
            video_len=self.train_vi_len,
            partial=args.fast_partial if self.args.fast_train else args.partial,
        )
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="val",
            video_len=self.val_vi_len,
            partial=1.0,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return val_loader

    def teacher_forcing_ratio_update(self):
        # For each tfr_sde steps , the ratio will decay tfr_d_step
        if (self.current_epoch) % self.tfr_sde == 0:
            self.tfr -= self.tfr_d_step
            self.tfr = max(0, self.tfr)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(
            f"({mode}) Epoch [{self.current_epoch + 1}/{self.args.num_epoch}], lr:{lr}",
            refresh=False,
        )
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.optim.state_dict(),
                "lr": self.scheduler.get_last_lr()[0],
                "tfr": self.tfr,
                "last_epoch": self.current_epoch,
                "history": self.history,
            },
            path,
        )
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:

            checkpoint = torch.load(self.args.ckpt_path, weights_only=False)
            self.load_state_dict(checkpoint["state_dict"], strict=True)

            # self.args.lr = checkpoint["lr"]

            self.tfr = checkpoint["tfr"]

            self.optim = optim.Adam(
                self.parameters(), lr=self.args.lr, weight_decay=1e-5
            )
            self.optim.load_state_dict(checkpoint["optimizer"])

            for i in self.optim.param_groups:
                i["lr"] = checkpoint["lr"]

            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optim,
                milestones=[],
                gamma=0.1,
            )
            self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
            )
            self.kl_annealing = kl_annealing(
                self.args, current_epoch=checkpoint["last_epoch"]
            )
            self.current_epoch = checkpoint["last_epoch"]

            history = checkpoint["history"]
            last_epoch = checkpoint["last_epoch"]
            self.history = {
                "train_loss": history["train_loss"][: last_epoch + 1],
                "val_loss": history["val_loss"][: last_epoch + 1],
                "psnr": history["psnr"][: last_epoch + 1],
                "beta": history["beta"][: last_epoch + 1],
                "tfr": history["tfr"][: last_epoch + 1],
            }

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()

    def load_weight(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)


def main(args):

    os.makedirs(args.save_root, exist_ok=True)
    # if args.ckpt_path is not None:
    #     os.makedirs(args.ckpt_path, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:

        model.eval()
    else:
        model.training_stage()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3, help="init ial learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--store_visualization",
        action="store_true",
        help="If you want to see the result while training",
    )
    parser.add_argument(
        "--DR",
        type=str,
        default="./LAB4_Dataset/LAB4_Dataset",
        help="Your Dataset Path",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="saves/train",
        help="The path to save your data",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=100, help="number of total epoch"
    )
    parser.add_argument(
        "--per_save", type=int, default=3, help="Save checkpoint every seted epoch"
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Part of the training dataset to be trained",
    )
    parser.add_argument(
        "--train_vi_len", type=int, default=16, help="Training video length"
    )
    parser.add_argument(
        "--val_vi_len", type=int, default=630, help="valdation video length"
    )
    parser.add_argument(
        "--frame_H", type=int, default=32, help="Height input image to be resize"
    )
    parser.add_argument(
        "--frame_W", type=int, default=64, help="Width input image to be resize"
    )

    # Module parameters setting
    parser.add_argument(
        "--F_dim", type=int, default=128, help="Dimension of feature human frame"
    )
    parser.add_argument(
        "--L_dim", type=int, default=32, help="Dimension of feature label frame"
    )
    parser.add_argument("--N_dim", type=int, default=12, help="Dimension of the Noise")
    parser.add_argument(
        "--D_out_dim",
        type=int,
        default=192,
        help="Dimension of the output in Decoder_Fusion",
    )

    # Teacher Forcing strategy
    parser.add_argument(
        "--tfr", type=float, default=1.0, help="The initial teacher forcing ratio"
    )
    parser.add_argument(
        "--tfr_sde",
        type=int,
        default=10,
        help="The epoch that teacher forcing ratio start to decay",
    )
    parser.add_argument(
        "--tfr_d_step",
        type=float,
        default=0.1,
        help="Decay step that teacher forcing ratio adopted",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="The path of your checkpoints",
    )

    # Training Strategy
    parser.add_argument("--fast_train", action="store_true")
    parser.add_argument(
        "--fast_partial",
        type=float,
        default=0.4,
        help="Use part of the training data to fasten the convergence",
    )
    parser.add_argument(
        "--fast_train_epoch",
        type=int,
        default=5,
        help="Number of epoch to use fast train mode",
    )

    # Kl annealing stratedy arguments
    parser.add_argument(
        "--kl_anneal_type",
        type=str,
        default="Cyclical",
        choices=["Cyclical", "Monotonic", "Without", "Cosine", "Constant"],
        help="",
    )
    parser.add_argument("--kl_anneal_cycle", type=int, default=10, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")

    args = parser.parse_args()

    main(args)
