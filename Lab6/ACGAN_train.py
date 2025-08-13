import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from PIL import Image

from dataset import iCLEVRDataset
from ACGAN import Generator, Discriminator
from evaluator import evaluation_model
from torchvision import transforms

import wandb

auxiliary_loss_fn = torch.nn.BCELoss()


def adversarial_loss_d(r_logit, f_logit):
    r_loss = torch.mean(torch.relu(1.0 - r_logit))
    f_loss = torch.mean(torch.relu(1.0 + f_logit))
    return r_loss + f_loss


def adversarial_loss_g(f_logit):
    return -torch.mean(f_logit)


@torch.no_grad()
def run_evaluation(generator, evaluator, latent_dim, device, mode):

    generator.eval()
    data = iCLEVRDataset(mode=mode)
    total_score = 0.0

    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    for i, (_, gt_condition) in enumerate(data):
        condition = gt_condition.unsqueeze(0).to(device)
        z = torch.randn(1, latent_dim, device=device)
        gen_img_tensor = generator(z, condition)

        gen_img_vis = (gen_img_tensor / 2 + 0.5).clamp(0, 1)

        pil_img = transforms.ToPILImage()(gen_img_vis.squeeze(0))

        image_tensor_eval = eval_transform(pil_img).unsqueeze(0).to(device)

        gt_condition = gt_condition.unsqueeze(0).to(device)
        score = evaluator.eval(image_tensor_eval, gt_condition)
        total_score += score

    avg_score = total_score / len(data) if len(data) > 0 else 0
    generator.train()
    return avg_score


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(
        project="AC-GAN Training",
        config=args,
        name=f"ACGAN-lr_{args.lr}-bs_{args.batch_size}",
    )
    transform = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = iCLEVRDataset(mode="train", transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    generator = Generator(z_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    evaluator = evaluation_model()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999)
    )

    save_dir = Path(args.workspace) / "AGAN_CKPT"
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb.watch(generator, log="all", log_freq=100)
    wandb.watch(discriminator, log="all", log_freq=100)

    best_test_accuracy = 0.0
    best_new_test_accuracy = 0.0

    n_vis_images = 32
    fixed_noise = torch.randn(n_vis_images, args.latent_dim, device=device)
    vis_dataloader = DataLoader(dataset, batch_size=n_vis_images, shuffle=True)
    _, fixed_conditions = next(iter(vis_dataloader))
    fixed_conditions = fixed_conditions.to(device)

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        progress_bar = tqdm(
            total=len(dataloader), desc=f"Epoch [{epoch+1}/{args.epochs}]"
        )

        for i, (real_imgs, conditions) in enumerate(dataloader):
            real_imgs, conditions = real_imgs.to(device), conditions.to(device)
            batch_size = real_imgs.size(0)

            # Trainining Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z, conditions).detach()

            real_pred_adv, real_pred_aux = discriminator(real_imgs)
            fake_pred_adv, _ = discriminator(gen_imgs)

            d_adv_loss = adversarial_loss_d(real_pred_adv, fake_pred_adv)
            d_aux_loss = auxiliary_loss_fn(real_pred_aux, conditions)
            d_loss = d_adv_loss + args.aux_weight * d_aux_loss
            d_loss.backward()
            optimizer_D.step()

            # Training Generator
            optimizer_G.zero_grad()
            z_g = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs_for_g = generator(z_g, conditions)
            g_pred_adv, g_pred_aux = discriminator(gen_imgs_for_g)

            g_adv_loss = adversarial_loss_g(g_pred_adv)
            g_aux_loss = auxiliary_loss_fn(g_pred_aux, conditions)
            g_loss = g_adv_loss + args.aux_weight * g_aux_loss
            g_loss.backward()
            optimizer_G.step()

            progress_bar.update(1)
            progress_bar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        wandb.log(
            {
                "epoch": epoch + 1,
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
            }
        )

        if (epoch + 1) % args.log_interval == 0:
            generator.eval()
            with torch.no_grad():
                img_grid = make_grid(
                    (generator(fixed_noise, fixed_conditions) / 2 + 0.5).clamp(0, 1),
                    nrow=8,
                )
                wandb.log({"generated_images": wandb.Image(img_grid)})
            test_acc = run_evaluation(
                generator, evaluator, args.latent_dim, device, "test"
            )
            new_test_acc = run_evaluation(
                generator, evaluator, args.latent_dim, device, "new_test"
            )

            tqdm.write(
                f"Epoch {epoch+1}: Test Accuracy = {test_acc:.4f}, New Test Accuracy = {new_test_acc:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "test_accuracy": test_acc,
                    "new_test_accuracy": new_test_acc,
                }
            )

            if test_acc > best_test_accuracy and new_test_acc > best_new_test_accuracy:
                best_new_test_accuracy = new_test_acc
                best_test_accuracy = test_acc
                torch.save(generator.state_dict(), save_dir / "best_generator.pth")
                torch.save(
                    discriminator.state_dict(), save_dir / "best_discriminator.pth"
                )
                tqdm.write(
                    "Save model with test_ccc {:.4f} and new_test_acc{:.4f}".format(
                        test_acc, new_test_acc
                    )
                )
            generator.train()

    wandb.finish()


@torch.no_grad()
def inference(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.workspace) / "ACGAN_inference_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator(z_dim=args.latent_dim).to(device)
    eval_model = evaluation_model()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    for mode in ["test", "new_test"]:
        tqdm.write(f"\nProcessing '{mode}.json'...")

        dataset = iCLEVRDataset(mode=mode)

        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        _, labels = next(iter(dataloader))
        labels = labels.to(device)

        z = torch.randn(len(labels), args.latent_dim).to(device)

        gen_imgs = generator(z, labels)

        acc = eval_model.eval(gen_imgs, labels)
        print(f"Accuracy for {mode}.json: {acc:.4f}")

        gen_imgs_for_saving = (gen_imgs / 2 + 0.5).clamp(0, 1)

        model_name_sanitized = (
            str(model_path.name).replace(".pth", "").replace("/", "_")
        )
        save_path = output_dir / f"{model_name_sanitized}_{mode}_acc_{acc:.4f}.png"

        save_image(gen_imgs_for_saving, save_path, nrow=8)
        print(f"Saved image grid to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional GAN (cGAN) for iCLEVR")
    parser.add_argument(
        "--workspace", type=str, default="./acgan_result", help="Root directory"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--inference", action="store_true", help="Generate images")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--lr_g", type=float, default=1e-4, help="Learning rate for Generator"
    )
    parser.add_argument(
        "--lr_d", type=float, default=4e-4, help="Learning rate for Discriminator"
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument(
        "--aux_weight",
        type=float,
        default=100,
        help="Weight for the auxiliary classification loss",
    )
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save model checkpoints every N epochs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="acgan_result\\AGAN_CKPT\\best_generator.pth",
        help="Path to trained generator for inference",
    )
    parser.add_argument("--log_interval", type=int, default=1)
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.inference:
        inference(args)
    else:
        print("Please specify --train or --inference.")
