import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm


from dataset import iCLEVRDataset
from conditionDDPM import DiffusionModel

from diffusers import DDPMScheduler
from accelerate import Accelerator
import wandb


def train(args):

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        wandb.init(
            project="DDPM Training",
            config=args,
            name=f"DDPM-lr_{args.lr}_bs_{args.batch_size}_epoch_{args.epochs}",
        )

    transform = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_dataset = iCLEVRDataset(transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = DiffusionModel(
        resolution=args.resolution,
        num_classes=full_dataset.num_classes,
        cond_dim=args.cond_dim,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=100)
        best_val_loss = float("inf")

    save_dir = Path(args.workspace) / "DDPM_CKPT"
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):

        model.train()
        train_loss_acc = 0.0
        train_progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1} / {args.epochs} Train",
            disable=not accelerator.is_local_main_process,
        )

        for step, (clean_images, conditions) in enumerate(train_dataloader):

            conditions = conditions.to(accelerator.device)
            # Sample noise from normal distribution
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=accelerator.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, conditions)
                loss = F.mse_loss(noise_pred.float(), noise.float())
                train_loss_acc += loss.detach().item()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            train_progress_bar.update(1)
            train_progress_bar.set_postfix(loss=loss.detach().item())

        # Evaluation
        model.eval()
        val_loss_acc = 0.0
        val_progress_bar = tqdm(
            total=len(val_dataloader),
            desc=f"Epoch {epoch+1} Val",
            disable=not accelerator.is_local_main_process,
        )

        with torch.no_grad():
            for step, (clean_images, conditions) in enumerate(val_dataloader):
                conditions = conditions.to(accelerator.device)
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (clean_images.shape[0],),
                    device=accelerator.device,
                ).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                noise_pred = model(noisy_images, timesteps, conditions)
                loss = F.mse_loss(noise_pred.float(), noise.float())
                val_loss_acc += loss.item()
                val_progress_bar.update(1)

        # Average losses
        avg_train_loss = train_loss_acc / len(train_dataloader)
        avg_val_loss = val_loss_acc / len(val_dataloader)

        tqdm.write(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Store model weight if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = save_dir / "best_model.pth"
            torch.save(accelerator.unwrap_model(model).state_dict(), str(save_path))
            tqdm.write(
                f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}"
            )

    if accelerator.is_main_process:
        wandb.finish()


@torch.no_grad()
def inference(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.workspace) / "images_from_best"
    test_dir = output_dir / "test"
    new_test_dir = output_dir / "new_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    new_test_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = iCLEVRDataset()
    model = DiffusionModel(
        resolution=args.resolution,
        num_classes=dataset_info.num_classes,
        cond_dim=args.cond_dim,
    ).to(device)

    model_path = Path(args.workspace) / "checkpoints_best" / "best_model.pth"
    if not model_path.exists():
        print(f"Error: Best model not found at {model_path}")
        print("Please run training first to generate the best model.")
        return

    model.load_state_dict(torch.load(model_path))

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )
    model.eval()

    def generate_and_save(mode):
        data = iCLEVRDataset(mode=mode)
        save_dir = test_dir if mode == "test" else new_test_dir

        all_images = []
        for i, (_, condition) in enumerate(data):
            condition = condition.unsqueeze(0).to(device)

            print(f"Generating image {i+1}/{len(data)} for {mode}.json")
            image = torch.randn((1, 3, args.resolution, args.resolution), device=device)
            noise_scheduler.set_timesteps(50)

            for t in tqdm(noise_scheduler.timesteps):
                noise_pred = model(image, t, condition)
                image = noise_scheduler.step(noise_pred, t, image).prev_sample

            image = (image / 2 + 0.5).clamp(0, 1)
            save_image(image, save_dir / f"{i}.png")
            all_images.append(image[0])

        grid = make_grid(all_images, nrow=8)
        save_image(grid, output_dir / f"grid_{mode}.png")
        print(f"Saved image grid to {output_dir / f'grid_{mode}.png'}")

    generate_and_save("test")
    generate_and_save("new_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separated Conditional DDPM with Hugging Face"
    )
    parser.add_argument(
        "--workspace", type=str, default="./result", help="Root directory"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--inference", action="store_true", help="Generate images")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--inference_epoch", type=int, default=100)
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.inference:
        inference(args)
    else:
        print("Please specify --train or --inference.")
