import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.transforms as transforms
import json
from PIL import Image


class iCLEVRDataset(Dataset):

    def __init__(self, mode="train", transform=None):

        assert mode in [
            "train",
            "test",
            "new_test",
        ], "mode should be 'train', 'test', or 'new_test'"
        self.mode = mode
        self.transform = transform

        with open("objects.json", "r") as f:
            self.object_to_idx = json.load(f)
            # print(self.object_to_idx)

        # Dictionary mapping object names to indices
        self.idx_to_object = {v: k for k, v in self.object_to_idx.items()}
        self.num_classes = len(self.object_to_idx)

        if self.mode == "train":
            with open("train.json", "r") as f:
                self.data = json.load(f)
            # name of the image files
            self.image_files = list(self.data.keys())
        else:
            file_path = "test.json" if mode == "test" else "new_test.json"
            with open(file_path, "r") as f:
                self.data = json.load(f)

    def __len__(self):
        if self.mode == "train":
            return len(self.image_files)

        return len(self.data)

    def __getitem__(self, idx):

        if self.mode == "train":
            img_name = self.image_files[idx]
            img_path = "iclevr\\" + img_name
            image = Image.open(img_path).convert("RGB")
            labels = self.data[img_name]
            # Transform the image if a transform is provided (e.g., normalization, tensor conversion)
            if self.transform:
                image = self.transform(image)
        else:
            # Testing data , we expect that the model could generate the image corresponding to the labels
            image = torch.zeros(3, 64, 64)
            labels = self.data[idx]

        condition = torch.zeros(self.num_classes)
        for obj in labels:
            condition[self.object_to_idx[obj]] = 1

        return image, condition


if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = iCLEVRDataset(mode="train", transform=transform)
    print(f"Dataset initialized with {len(dataset)} samples.")

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        print(batch)
