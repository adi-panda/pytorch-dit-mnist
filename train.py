import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from load_data import MnistDataloader
from scheduler import LinearScheduler
import matplotlib.pyplot as plt
from dit import DiT
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data import TensorDataset

input_path = "./mnist_data"
training_images_filepath = join(
    input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = join(
    input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def to_tensor(images, labels, norm="0_1"):
    # images: list of (28,28) uint8 arrays; labels: array/list of ints
    x = torch.tensor(np.array(images), dtype=torch.float32)  # (N, 28, 28)
    if norm == "0_1":
        x = x / 255.0
    elif norm == "minus1_1":
        x = (x / 255.0) * 2 - 1
    x = x.unsqueeze(1)  # (N, 1, 28, 28)
    y = torch.tensor(np.array(labels), dtype=torch.long)  # (N,)
    return x, y


def train():
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    X_train, Y_train = to_tensor(x_train, y_train, norm="0_1")
    X_test, Y_test = to_tensor(x_test, y_test, norm="0_1")

    train_dataset = TensorDataset(X_train, Y_train)

    scheduler = LinearScheduler(1000, 0.0001, 0.02)

    data_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    im_size = X_train[0].shape
    print(f"Image size: {im_size}")

    model = DiT(
        img_height=im_size[1],
        img_width=im_size[2],
        img_channels=im_size[0],
        patch_height=7,
        patch_width=7,
        hidden_size=128,
        num_heads=8,
        num_layers=12,
        timestep_embed_dim=128,
        class_embed_dim=128,
    ).to(device)
    model.train()

    num_epochs = 100
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    criterion = torch.nn.MSELoss()

    acc_steps = 1
    for epoch in tqdm(range(num_epochs), desc="Training"):
        losses = []
        step_count = 0
        for imgs, labels in data_loader:
            step_count += 1
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)

            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, 1000, (imgs.shape[0],)).to(device)

            noisy = scheduler.add_noise(imgs, t, noise)

            pred = model(noisy, t, labels)
            loss = criterion(pred, noise)
            losses.append(loss.item())
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()
        print("Finished epoch:{} | Loss : {:.4f}".format(epoch + 1, np.mean(losses)))
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")

    print("Training complete")


if __name__ == "__main__":
    train()
