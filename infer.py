import torch
import torchvision
import argparse
from tqdm import tqdm
import os
from dit import DiT
from scheduler import LinearScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def sample(model, scheduler, class_label, img_size, output_dir, epoch):
    xt = torch.randn(img_size).to(device)

    for t in tqdm(range(1000), desc="Sampling"):

        noise_prediction = model(
            xt,
            torch.as_tensor(t).unsqueeze(0).to(device),
            torch.tensor([class_label]).to(device),
        )
        xt, x0_pred = scheduler.get_prev_timestep(xt, noise_prediction, t)

        img = xt
        img = xt[0, 0, :, :]

    img = torch.clamp(img, -1.0, 1.0).detach().cpu()
    img = (img + 1.0) / 2.0

    torchvision.utils.save_image(img, os.path.join(output_dir, f"sample_{epoch}.png"))

    print("Sampling complete")


def infer(epoch):

    scheduler = LinearScheduler(1000, 0.0001, 0.02)

    img_size = (1, 1, 28, 28)

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
    model.eval()

    model.load_state_dict(torch.load(f"checkpoints/model_epoch_{epoch}.pth"))

    print(f"loaded dit checkpoint")

    with torch.no_grad():
        for class_label in range(10):
            sample(model, scheduler, class_label, img_size, "samples", epoch)


if __name__ == "__main__":
    sample()
