import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt


def get_sinusoidal_pos_embed(embed_dim, grid_size_x, grid_size_y):
    grid_h = torch.arange(grid_size_y)
    grid_w = torch.arange(grid_size_x)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)

    # print for debugging
    # print(grid)

    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor is
    factor = 10_000 ** (
        torch.arange(start=0, end=embed_dim // 4, dtype=torch.float32)
        / (embed_dim // 4)
    )

    grid_h_embed = grid_h_positions[:, None].repeat(1, embed_dim // 4) / factor
    grid_h_embed = torch.cat([grid_h_embed.sin(), grid_h_embed.cos()], dim=-1)

    grid_w_embed = grid_w_positions[:, None].repeat(1, embed_dim // 4) / factor
    grid_w_embed = torch.cat([grid_w_embed.sin(), grid_w_embed.cos()], dim=-1)

    return torch.cat([grid_h_embed, grid_w_embed], dim=-1)


class Patchify(nn.Module):
    def __init__(
        self,
        img_height,
        img_width,
        img_channels,
        patch_height,
        patch_width,
        hidden_size,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.hidden_size = hidden_size

        patch_dim = self.img_channels * self.patch_height * self.patch_width
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_size).to(self.device)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.patch_embed[0].weight.to(self.device))
        nn.init.constant_(self.patch_embed[0].bias.to(self.device), 0)

    def forward(self, x):
        grid_size_x = self.img_width // self.patch_width
        grid_size_y = self.img_height // self.patch_height

        # rearrange using einpos
        # B, C, H, W -> B, (Patches along height * Patches along width), Patch Dimension
        # Number of tokens = Patches along height * Patches along width
        out = rearrange(
            x,
            "b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)",
            ph=self.patch_height,
            pw=self.patch_width,
        ).to(self.device)

        # show one patch for debugging
        # first_patch = out[0, 9, :].reshape(self.patch_height, self.patch_width)
        # plt.imsave("test_patch.png", first_patch, cmap=plt.cm.gray)

        out = self.patch_embed(out).to(self.device)

        # get positional embeddings from sinusoidal position embedding
        pos_embed = get_sinusoidal_pos_embed(
            self.hidden_size,
            grid_size_x,
            grid_size_y,
        ).to(self.device)

        # print(out.shape)
        # print(pos_embed.shape)
        out = out + pos_embed.to(self.device)

        return out


if __name__ == "__main__":
    x = torch.from_numpy(plt.imread("train_image_385.png"))
    x = x[:, :, 0]
    x = x.reshape(1, 1, 28, 28)

    # show the image for debugging
    plt.imsave("test_image.png", x[0, 0, :, :], cmap=plt.cm.gray)
    plt.show()

    patchify = Patchify(28, 28, 1, 7, 7, 128)
    out = patchify(x)

    # print(out.shape)
