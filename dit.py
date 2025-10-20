import torch.nn as nn
from patchify import Patchify
from transformer import Transformer
import torch
import matplotlib.pyplot as plt
from einops import rearrange


def get_time_embedding(timestep, timestep_embed_dim):

    factor = 10_000 ** (
        torch.arange(start=0, end=timestep_embed_dim // 2, dtype=torch.float32)
        / (timestep_embed_dim // 2)
    )

    t_embed = timestep[:, None].repeat(1, timestep_embed_dim // 2) / factor
    t_embed = torch.cat([t_embed.sin(), t_embed.cos()], dim=-1)

    return t_embed


class DiT(nn.Module):
    def __init__(
        self,
        img_height,
        img_width,
        img_channels,
        patch_height,
        patch_width,
        hidden_size,
        num_heads,
        num_layers,
        timestep_embed_dim,
        class_embed_dim,
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.timestep_embed_dim = timestep_embed_dim
        self.class_embed_dim = class_embed_dim

        self.patchify = Patchify(
            img_height, img_width, img_channels, patch_height, patch_width, hidden_size
        )

        # class embedding (10 classes for MNIST)
        self.class_embed = nn.Embedding(10, class_embed_dim)
        self.c_proj = nn.Sequential(
            nn.Linear(class_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.t_proj = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.layers = nn.ModuleList(
            [Transformer(hidden_size, num_heads, num_layers) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaptive_norm_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.proj_out = nn.Linear(
            self.hidden_size, self.patch_height * self.patch_width * self.img_channels
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)
        nn.init.normal_(self.c_proj[0].weight, std=0.02)
        nn.init.normal_(self.c_proj[2].weight, std=0.02)

        nn.init.normal_(self.adaptive_norm_mlp[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t, c):
        out = self.patchify(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_embed_dim)
        t_emb = self.t_proj(t_emb)

        c_emb = self.class_embed(torch.as_tensor(c).long())
        c_emb = self.c_proj(c_emb)

        cond = t_emb + c_emb

        for layer in self.layers:
            out = layer(out, cond)

        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_mlp(cond).chunk(2, dim=1)
        out = self.norm(out) * (
            1 + pre_mlp_scale.unsqueeze(1)
        ) + pre_mlp_shift.unsqueeze(1)

        out = self.proj_out(out)
        out = rearrange(
            out,
            "b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)",
            nh=self.img_height // self.patch_height,
            nw=self.img_width // self.patch_width,
            ph=self.patch_height,
            pw=self.patch_width,
        )

        return out


if __name__ == "__main__":
    dit = DiT(
        img_height=28,
        img_width=28,
        img_channels=1,
        patch_height=7,
        patch_width=7,
        hidden_size=128,
        num_heads=8,
        num_layers=12,
        timestep_embed_dim=128,
        class_embed_dim=128,
    )
    x = torch.from_numpy(plt.imread("train_image_385.png"))
    x = x[:, :, 0]
    x = x.reshape(1, 1, 28, 28)
    t = torch.randint(0, 100, (1,))
    c = torch.randint(0, 10, (1,))
    print("x.shape: ", x.shape)
    print("t ", t.shape)
    print("c ", c)
    out = dit(x, t, c)
    print("out.shape: ", out.shape)
