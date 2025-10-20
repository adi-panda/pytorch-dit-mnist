import torch.nn as nn
from attention import AttentionBlock
import torch


class Transformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers):
        super().__init__()

        self.hidden_size = hidden_size
        mlp_hidden_dim = 4 * hidden_size

        self.num_heads = num_heads
        self.num_layers = num_layers

        self.attention_layer_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.attention_block = AttentionBlock(hidden_size, num_heads)

        self.mlp_attention_layer_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.mlp_block = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.adaptive_norm_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)

        nn.init.constant_(self.adaptive_norm_mlp[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)

    def forward(self, x, condition):
        scale_shift = self.adaptive_norm_mlp(condition).chunk(6, dim=1)
        (
            pre_attn_shift,
            pre_attn_scale,
            post_attn_scale,
            pre_mlp_shift,
            pre_mlp_scale,
            post_mlp_scale,
        ) = scale_shift
        out = x

        attn_norm_out = self.attention_layer_norm(out) * (
            1 + pre_attn_scale.unsqueeze(1)
        ) + pre_attn_shift.unsqueeze(1)

        out = out + post_attn_scale.unsqueeze(1) * self.attention_block(attn_norm_out)

        mlp_norm_out = self.mlp_attention_layer_norm(out) * (
            1 + pre_mlp_scale.unsqueeze(1)
        ) + pre_mlp_shift.unsqueeze(1)

        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_out)

        return out


if __name__ == "__main__":
    transformer = Transformer(128, 8, 12)
    x = torch.randn(1, 100, 128)
    condition = torch.randn(1, 128)
    out = transformer(x, condition)
    print(out.shape)
