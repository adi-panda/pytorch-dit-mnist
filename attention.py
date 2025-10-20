import torch.nn as nn
import math
import torch
from einops import rearrange


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # head dim is hidden size divided by number of heads
        self.head_dim = self.hidden_size // self.num_heads

        # so attention dim is just hidden size
        # idk if this is correct, but oh well

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.xavier_uniform_(self.out_proj[0].weight)
        nn.init.constant_(self.out_proj[0].bias, 0)

    def forward(self, x):

        # batch size x number of tokens(patches) x hidden size
        B, N = x.shape[:2]

        q, k, v = self.qkv_proj(x).split(self.hidden_size, dim=-1)

        q = rearrange(
            q,
            "b n (num_head head_dim) -> b num_head n head_dim",
            num_head=self.num_heads,
            head_dim=self.head_dim,
        )

        k = rearrange(
            k,
            "b n (num_head head_dim) -> b num_head n head_dim",
            num_head=self.num_heads,
            head_dim=self.head_dim,
        )

        v = rearrange(
            v,
            "b n (num_head head_dim) -> b num_head n head_dim",
            num_head=self.num_heads,
            head_dim=self.head_dim,
        )

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)

        out = rearrange(
            out,
            "b num_head n head_dim -> b n (num_head head_dim)",
            num_head=self.num_heads,
            head_dim=self.head_dim,
        )

        out = self.out_proj(out)

        return out
