# Modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py #

import math
from einops import rearrange, repeat

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed Forward Network"""
    def __init__(self, dim, embed_dim, dropout):
        super(MLP, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        out = self.main(x)
        return out


class Attention(nn.Module):
    """Attention Module for Self-Attention"""
    def __init__(self, embed_dim, num_heads, dropout):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        
        self.scale = math.sqrt(head_dim)

        self.qkv = nn.Linear(embed_dim, head_dim * num_heads * 3, bias = False)
        self.softmax = nn.Softmax(dim=-1)
        
        self.fc = nn.Sequential(
            nn.Linear(head_dim * num_heads, embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        b, n, h = batch_size, seq_len, self.num_heads

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        attn = self.scale * torch.einsum('bhid, bhjd -> bhij', q, k)
        
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.fc(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer Block that Combines Attention Module and MLP Module"""
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_out):
        super(TransformerBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.drop_out = nn.Dropout(drop_out)

        self.attn = Attention(embed_dim, num_heads, drop_out)
        self.mlp = MLP(embed_dim, mlp_dim, drop_out)

    def forward(self, x):
        out = self.layer_norm1(x)
        out = self.attn(out)
        out = self.drop_out(out)
        x = x + out

        out = self.layer_norm2(x)
        out = self.mlp(out)
        out = out + x
        return out


class VisionTransformer(nn.Module):
    """End-to-End Architecture for Vision Transformer"""
    def __init__(self,  in_channels, embed_dim, patch_size, num_layers, num_heads, mlp_dim, dropout, input_size, num_classes):
        super(VisionTransformer, self).__init__()
        
        num_patches = (input_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

        blocks = [TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        self.transformer = nn.Sequential(*blocks)

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        batch_size, seq_len, _ = x.shape
        b, n = batch_size, seq_len

        class_tokens = repeat(self.class_token, '() n d -> b n d', b = b)

        out = torch.cat((class_tokens, x), dim=1)
        out += self.pos_embedding[:, :(n + 1)]
        out = self.dropout(out)

        out = self.transformer(out)

        out = out[:, 0]
        out = self.mlp(out)

        return out