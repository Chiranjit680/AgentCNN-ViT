from typing import List
import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .pos_encoding import PosEncoding
from vit_mutual.models.transformer import Transformer
from vit_mutual.models.transformer.transformer import MLP, MultiHeadSelfAttention


class ViT(nn.Module):
    def __init__(
        self,
        patch_embed: PatchEmbed,
        pos_embed: PosEncoding,
        transformer: Transformer,
        num_classes: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.transformer = transformer

        embed_dim = self.transformer.embed_dim
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def get_mhsa(self) -> List[MultiHeadSelfAttention]:
        return [layer.attention for layer in self.transformer.layers]

    def get_mlp(self) -> List[MLP]:
        return [layer.mlp for layer in self.transformer.layers]

    def forward(self, img: torch.Tensor):
        # Patch embedding and positional encoding
        seq, (Hp, Wp) = self.patch_embed(img)       # seq: [N, B, D]
        seq = self.pos_embed(seq)                   # [N, B, D]

        # Transformer
        seq = self.transformer(seq)                 # [N, B, D]

        # Convert back to spatial map
        seq = seq.permute(1, 2, 0).contiguous()     # [B, D, N]
        x = seq.view(img.size(0), self.transformer.embed_dim, Hp, Wp)  # [B, D, Hp, Wp]

        # Segmentation head
        x = self.seg_head(x)                        # [B, num_classes, Hp, Wp]

        # Upsample to original image resolution
        seg_map = nn.functional.interpolate(
            x,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return seg_map  # [B, num_classes, H, W]
