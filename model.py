import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


# ---------------------------------------------------------------- ViT (scratch)
class PatchEmbed(nn.Module):
    """Conv with stride = patch size == linear projection of flattened patches."""

    def __init__(self, patch, in_ch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):                       # [B,C,H,W] -> [B,N,D]
        return self.proj(x).flatten(2).transpose(1, 2)


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim, heads, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    def __init__(self, img_size=224, patch=16, in_ch=3, dim=384, depth=6, heads=6, drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(patch, in_ch, dim)
        n = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]               # CLS token


# ---------------------------------------------------------------- BERT tower
class TextEncoder(nn.Module):
    def __init__(self, name="distilbert-base-uncased", pretrained=True, tiny=False):
        super().__init__()
        if tiny:                                # offline, for smoke tests
            cfg = AutoConfig.for_model("bert", hidden_size=64, num_hidden_layers=2,
                                       num_attention_heads=2, intermediate_size=128,
                                       vocab_size=1000)
            self.bert = AutoModel.from_config(cfg)
        elif pretrained:
            self.bert = AutoModel.from_pretrained(name)
        else:
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(name))
        self.out_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)     # masked mean pooling


# ---------------------------------------------------------------- dual encoder
class DualEncoder(nn.Module):
    def __init__(self, embed_dim=256, text_name="distilbert-base-uncased", tiny=False, **vit_kw):
        super().__init__()
        self.visual = ViT(**vit_kw)
        self.text = TextEncoder(text_name, tiny=tiny)
        self.img_proj = nn.Linear(self.visual.out_dim, embed_dim, bias=False)
        self.txt_proj = nn.Linear(self.text.out_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images):
        return F.normalize(self.img_proj(self.visual(images)), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        return F.normalize(self.txt_proj(self.text(input_ids, attention_mask)), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        return self.encode_image(images), self.encode_text(input_ids, attention_mask), scale


def clip_loss(img, txt, scale, labels=None):
    """Symmetric InfoNCE: image->text and text->image.

    labels (optional, [B] ints): items sharing a label are all positives of each
    other (multi-positive loss). Use for class-label captions (CIFAR) where many
    batch items have identical text. labels=None -> standard CLIP loss."""
    logits = scale * img @ txt.t()
    if labels is None:
        t = torch.arange(img.size(0), device=img.device)
        return (F.cross_entropy(logits, t) + F.cross_entropy(logits.t(), t)) / 2
    pos = (labels[:, None] == labels[None, :]).float()
    pos = pos / pos.sum(1, keepdim=True)                  # rows sum to 1
    l_i2t = -(pos * F.log_softmax(logits, dim=1)).sum(1).mean()
    l_t2i = -(pos * F.log_softmax(logits.t(), dim=1)).sum(1).mean()
    return (l_i2t + l_t2i) / 2


