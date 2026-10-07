"""2.5-D attention U-Net with a three-class softmax head (needs torch).

This is the only model definition. Training, evaluation and serving build
the network from the same ``UNetConfig``, and a checkpoint stores that config
next to the weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from ..labels import NUM_CLASSES


class UNetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    in_windows: int = Field(2, ge=1)  # CT windows per slice
    context: int = Field(1, ge=0)  # neighbour slices on each side (2.5-D input)
    base: int = Field(16, ge=4)  # channels of the first level
    depth: int = Field(3, ge=2, le=5)
    attention_gates: bool = True
    num_classes: int = NUM_CLASSES

    @property
    def in_channels(self) -> int:
        return self.in_windows * (2 * self.context + 1)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive attention gate: the decoder signal weights the skip features."""

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.w_skip = nn.Conv2d(skip_ch, inter_ch, 1)
        self.w_gate = nn.Conv2d(gate_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.w_skip(skip) + self.w_gate(gate))
        return skip * torch.sigmoid(self.psi(a))


class AttentionUNet(nn.Module):
    def __init__(self, cfg: UNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        chans = [cfg.base * 2**i for i in range(cfg.depth)]
        self.down = nn.ModuleList()
        cin = cfg.in_channels
        for c in chans:
            self.down.append(ConvBlock(cin, c))
            cin = c
        self.bottom = ConvBlock(chans[-1], chans[-1] * 2)
        self.up = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = chans[-1] * 2
        for c in reversed(chans):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.gates.append(AttentionGate(c, c, max(c // 2, 1)) if cfg.attention_gates else nn.Identity())
            self.dec.append(ConvBlock(2 * c, c))
            cin = c
        self.head = nn.Conv2d(chans[0], cfg.num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        m = 2 ** self.cfg.depth
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, gate, dec, skip in zip(self.up, self.gates, self.dec, reversed(skips)):
            x = up(x)
            s = gate(skip, x) if self.cfg.attention_gates else skip
            x = dec(torch.cat([s, x], dim=1))
        return self.head(x)[..., :H, :W]


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Cross-entropy plus mean soft Dice over the foreground classes (liver, tumor)."""
    ce = F.cross_entropy(logits, target)
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return ce + (1 - dice[1:]).mean()


def build(cfg: UNetConfig) -> AttentionUNet:
    return AttentionUNet(cfg)
