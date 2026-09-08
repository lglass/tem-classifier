"""
CryoEM Particle Species Classifier — model architecture
=========================================================
CryoFormerLite: a rotation-aware convolutional network with physics-informed
preprocessing for cryo-EM single-particle classification.

Design rationale
----------------
Problem constraints:
  - Compositional heterogeneity: multiple species present in the dataset.
  - Orientational heterogeneity: particles appear at arbitrary SO(3) poses.
  - CTF distortion: each micrograph (and thus each particle box) is blurred /
    phase-flipped by a contrast transfer function with unknown defocus.
  - Coloured (non-white) detector noise: power spectrum of noise is not flat.

Architecture
--------------------------
1. Learnable lowpass Butterworth filter (LearnableButterworth)
   Applied in Fourier space before any learned convolutions.  Follows the
   skimage.filters.butterworth convention (squared Butterworth, high_pass=False)
   with learnable cutoff r_c and order n; initialised at cutoff_frequency_ratio
   = 12/img_size, order = 1.

2. Steerable / group-equivariant stem (CyclicEquivariantStem)
   Four parallel convolution streams rotated by 0°, 90°, 180°, 270° (C4
   symmetry) whose feature maps are pooled by max-over-group. This gives
   exact C4 invariance cheaply and approximate SO(2) invariance via the
   later global average pooling. For full SO(2) equivariance the stem can be
   replaced by e2cnn (escnn) layers — see the comment in the class.

3. Efficient backbone: ConvNeXt-Tiny blocks
   Depthwise separable 7×7 convolutions with LayerNorm, following the
   ConvNeXt design. This gives a good inductive bias for spatial structure
   while being parameter-efficient relative to a ViT.

4. Radial attention gate (RadialAttentionGate)
   A 1D attention over the radial frequency profile learned from the power
   spectrum. This lets the network up-weight informative frequency bands
   (e.g. around the first zero of the CTF) and down-weight others.

5. Head: linear classifier over global-average-pooled features.

Input / output
--------------
  Input : float32 tensor, shape (B, 1, 128, 128), values in real-space electrons/Å².
  Output: float32 tensor, shape (B, num_classes), unnormalised log-probabilities.

Usage
-----
    model = CryoFormerLite(num_classes=5)
    logits = model(images)               # (B, 5)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1.  Learnable radial bandpass filter (Fourier-domain)
# ---------------------------------------------------------------------------


class LearnableButterworth(nn.Module):
    """
    Learnable lowpass Butterworth filter applied in Fourier space.

    Matches the skimage.filters.butterworth convention (squared Butterworth,
    high_pass=False):

        H(r) = 1 / (1 + (r / r_c)^(2n))

    where r ∈ [0, 1] is the radial frequency normalised to Nyquist=1 and
    r_c = 2 * cutoff_frequency_ratio in that same space (the factor-of-2
    arises because skimage's cutoff_frequency_ratio is defined relative to
    the full sampling frequency, while Nyquist sits at 0.5).

    Both r_c and n are learnable.  Defaults match skimage's high_pass=False,
    order=1, cutoff_frequency_ratio=12/img_size.
    """

    def __init__(self, img_size: int = 128, pixel_size_A: float = 2.0):
        super().__init__()
        self.img_size = img_size

        # cutoff_frequency_ratio = 12 / img_size; r_c = 2 * that in Nyquist-1 space
        r_c_init = 2.0 * (12.0 / img_size)  # 0.1875 for 128×128
        self.log_r_c = nn.Parameter(torch.tensor(math.log(r_c_init)))
        self.log_n = nn.Parameter(torch.tensor(math.log(1.0)))  # order = 1

        self.register_buffer("r", _radial_frequency_grid(img_size))  # (H, W//2+1)

    def _butterworth_response(self) -> torch.Tensor:
        r = self.r  # (H, W//2+1)
        r_c = self.log_r_c.exp().clamp(1e-3, 0.99)
        n = self.log_n.exp().clamp(0.5, 8.0)

        # Squared lowpass Butterworth — matches skimage squared_butterworth=True.
        # Clamp base away from zero to keep autograd safe (0^n → NaN via log).
        t = (r / r_c).clamp(min=1e-8)
        return 1.0 / (1.0 + t.pow(2.0 * n))  # (H, W//2+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W) real-space image
        returns same shape, filtered.
        """
        B, C, H, W = x.shape
        # rfft2 — only positive frequencies needed (Hermitian symmetry)
        X = torch.fft.rfft2(x)  # (B, C, H, W//2+1) complex
        H_filt = self._butterworth_response()  # (H, W//2+1)
        X_filt = X * H_filt.unsqueeze(0).unsqueeze(0)
        return torch.fft.irfft2(X_filt, s=(H, W))  # (B, C, H, W)


def _radial_frequency_grid(img_size: int) -> torch.Tensor:
    """Normalised radial spatial frequency for an rfft2 grid (0 = DC, 1 = Nyquist)."""
    freq_y = torch.fft.fftfreq(img_size)  # (H,)
    freq_x = torch.fft.rfftfreq(img_size)  # (W//2+1,)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
    return torch.sqrt(fy**2 + fx**2) / 0.5  # (H, W//2+1)



# ---------------------------------------------------------------------------
# 2.  C4-equivariant stem
# ---------------------------------------------------------------------------


class CyclicEquivariantStem(nn.Module):
    """
    Approximate rotation-invariant stem using a C4 symmetry group.

    Four parallel 7×7 conv branches process the image at 0°, 90°, 180°, 270°
    (implemented by rotating the *kernel*, which is equivalent to rotating
    the input).  The four feature maps are combined by a max-over-group pool,
    giving exact C4 invariance and approximate full-circle invariance when
    composed with the later global average pool.

    NOTE: For strict SO(2) equivariance, replace this stem with
    e2cnn (escnn) CyclicGroup(N) steerable convolutions — the rest of the
    architecture is compatible.  Install via: pip install escnn
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 7, 7))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

    @staticmethod
    def _rot90_kernel(w: torch.Tensor, k: int) -> torch.Tensor:
        """Rotate kernel by k*90 degrees (last two dims are spatial)."""
        return torch.rot90(w, k, dims=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [
            F.conv2d(x, self._rot90_kernel(self.weight, k), self.bias, padding=3)
            for k in range(4)
        ]
        stacked = torch.stack(outs, dim=1)  # (B, 4, out_channels, H, W)
        pooled = stacked.max(dim=1).values  # (B, out_channels, H, W)
        return self.act(self.norm(pooled))


# ---------------------------------------------------------------------------
# 3.  Radial attention gate
# ---------------------------------------------------------------------------


class RadialAttentionGate(nn.Module):
    """
    Compute a 1D attention weight over the radial power spectrum of the
    feature maps, then modulate each channel by its radial profile score.

    This lets the network suppress frequency bands dominated by CTF zeros or
    noise and amplify discriminative bands.
    """

    def __init__(self, channels: int, img_size: int = 32, n_bins: int = 16):
        super().__init__()
        self.n_bins = n_bins

        r = _radial_frequency_grid(img_size)  # 0..1
        bin_idx = (r * n_bins).long().clamp(0, n_bins - 1)
        self.register_buffer("bin_idx", bin_idx)  # (H, W//2+1)

        # Small MLP: maps radial power spectrum → per-channel scale
        self.attn_mlp = nn.Sequential(
            nn.Linear(n_bins, 64), nn.GELU(), nn.Linear(64, channels), nn.Sigmoid()
        )

    def _radial_power_profile(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample power spectrum, radially binned and normalised: (B, n_bins)."""
        B = x.shape[0]
        # Keep in float32: squared FFT magnitudes easily exceed float16 max
        # (~65504) and would produce inf → NaN after the subsequent division.
        Xf = torch.fft.rfft2(x.float())  # (B, C, H, W//2+1)
        power = Xf.abs().pow(2).real  # (B, C, H, W//2+1)
        flat_power = power.mean(dim=1)  # (B, H, W//2+1), averaged over channels

        idx = self.bin_idx.view(1, -1).expand(B, -1)  # (B, H*(W//2+1))
        profile = torch.zeros(B, self.n_bins, device=x.device, dtype=torch.float32)
        profile.scatter_add_(1, idx, flat_power.view(B, -1))

        counts = torch.zeros(self.n_bins, device=x.device)
        counts.scatter_add_(
            0, self.bin_idx.view(-1), torch.ones(self.bin_idx.numel(), device=x.device)
        )
        profile = profile / (counts.unsqueeze(0) + 1e-8)
        return profile / (profile.max(dim=1, keepdim=True).values + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        profile = self._radial_power_profile(x)  # (B, n_bins)
        scale = self.attn_mlp(profile)  # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# 4.  ConvNeXt-style block
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: depthwise 7×7 conv → LayerNorm → 1×1 expand → GELU → 1×1 contract.
    Layer scale and stochastic depth included for stable deep training.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        expand_ratio: int = 4,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expand_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return residual + self.drop_path(x)


class DropPath(nn.Module):
    """Stochastic depth regularisation."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep) / keep
        return x * mask


# ---------------------------------------------------------------------------
# 5.  Downsampling layer
# ---------------------------------------------------------------------------


class ConvNeXtDownsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.conv(x)


# ---------------------------------------------------------------------------
# 6.  Shared backbone-construction helpers
# ---------------------------------------------------------------------------
#
# Builds the "stem → 4 ConvNeXt stages (with downsampling + a radial
# attention gate after stage 2) → pooled linear head" backbone used by
# CryoFormerLite, parameterised by channel widths and stage depths. The two
# helpers below build it in place on a model instance (rather than as a
# nested submodule) so the class keeps its original flat attribute names —
# and therefore stays compatible with checkpoints trained before this
# refactor.


def _init_weights(module: nn.Module) -> None:
    """Truncated-normal init for conv/linear weights, unit init for norms."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _make_stage(dim: int, depth: int, drop_path_rates) -> nn.Sequential:
    """`depth` stacked ConvNeXtBlocks at a fixed channel width, consuming
    one drop-path rate per block from the `drop_path_rates` iterator."""
    return nn.Sequential(
        *[ConvNeXtBlock(dim, drop_path=next(drop_path_rates)) for _ in range(depth)]
    )


def _attach_convnext_backbone(
    model: nn.Module,
    dims: list[int],
    depths: list[int],
    img_size: int,
    drop_path_rate: float,
) -> None:
    """
    Build the shared stem/stages/downsamples/radial-attention/head and set
    them as attributes directly on `model` (stem, stage1, down1, stage2,
    down2, radial_attn, stage3, down3, stage4, head_norm, head).

    `dims` and `depths` are the four per-stage channel widths and block
    counts, e.g. dims=[64, 128, 256, 512], depths=[2, 2, 6, 2].
    """
    assert len(dims) == len(depths) == 4, "expected 4 stages"

    model.stem = CyclicEquivariantStem(in_channels=1, out_channels=dims[0])

    drop_path_rates = iter(torch.linspace(0, drop_path_rate, sum(depths)).tolist())
    model.stage1 = _make_stage(dims[0], depths[0], drop_path_rates)
    model.down1 = ConvNeXtDownsample(dims[0], dims[1])

    model.stage2 = _make_stage(dims[1], depths[1], drop_path_rates)
    model.down2 = ConvNeXtDownsample(dims[1], dims[2])

    model.radial_attn = RadialAttentionGate(dims[2], img_size=img_size // 4, n_bins=16)

    model.stage3 = _make_stage(dims[2], depths[2], drop_path_rates)
    model.down3 = ConvNeXtDownsample(dims[2], dims[3])

    model.stage4 = _make_stage(dims[3], depths[3], drop_path_rates)

    model.head_norm = nn.LayerNorm(dims[3])
    model.head = nn.Linear(dims[3], model.num_classes)


def _run_convnext_backbone(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the attributes `_attach_convnext_backbone` set up."""
    x = model.stem(x)
    x = model.stage1(x)
    x = model.down1(x)
    x = model.stage2(x)
    x = model.down2(x)
    x = model.radial_attn(x)
    x = model.stage3(x)
    x = model.down3(x)
    x = model.stage4(x)
    x = x.mean(dim=(-2, -1))
    x = model.head_norm(x)
    return model.head(x)


# ---------------------------------------------------------------------------
# 7.  CryoFormerLite
# ---------------------------------------------------------------------------


class CryoFormerLite(nn.Module):
    """
    CryoFormerLite: lightweight, rotation-aware convolutional classifier for cryo-EM particles.

    Architecture summary
    --------------------
    Input (B,1,128,128)
        │
        ├── LearnableButterworth        [Fourier-domain noise filter]
        │
        ├── CyclicEquivariantStem    [C4-invariant 7×7 conv → 32 ch, same size]
        │
        ├── Stage 1: 2× ConvNeXtBlock (32 ch, 128×128)
        │     └─ Downsample → (64 ch, 64×64)
        │
        ├── Stage 2: 2× ConvNeXtBlock (64 ch, 64×64)
        │     └─ Downsample → (128 ch, 32×32)
        │
        ├── RadialAttentionGate      [frequency-band gating at 32×32]
        │
        ├── Stage 3: 4× ConvNeXtBlock (128 ch, 32×32)
        │     └─ Downsample → (256 ch, 16×16)
        │
        ├── Stage 4: 2× ConvNeXtBlock (256 ch, 16×16)
        │
        ├── Global Average Pool      → (B, 256)
        │
        └── LayerNorm → Linear(256, num_classes)

    Total parameters: ~2 M (comparable to a ResNet-50).

    Args:
        num_classes  : number of particle species.
        img_size     : image side length in pixels (default 128).
        pixel_size_A : Å per pixel (default 2.0).
        drop_path_rate: stochastic depth rate (default 0.1).
    """

    _DIMS = [32, 64, 128, 256]
    _DEPTHS = [2, 2, 4, 2]

    def __init__(
        self,
        num_classes: int,
        img_size: int = 128,
        pixel_size_A: float = 2.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.bandpass = LearnableButterworth(img_size, pixel_size_A)

        _attach_convnext_backbone(
            self, self._DIMS, self._DEPTHS, img_size, drop_path_rate
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bandpass(x)
        return _run_convnext_backbone(self, x)
