"""Image preprocessing shared by training and evaluation (same functions, same interpolation).

Input: uint8 tensor [B, N_cam, H, W, 3]. Output: float32 [B, N_cam, 3, H, W] in [0, 1].
augment=False: identity (train and eval).
augment=True : train = per-sample, per-camera RandomResizedCrop(area 0.9, ratio 1) + ColorJitter;
               eval  = deterministic centre crop of side round(sqrt(0.9) * H) resized back to H.
"""
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

CROP_AREA = 0.9
JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)


def to_float(images_u8: torch.Tensor) -> torch.Tensor:
    assert images_u8.dtype == torch.uint8 and images_u8.shape[-1] == 3, (images_u8.dtype, images_u8.shape)
    return images_u8.permute(0, 1, 4, 2, 3).float() / 255.0


def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def crop_side(h: int) -> int:
    return int(round(math.sqrt(CROP_AREA) * h))


def _crop_resize(img: torch.Tensor, top: int, left: int, side: int, out: int) -> torch.Tensor:
    return _resize(img[None, :, top : top + side, left : left + side], out)[0]


def _color_jitter(img: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    def u(lo, hi):
        return float(lo + (hi - lo) * torch.rand((), generator=g))

    b = u(1 - JITTER["brightness"], 1 + JITTER["brightness"])
    c = u(1 - JITTER["contrast"], 1 + JITTER["contrast"])
    s = u(1 - JITTER["saturation"], 1 + JITTER["saturation"])
    h = u(-JITTER["hue"], JITTER["hue"])
    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    img = TF.adjust_saturation(img, s)
    img = TF.adjust_hue(img, h)
    return img.clamp(0.0, 1.0)


def preprocess(images_u8: torch.Tensor, *, augment: bool, train: bool, generator: torch.Generator | None = None):
    x = to_float(images_u8)
    if not augment:
        return x
    B, N, C, H, W = x.shape
    assert H == W
    side = crop_side(H)
    out = torch.empty_like(x)
    if not train:
        top = left = (H - side) // 2
        for b in range(B):
            for n in range(N):
                out[b, n] = _crop_resize(x[b, n], top, left, side, H)
        return out
    g = generator if generator is not None else torch.default_generator
    for b in range(B):
        for n in range(N):
            top = int(torch.randint(0, H - side + 1, (), generator=g))
            left = int(torch.randint(0, W - side + 1, (), generator=g))
            img = _crop_resize(x[b, n], top, left, side, H)
            out[b, n] = _color_jitter(img, g)
    return out
