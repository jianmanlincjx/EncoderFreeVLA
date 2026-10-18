"""Image pipeline aligned with MolmoAct2 (both arms see exactly the same pixels).

Training augmentation (per camera, independently, on the 256x256 frame, float CHW in [0, 1]):
  LeRobot ImageTransforms with the settings of the user's MolmoAct2 LIBERO run
  (release_ckpt/molmoact2/baseline/train_config.json -> dataset.image_transforms):
  pick 3 of 6 transforms by weight (no repetition), applied in config order:
  brightness [0.8,1.2], contrast [0.8,1.2], saturation [0.5,1.5], hue [-0.05,0.05],
  sharpness [0.5,1.5] (SharpnessJitter), RandomAffine(degrees +-5, translate 5%).
Model input (train and eval), as MolmoAct2ImageProcessor with crop_mode="resize" + the model's on-device
canonicalisation (Checkpoint/MolmoAct2/image_processing_molmoact2.py resize_image / normalize_image and
modeling_molmoact2.py MolmoAct2VisionBackbone.forward):
  resize to 378x378 (torchvision Resize, BILINEAR, antialias=False, no aspect preservation),
  values on the uint8 grid, then x*2-1 in [-1, 1].
  uint8 input  -> resize on uint8 with torchvision v1 (bit-exact with MolmoAct2's processor on CPU)
  float input  -> resize on float, clip [0,1], round to the uint8 grid (MolmoAct2's canonicalisation)
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF  # v1 functional, as torchvision.transforms.Resize in MolmoAct2
from torchvision.transforms import InterpolationMode

MODEL_IMG_SIZE = 378

MOLMOACT2_LIBERO_TFS = {
    "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
    "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
    "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
    "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
    "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
    "affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}},
}


def make_train_augment():
    """LeRobot ImageTransforms(enable, max_num_transforms=3, random_order=False, tfs as above)."""
    from lerobot.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig

    tfs = {k: ImageTransformConfig(weight=v["weight"], type=v["type"], kwargs=v["kwargs"])
           for k, v in MOLMOACT2_LIBERO_TFS.items()}
    return ImageTransforms(ImageTransformsConfig(enable=True, max_num_transforms=3, random_order=False, tfs=tfs))


def augment_frames(frames_u8: np.ndarray, aug) -> torch.Tensor:
    """frames_u8 [N_cam, H, W, 3] uint8 -> float [N_cam, 3, H, W] in [0, 1], each camera augmented independently."""
    x = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float() / 255.0
    return torch.stack([aug(x[i]) for i in range(x.shape[0])], 0)


def preprocess(images: torch.Tensor, size: int = MODEL_IMG_SIZE) -> torch.Tensor:
    """images: uint8 [B, N, H, W, 3] (raw) or float [B, N, 3, H, W] in [0, 1] (augmented)
    -> float32 [B, N, 3, size, size] in [-1, 1]."""
    if images.dtype == torch.uint8:
        assert images.shape[-1] == 3, images.shape
        B, N = images.shape[:2]
        x = images.permute(0, 1, 4, 2, 3).reshape(B * N, 3, *images.shape[2:4])
        x = TF.resize(x, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=False)
        x = x.float() / 255.0
    else:
        assert images.dim() == 5 and images.shape[2] == 3, images.shape
        B, N = images.shape[:2]
        x = images.reshape(B * N, *images.shape[2:]).float()
        x = TF.resize(x, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=False)
        x = torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0
    return (x * 2.0 - 1.0).view(B, N, 3, size, size)
