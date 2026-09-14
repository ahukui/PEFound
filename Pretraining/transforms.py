"""
Local transforms extracted from MONAI for standalone use.
No dependency on monai package required.
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Sequence, Dict, Any
from abc import ABC, abstractmethod


# =============================================================================
# BASE CLASSES
# =============================================================================

class Transform(ABC):
    """Base class for all transforms."""
    
    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError


class MapTransform(Transform):
    """Base class for dictionary-based transforms."""
    
    def __init__(self, keys: Union[str, Sequence[str]], allow_missing_keys: bool = False):
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.allow_missing_keys = allow_missing_keys
    
    def key_iterator(self, data: Dict) -> Tuple[str, Any]:
        for key in self.keys:
            if key in data:
                yield key, data[key]
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in data")


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


# =============================================================================
# INTENSITY TRANSFORMS
# =============================================================================

class ScaleIntensity(Transform):
    """Scale intensity values to a given range."""
    
    def __init__(self, minv: float = 0.0, maxv: float = 1.0):
        self.minv = minv
        self.maxv = maxv
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min == 0:
            return np.full_like(img, self.minv)
        img = (img - img_min) / (img_max - img_min)
        return img * (self.maxv - self.minv) + self.minv


class ScaleIntensityd(MapTransform):
    """Dictionary-based wrapper for ScaleIntensity."""
    
    def __init__(self, keys: Union[str, Sequence[str]], minv: float = 0.0, maxv: float = 1.0):
        super().__init__(keys)
        self.transform = ScaleIntensity(minv, maxv)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class ScaleIntensityRange(Transform):
    """Scale intensity from [a_min, a_max] to [b_min, b_max]."""
    
    def __init__(self, a_min: float, a_max: float, b_min: float = 0.0, b_max: float = 1.0, clip: bool = False):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = (img - self.a_min) / (self.a_max - self.a_min)
        img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)
        return img


class ScaleIntensityRanged(MapTransform):
    """Dictionary-based wrapper for ScaleIntensityRange."""
    
    def __init__(self, keys: Union[str, Sequence[str]], a_min: float, a_max: float, 
                 b_min: float = 0.0, b_max: float = 1.0, clip: bool = False):
        super().__init__(keys)
        self.transform = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class NormalizeIntensity(Transform):
    """Normalize intensity using mean and std."""
    
    def __init__(self, subtrahend: Optional[float] = None, divisor: Optional[float] = None,
                 nonzero: bool = False, channel_wise: bool = False):
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.channel_wise:
            # Assume channel is first dimension
            for i in range(img.shape[0]):
                img[i] = self._normalize(img[i])
        else:
            img = self._normalize(img)
        return img
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        if self.nonzero:
            mask = img != 0
            if mask.any():
                mean = img[mask].mean() if self.subtrahend is None else self.subtrahend
                std = img[mask].std() if self.divisor is None else self.divisor
                img = img.copy()
                img[mask] = (img[mask] - mean) / (std + 1e-8)
        else:
            mean = img.mean() if self.subtrahend is None else self.subtrahend
            std = img.std() if self.divisor is None else self.divisor
            img = (img - mean) / (std + 1e-8)
        return img


class NormalizeIntensityd(MapTransform):
    """Dictionary-based wrapper for NormalizeIntensity."""
    
    def __init__(self, keys: Union[str, Sequence[str]], subtrahend: Optional[float] = None,
                 divisor: Optional[float] = None, nonzero: bool = False, channel_wise: bool = False):
        super().__init__(keys)
        self.transform = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class ThresholdIntensity(Transform):
    """Threshold intensity values."""
    
    def __init__(self, threshold: float, above: bool = True, cval: float = 0.0):
        self.threshold = threshold
        self.above = above
        self.cval = cval
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        if self.above:
            img[img > self.threshold] = self.cval
        else:
            img[img < self.threshold] = self.cval
        return img


class ThresholdIntensityd(MapTransform):
    """Dictionary-based wrapper for ThresholdIntensity."""
    
    def __init__(self, keys: Union[str, Sequence[str]], threshold: float, 
                 above: bool = True, cval: float = 0.0):
        super().__init__(keys)
        self.transform = ThresholdIntensity(threshold, above, cval)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


# =============================================================================
# SPATIAL TRANSFORMS
# =============================================================================

class Resize(Transform):
    """Resize image to given spatial size."""
    
    def __init__(self, spatial_size: Union[int, Sequence[int]], mode: str = "nearest"):
        self.spatial_size = spatial_size if isinstance(spatial_size, (list, tuple)) else [spatial_size]
        self.mode = mode
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError("scipy is required for Resize transform")
        
        # Calculate zoom factors
        spatial_dims = img.ndim
        if img.ndim > len(self.spatial_size):
            # Assume first dim is channel
            spatial_dims = img.ndim - 1
            zoom_factors = [1.0]  # Keep channel dimension
            for i, (orig, new) in enumerate(zip(img.shape[1:], self.spatial_size)):
                zoom_factors.append(new / orig)
        else:
            zoom_factors = [new / orig for orig, new in zip(img.shape, self.spatial_size)]
        
        order = 0 if self.mode == "nearest" else 1
        return zoom(img, zoom_factors, order=order)


class Resized(MapTransform):
    """Dictionary-based wrapper for Resize."""
    
    def __init__(self, keys: Union[str, Sequence[str]], spatial_size: Union[int, Sequence[int]], 
                 mode: str = "nearest"):
        super().__init__(keys)
        self.transform = Resize(spatial_size, mode)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class RandFlip(Transform):
    """Randomly flip image along axes."""
    
    def __init__(self, prob: float = 0.5, spatial_axis: Optional[Union[int, Sequence[int]]] = None):
        self.prob = prob
        self.spatial_axis = spatial_axis
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            axes = self.spatial_axis
            if axes is None:
                axes = list(range(img.ndim))
            elif isinstance(axes, int):
                axes = [axes]
            
            for axis in axes:
                if np.random.random() < 0.5:
                    img = np.flip(img, axis=axis)
        return np.ascontiguousarray(img)


class RandFlipd(MapTransform):
    """Dictionary-based wrapper for RandFlip."""
    
    def __init__(self, keys: Union[str, Sequence[str]], prob: float = 0.5,
                 spatial_axis: Optional[Union[int, Sequence[int]]] = None):
        super().__init__(keys)
        self.prob = prob
        self.spatial_axis = spatial_axis
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        do_flip = np.random.random() < self.prob
        
        if do_flip:
            for key, val in self.key_iterator(d):
                axes = self.spatial_axis
                if axes is None:
                    axes = list(range(val.ndim))
                elif isinstance(axes, int):
                    axes = [axes]
                
                for axis in axes:
                    val = np.flip(val, axis=axis)
                d[key] = np.ascontiguousarray(val)
        return d


class RandRotate90(Transform):
    """Randomly rotate image by 90 degrees."""
    
    def __init__(self, prob: float = 0.5, max_k: int = 3, spatial_axes: Tuple[int, int] = (0, 1)):
        self.prob = prob
        self.max_k = max_k
        self.spatial_axes = spatial_axes
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            k = np.random.randint(1, self.max_k + 1)
            img = np.rot90(img, k=k, axes=self.spatial_axes)
        return np.ascontiguousarray(img)


class RandRotate90d(MapTransform):
    """Dictionary-based wrapper for RandRotate90."""
    
    def __init__(self, keys: Union[str, Sequence[str]], prob: float = 0.5, 
                 max_k: int = 3, spatial_axes: Tuple[int, int] = (0, 1)):
        super().__init__(keys)
        self.prob = prob
        self.max_k = max_k
        self.spatial_axes = spatial_axes
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        
        if np.random.random() < self.prob:
            k = np.random.randint(1, self.max_k + 1)
            for key, val in self.key_iterator(d):
                d[key] = np.ascontiguousarray(np.rot90(val, k=k, axes=self.spatial_axes))
        return d


class CenterSpatialCrop(Transform):
    """Crop center region of image."""
    
    def __init__(self, roi_size: Union[int, Sequence[int]]):
        self.roi_size = roi_size if isinstance(roi_size, (list, tuple)) else [roi_size]
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        slices = []
        for i, (img_dim, roi_dim) in enumerate(zip(img.shape, self.roi_size)):
            if roi_dim <= 0 or roi_dim > img_dim:
                slices.append(slice(None))
            else:
                start = (img_dim - roi_dim) // 2
                slices.append(slice(start, start + roi_dim))
        return img[tuple(slices)]


class CenterSpatialCropd(MapTransform):
    """Dictionary-based wrapper for CenterSpatialCrop."""
    
    def __init__(self, keys: Union[str, Sequence[str]], roi_size: Union[int, Sequence[int]]):
        super().__init__(keys)
        self.transform = CenterSpatialCrop(roi_size)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class RandSpatialCrop(Transform):
    """Randomly crop a region from image."""
    
    def __init__(self, roi_size: Union[int, Sequence[int]], random_size: bool = False):
        self.roi_size = roi_size if isinstance(roi_size, (list, tuple)) else [roi_size]
        self.random_size = random_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        slices = []
        for i, (img_dim, roi_dim) in enumerate(zip(img.shape, self.roi_size)):
            if roi_dim <= 0 or roi_dim > img_dim:
                slices.append(slice(None))
            else:
                max_start = img_dim - roi_dim
                start = np.random.randint(0, max_start + 1)
                slices.append(slice(start, start + roi_dim))
        return img[tuple(slices)]


class RandSpatialCropd(MapTransform):
    """Dictionary-based wrapper for RandSpatialCrop."""
    
    def __init__(self, keys: Union[str, Sequence[str]], roi_size: Union[int, Sequence[int]], 
                 random_size: bool = False):
        super().__init__(keys)
        self.roi_size = roi_size if isinstance(roi_size, (list, tuple)) else [roi_size]
        self.random_size = random_size
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        
        # Get first image to determine crop location
        first_key = self.keys[0]
        img = d[first_key]
        
        # Calculate crop slices (same for all keys)
        slices = []
        for i, (img_dim, roi_dim) in enumerate(zip(img.shape, self.roi_size)):
            if roi_dim <= 0 or roi_dim > img_dim:
                slices.append(slice(None))
            else:
                max_start = img_dim - roi_dim
                start = np.random.randint(0, max_start + 1)
                slices.append(slice(start, start + roi_dim))
        
        for key, val in self.key_iterator(d):
            d[key] = val[tuple(slices)]
        return d


class SpatialPad(Transform):
    """Pad image to given size."""
    
    def __init__(self, spatial_size: Union[int, Sequence[int]], mode: str = "constant", value: float = 0):
        self.spatial_size = spatial_size if isinstance(spatial_size, (list, tuple)) else [spatial_size]
        self.mode = mode
        self.value = value
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        pad_width = []
        for img_dim, target_dim in zip(img.shape, self.spatial_size):
            if target_dim > img_dim:
                diff = target_dim - img_dim
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))
        
        return np.pad(img, pad_width, mode=self.mode, constant_values=self.value)


class SpatialPadd(MapTransform):
    """Dictionary-based wrapper for SpatialPad."""
    
    def __init__(self, keys: Union[str, Sequence[str]], spatial_size: Union[int, Sequence[int]],
                 mode: str = "constant", value: float = 0):
        super().__init__(keys)
        self.transform = SpatialPad(spatial_size, mode, value)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


# =============================================================================
# UTILITY TRANSFORMS
# =============================================================================

class ToTensor(Transform):
    """Convert numpy array to torch tensor."""
    
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, img: np.ndarray):
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for ToTensor transform")
        
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if self.dtype is not None:
            tensor = tensor.to(self.dtype)
        return tensor


class ToTensord(MapTransform):
    """Dictionary-based wrapper for ToTensor."""
    
    def __init__(self, keys: Union[str, Sequence[str]], dtype=None):
        super().__init__(keys)
        self.transform = ToTensor(dtype)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class EnsureChannelFirst(Transform):
    """Ensure channel is the first dimension."""
    
    def __init__(self, channel_dim: int = -1):
        self.channel_dim = channel_dim
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.channel_dim != 0:
            img = np.moveaxis(img, self.channel_dim, 0)
        return img


class EnsureChannelFirstd(MapTransform):
    """Dictionary-based wrapper for EnsureChannelFirst."""
    
    def __init__(self, keys: Union[str, Sequence[str]], channel_dim: int = -1):
        super().__init__(keys)
        self.transform = EnsureChannelFirst(channel_dim)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class AddChannel(Transform):
    """Add a channel dimension."""
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img[np.newaxis, ...]


class AddChanneld(MapTransform):
    """Dictionary-based wrapper for AddChannel."""
    
    def __init__(self, keys: Union[str, Sequence[str]]):
        super().__init__(keys)
        self.transform = AddChannel()
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class SqueezeDim(Transform):
    """Squeeze a dimension."""
    
    def __init__(self, dim: int = 0):
        self.dim = dim
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.squeeze(img, axis=self.dim)


class SqueezeDimd(MapTransform):
    """Dictionary-based wrapper for SqueezeDim."""
    
    def __init__(self, keys: Union[str, Sequence[str]], dim: int = 0):
        super().__init__(keys)
        self.transform = SqueezeDim(dim)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class CastToType(Transform):
    """Cast image to given numpy dtype."""
    
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(self.dtype)


class CastToTyped(MapTransform):
    """Dictionary-based wrapper for CastToType."""
    
    def __init__(self, keys: Union[str, Sequence[str]], dtype=np.float32):
        super().__init__(keys)
        self.transform = CastToType(dtype)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


# =============================================================================
# AUGMENTATION TRANSFORMS
# =============================================================================

class RandGaussianNoise(Transform):
    """Add random Gaussian noise."""
    
    def __init__(self, prob: float = 0.5, mean: float = 0.0, std: float = 0.1):
        self.prob = prob
        self.mean = mean
        self.std = std
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            noise = np.random.normal(self.mean, self.std, img.shape)
            img = img + noise
        return img.astype(img.dtype)


class RandGaussianNoised(MapTransform):
    """Dictionary-based wrapper for RandGaussianNoise."""
    
    def __init__(self, keys: Union[str, Sequence[str]], prob: float = 0.5,
                 mean: float = 0.0, std: float = 0.1):
        super().__init__(keys)
        self.transform = RandGaussianNoise(prob, mean, std)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class RandScaleIntensity(Transform):
    """Randomly scale intensity."""
    
    def __init__(self, factors: Union[float, Tuple[float, float]], prob: float = 0.5):
        self.factors = factors if isinstance(factors, tuple) else (-factors, factors)
        self.prob = prob
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            factor = np.random.uniform(self.factors[0], self.factors[1])
            img = img * (1 + factor)
        return img


class RandScaleIntensityd(MapTransform):
    """Dictionary-based wrapper for RandScaleIntensity."""
    
    def __init__(self, keys: Union[str, Sequence[str]], factors: Union[float, Tuple[float, float]],
                 prob: float = 0.5):
        super().__init__(keys)
        self.transform = RandScaleIntensity(factors, prob)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


class RandShiftIntensity(Transform):
    """Randomly shift intensity."""
    
    def __init__(self, offsets: Union[float, Tuple[float, float]], prob: float = 0.5):
        self.offsets = offsets if isinstance(offsets, tuple) else (-offsets, offsets)
        self.prob = prob
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            offset = np.random.uniform(self.offsets[0], self.offsets[1])
            img = img + offset
        return img


class RandShiftIntensityd(MapTransform):
    """Dictionary-based wrapper for RandShiftIntensity."""
    
    def __init__(self, keys: Union[str, Sequence[str]], offsets: Union[float, Tuple[float, float]],
                 prob: float = 0.5):
        super().__init__(keys)
        self.transform = RandShiftIntensity(offsets, prob)
    
    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key, val in self.key_iterator(d):
            d[key] = self.transform(val)
        return d


# =============================================================================
# CONVENIENCE ALIASES (matching MONAI naming)
# =============================================================================

# Dictionary versions with 'd' suffix
ScaleIntensityD = ScaleIntensityDict = ScaleIntensityd
ScaleIntensityRangeD = ScaleIntensityRangeDict = ScaleIntensityRanged
NormalizeIntensityD = NormalizeIntensityDict = NormalizeIntensityd
ThresholdIntensityD = ThresholdIntensityDict = ThresholdIntensityd
ResizeD = ResizeDict = Resized
RandFlipD = RandFlipDict = RandFlipd
RandRotate90D = RandRotate90Dict = RandRotate90d
CenterSpatialCropD = CenterSpatialCropDict = CenterSpatialCropd
RandSpatialCropD = RandSpatialCropDict = RandSpatialCropd
SpatialPadD = SpatialPadDict = SpatialPadd
ToTensorD = ToTensorDict = ToTensord
EnsureChannelFirstD = EnsureChannelFirstDict = EnsureChannelFirstd
AddChannelD = AddChannelDict = AddChanneld
SqueezeDimD = SqueezeDimDict = SqueezeDimd
CastToTypeD = CastToTypeDict = CastToTyped
RandGaussianNoiseD = RandGaussianNoiseDict = RandGaussianNoised
RandScaleIntensityD = RandScaleIntensityDict = RandScaleIntensityd
RandShiftIntensityD = RandShiftIntensityDict = RandShiftIntensityd


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Create a transform pipeline
    train_transforms = Compose([
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])
    
    # Test with dummy data
    data = {
        "image": np.random.randn(64, 64, 64).astype(np.float32),
        "label": np.random.randint(0, 2, (64, 64, 64)).astype(np.float32),
    }
    
    result = train_transforms(data)
    print(f"Image shape: {result['image'].shape}")
    print(f"Label shape: {result['label'].shape}")
    print("✓ Transforms work correctly!")