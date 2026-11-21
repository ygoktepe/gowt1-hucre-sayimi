"""Classical image processing stages for bright fluorescent nuclei."""

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass(frozen=True)
class Config:
    """All algorithm settings for one reproducible experiment."""

    id: str
    family: str
    intensity: str = "original"
    filter: str = "none"
    segmentation: str = "otsu"
    morphology: tuple[str, ...] = ()
    stretch_percentiles: tuple[float, float] = (1.0, 99.5)
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    gamma: float = 0.4
    kernel_size: int = 3
    fixed_threshold: float = 9.0
    adaptive_block: int = 121
    adaptive_c: float = -4.0
    morph_size: int = 5
    fill_holes: bool = False
    min_area: int = 0
    split: bool = False
    peak_distance: int = 24
    peak_min_radius: float = 7.0


@dataclass
class SegmentationResult:
    """Count, integer instance labels and requested diagnostic arrays."""

    count: int
    labels: np.ndarray
    steps: dict[str, np.ndarray]


BASELINE = Config("B00", "baseline")


def build_experiments():
    """Return the classical configurations used in the development comparisons."""
    reference = Config(
        "E01", "fixed", filter="gaussian", kernel_size=5,
        segmentation="fixed", morphology=("opening", "closing"),
        fill_holes=True, min_area=250,
    )
    otsu = replace(reference, id="E02", family="otsu", segmentation="otsu")
    adaptive = replace(reference, id="E03", family="adaptive", segmentation="adaptive")
    return [
        reference, otsu, adaptive,
        # Transformation comparisons use the same Otsu chain.
        replace(otsu, id="E04", intensity="stretch"),
        replace(otsu, id="E05", intensity="equalize"),
        replace(otsu, id="E06", intensity="clahe"),
        replace(otsu, id="E07", intensity="gamma"),
        # Smoothing and sharpening use the same fixed-threshold chain.
        replace(reference, id="E08", filter="none"),
        replace(reference, id="E09", kernel_size=3),
        replace(reference, id="E10", filter="median"),
        replace(reference, id="E11", filter="sharpen", kernel_size=3),
        # Isolated operations and size/order comparisons use elliptical elements.
        replace(reference, id="E12", morphology=()),
        replace(reference, id="E13", morphology=("erosion",)),
        replace(reference, id="E14", morphology=("dilation",)),
        replace(reference, id="E15", morphology=("opening",)),
        replace(reference, id="E16", morphology=("closing",)),
        replace(reference, id="E17", morphology=("closing", "opening")),
        replace(reference, id="E18", morph_size=3),
        replace(reference, id="E19", morph_size=7),
        # Object separation and threshold sensitivity.
        replace(reference, id="E20", split=True),
        replace(reference, id="E21", fixed_threshold=6),
        replace(reference, id="E22", fixed_threshold=12),
        replace(reference, id="E23", fixed_threshold=16),
        replace(adaptive, id="E24", adaptive_c=-2),
        replace(adaptive, id="E25", adaptive_c=-6),
        replace(adaptive, id="E26", adaptive_block=81),
        replace(adaptive, id="E27", split=True),
        replace(otsu, id="E28", intensity="gamma", gamma=0.25),
        replace(reference, id="E29", min_area=125),
        replace(reference, id="E30", min_area=500),
    ]


def _as_uint8(image):
    if image.dtype == np.uint8:
        return image
    scale = 255.0 / np.iinfo(image.dtype).max
    return np.rint(image * scale).astype(np.uint8)


def _intensity(image, config):
    if config.intensity == "original":
        return image
    if config.intensity == "stretch":
        low, high = np.percentile(image, config.stretch_percentiles)
        if high == low:
            return _as_uint8(image)
        return np.rint(np.clip((image - low) * (255.0 / (high - low)), 0, 255)).astype(
            np.uint8
        )
    if config.intensity == "equalize":
        return cv2.equalizeHist(_as_uint8(image))
    if config.intensity == "clahe":
        grid = (config.clahe_grid, config.clahe_grid)
        return cv2.createCLAHE(clipLimit=config.clahe_clip, tileGridSize=grid).apply(
            _as_uint8(image)
        )
    if config.intensity == "gamma":
        lut = np.rint(255 * (np.arange(256) / 255) ** config.gamma).astype(np.uint8)
        return cv2.LUT(_as_uint8(image), lut)
    raise ValueError(f"Bilinmeyen yoğunluk dönüşümü: {config.intensity}")


def _filter(image, config):
    kernel = (config.kernel_size, config.kernel_size)
    if config.filter == "none":
        return image
    if config.filter == "gaussian":
        return cv2.GaussianBlur(image, kernel, sigmaX=0)
    if config.filter == "median":
        return cv2.medianBlur(image, config.kernel_size)
    if config.filter == "sharpen":
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, sharpen_kernel)
    raise ValueError(f"Bilinmeyen filtre: {config.filter}")


def _threshold(image, config):
    if config.segmentation == "fixed":
        limit = config.fixed_threshold * (np.iinfo(image.dtype).max / 255)
        return (image > limit).astype(np.uint8)
    if config.segmentation == "otsu":
        return cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1].astype(
            np.uint8, copy=False
        )
    if config.segmentation == "adaptive":
        return cv2.adaptiveThreshold(
            _as_uint8(image), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            config.adaptive_block, config.adaptive_c,
        )
    raise ValueError(f"Bilinmeyen eşikleme: {config.segmentation}")


def _morphology(mask, config):
    if config.morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_size,) * 2)
        operations = {
            "erosion": cv2.MORPH_ERODE, "dilation": cv2.MORPH_DILATE,
            "opening": cv2.MORPH_OPEN, "closing": cv2.MORPH_CLOSE,
        }
        for operation in config.morphology:
            mask = cv2.morphologyEx(mask, operations[operation], kernel)
    if config.fill_holes:
        mask = ndi.binary_fill_holes(mask).astype(np.uint8)
    return mask


def _label_components(mask, min_area):
    total, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    if len(kept) == total - 1:
        return total - 1, labels
    remap = np.zeros(total, dtype=np.int32)
    remap[kept] = np.arange(1, len(kept) + 1)
    return len(kept), remap[labels]


def _split_touching(labels, config):
    mask = (labels > 0).astype(np.uint8)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    points = peak_local_max(
        distance, min_distance=config.peak_distance, threshold_abs=config.peak_min_radius,
        labels=labels, exclude_border=False, p_norm=2,
    )
    found = set(labels[tuple(points.T)])
    missing = [label for label in range(1, int(labels.max()) + 1) if label not in found]
    if missing:
        # Retain one marker per component even when its distance peak is too low.
        points = np.vstack([points, ndi.maximum_position(distance, labels, index=missing)])
    markers = np.zeros_like(labels)
    markers[tuple(points.T)] = np.arange(1, len(points) + 1)
    labels = watershed(-distance, markers, mask=mask, connectivity=np.ones((3, 3), bool))

    areas = np.bincount(labels.ravel(), minlength=len(points) + 1)
    kept = np.flatnonzero(areas[1:] >= max(1, config.min_area)) + 1
    remap = np.zeros(len(areas), dtype=np.int32)
    remap[kept] = np.arange(1, len(kept) + 1)
    return len(kept), remap[labels], distance, markers


def process(image, config, keep_steps=False):
    """Process one image without loading annotations or consulting other frames."""
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16) or not image.size:
        raise ValueError("Beklenen girdi boş olmayan, tek kanallı uint8 veya uint16 görüntüdür.")
    intensity = _intensity(image, config)
    filtered = _filter(intensity, config)
    threshold = _threshold(filtered, config)
    morphology = _morphology(threshold, config)
    count, labels = _label_components(morphology, config.min_area)
    steps = dict(original=image, intensity=intensity, filtered=filtered,
                 threshold=threshold, morphology=morphology) if keep_steps else {}
    if config.split and count:
        count, labels, distance, markers = _split_touching(labels, config)
        if keep_steps:
            steps.update(distance=distance, markers=markers)
    if keep_steps:
        steps["labels"] = labels
    return SegmentationResult(count, labels, steps)
