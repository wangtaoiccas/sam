import os
from typing import Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may not be present in some environments
    torch = None  # type: ignore


class SegmenterInterface:
    """Abstract interface for a point-based segmenter."""

    def segment(self, image_bgr: np.ndarray, point_xy: Tuple[int, int]) -> np.ndarray:
        """Return a boolean mask for the object containing the clicked point.
        image_bgr: HxWx3 BGR image (OpenCV convention)
        point_xy: (x, y) in image pixel coordinates
        """
        raise NotImplementedError


class Sam2Segmenter(SegmenterInterface):
    """SAM2 segmenter wrapper. Attempts to load the checkpoint and run point prompts.

    Note: This requires the SAM2 Python package installed from source and a compatible
    checkpoint, e.g. `/home/tao/sam2.1_hiera_base_plus.pt`.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self._predictor = None
        self._device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self._init_model()

    def _init_model(self) -> None:
        try:
            # Placeholder import path. Actual API may differ depending on SAM2 release.
            # The code tries a few common entry points and fails gracefully.
            try:
                from sam2.build_sam2 import build_sam2  # type: ignore
                from sam2.sam2_image_predictor import Sam2ImagePredictor  # type: ignore
                model = build_sam2(self.checkpoint_path)
                self._predictor = Sam2ImagePredictor(model)
            except Exception:
                # Some forks or future names
                from sam2 import Sam2Predictor  # type: ignore
                self._predictor = Sam2Predictor(self.checkpoint_path, device=self._device)
        except Exception as exc:  # pragma: no cover
            self._predictor = None
            self._init_error = str(exc)
        else:
            self._init_error = None

    def is_ready(self) -> bool:
        return self._predictor is not None

    def get_error(self) -> Optional[str]:
        return getattr(self, "_init_error", None)

    def segment(self, image_bgr: np.ndarray, point_xy: Tuple[int, int]) -> np.ndarray:
        if self._predictor is None:
            raise RuntimeError(self._init_error or "SAM2 predictor not initialized")

        # Convert BGR->RGB as most vision models expect RGB
        image_rgb = image_bgr[:, :, ::-1]

        # Try a few common predictor APIs
        x, y = point_xy
        try:
            # API style 1: set image then predict
            self._predictor.set_image(image_rgb)
            input_point = np.array([[x, y]])
            input_label = np.array([1])  # 1 = foreground
            result = self._predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            if isinstance(result, dict) and "masks" in result:
                mask = result["masks"]
            else:
                mask = result[0]
        except Exception:
            # API style 2: direct call
            result = self._predictor(image_rgb, (x, y))
            mask = result[0]

        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.uint8)
        return mask


class FloodFillFallbackSegmenter(SegmenterInterface):
    """Simple color-based flood fill fallback when SAM2 isn't available.

    Not intended to be perfect; provides a usable demo until SAM2 is installed.
    """

    def __init__(self, tolerance: int = 20) -> None:
        self.tolerance = int(max(0, tolerance))

    def segment(self, image_bgr: np.ndarray, point_xy: Tuple[int, int]) -> np.ndarray:
        import cv2

        h, w = image_bgr.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # OpenCV floodFill requires +2
        x, y = point_xy
        seed_color = image_bgr[y, x].astype(int)
        lo = (self.tolerance, self.tolerance, self.tolerance)
        hi = (self.tolerance, self.tolerance, self.tolerance)
        _img = image_bgr.copy()
        cv2.floodFill(_img, mask, (x, y), newVal=(0, 0, 255), loDiff=lo, upDiff=hi, flags=4)
        # mask returned by floodFill marks filled area as 1 in mask[1:-1,1:-1] with value 1 or 255
        filled = (mask[1:-1, 1:-1] > 0).astype(np.uint8)
        return filled


class ModelWrapper:
    """Decides whether to use SAM2 or a fallback segmenter."""

    def __init__(self, checkpoint_path: str = "/home/tao/sam2.1_hiera_base_plus.pt") -> None:
        self.checkpoint_path = checkpoint_path
        self.segmenter: SegmenterInterface

        use_sam2 = False
        if os.path.exists(self.checkpoint_path):
            try:
                self.segmenter = Sam2Segmenter(self.checkpoint_path)
                use_sam2 = self.segmenter.is_ready()
            except Exception:
                use_sam2 = False
        if not use_sam2:
            self.segmenter = FloodFillFallbackSegmenter(tolerance=18)

    def is_sam2(self) -> bool:
        return isinstance(self.segmenter, Sam2Segmenter) and getattr(self.segmenter, "_predictor", None) is not None

    def get_backend_info(self) -> str:
        if self.is_sam2():
            return "SAM2"
        return "FloodFill"

    def segment(self, image_bgr: np.ndarray, point_xy: Tuple[int, int]) -> np.ndarray:
        return self.segmenter.segment(image_bgr, point_xy)