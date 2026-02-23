from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage import filters, io, transform
from skimage.color import rgb2gray

from .onnxruntime_compat import InferenceSession
from .utils import DEFAULT_SIZE


def load_normalized(
    path_or_bytes: Path | bytes, size: int = DEFAULT_SIZE
) -> NDArray[np.float32]:
    """
    Load an image and normalize to [0, 1] range, resized to target size.

    If image has a varying alpha channel, use that. Otherwise convert to grayscale.
    Prints a warning if both alpha and intensity vary significantly.
    """
    if isinstance(path_or_bytes, bytes):
        img_raw = Image.open(BytesIO(path_or_bytes))
    else:
        img_raw = io.imread(path_or_bytes)

    img_raw = np.array(img_raw)

    # Handle images with alpha channel (2 or 4 channels)
    if img_raw.ndim == 3 and img_raw.shape[2] in (2, 4):
        alpha = img_raw[:, :, -1] / 255.0
        color_channels = img_raw[:, :, :-1]
        brightness = (
            color_channels[:, :, 0] / 255.0
            if color_channels.shape[2] == 1
            else rgb2gray(color_channels / 255.0)
        )

        # Combine: transparent→0, white opaque→low, black opaque→high
    else:
        alpha = 1
        brightness = rgb2gray(img_raw) if img_raw.ndim == 3 else img_raw

    img = alpha * (1.0 - brightness)
    # Resize based on smallest dimension
    scale = size / min(img.shape)
    img = transform.rescale(img, scale, anti_aliasing=True)

    # Normalize to [0, 1]
    img = img - img.min()
    img = img / img.max() if img.max() > 0 else img

    return img.astype(np.float32)


def compute_thickness_map(image: NDArray[np.bool]) -> NDArray[np.float32]:
    """
    Compute a thickness map for the given binary image using iterative erosion.

    The thickness at each pixel represents how many erosion iterations are needed
    to remove that pixel (i.e., the distance to the edge of the stroke).

    Args:
        image: Binary image where True = sketch pixels, False = background

    Returns:
        Thickness map where each value indicates local stroke thickness
    """
    from skimage.morphology import erosion

    thicknesses = np.array(image, dtype=int)
    eroding = image.copy()
    i = 2

    while np.sum(eroding) > 0:
        eroded = erosion(eroding)
        diff = eroding ^ eroded  # Pixels removed in this iteration
        thicknesses[diff] = i
        eroding = eroded
        i += 1

    return thicknesses.astype(np.float32) / 2.0


class BinarySketchPredictor:
    def __init__(
        self,
        threshold: float = 0.5,
        model_path: Optional[Path] = None,
        gaussian_blur_sigma: Optional[float] = 1,
    ):
        # Initialize any necessary parameters or model components here
        self.threshold = threshold
        self.model_path = model_path
        self.gaussian_blur_sigma = gaussian_blur_sigma
        if model_path is None:
            model_path = Path(__file__).parent / "cnn/model.onnx"
        self.inference = InferenceSession(model_path)

    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32]):
        raise ValueError(
            "This model is not trainable. It is a pre-trained model that can only be used for prediction."
        )

    async def predict(self, X: NDArray[np.float32]) -> NDArray[np.bool]:
        """
        Predict a binary mask indicating which pixels are part of the sketch.
        Args:
            X (NDArray[float]): The input image as a 2D array of floats. Each element must be in the range [0, 1].
        Returns:
            NDArray[bool]: A 2D array of booleans where True indicates that the pixel is predicted to be part of the sketch, and False indicates background.
        """
        proba = await self.predict_proba(X)
        if self.gaussian_blur_sigma is not None:
            proba = filters.gaussian(
                proba, sigma=self.gaussian_blur_sigma, preserve_range=True
            )
        return proba > self.threshold

    async def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Predict the probability of each pixel being part of the sketch.
        Args:
            X (NDArray[float]): The input image as a 2D array of floats. Each element must be in the range [0, 1].
        Returns:
            NDArray[float]: A 2D array of floats representing the predicted probabilities for each pixel.
        """
        X_with_color = X[np.newaxis, :, :]
        result = await self.inference.run(["output"], {"input": X_with_color})
        return result[0][0]
