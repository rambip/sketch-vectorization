from skimage import io
from skimage.transform import rescale
from skimage.color import rgb2gray
import numpy as np


def load_normalized(path):
    """
    load the image with values between 0 and 1, and with smallest dimension equal to 512 pixels
    """
    img_raw = io.imread(path)

    if len(img_raw.shape) == 2:
        img = img_raw
    elif img_raw.shape[2] == 2:
        img = img_raw[:, :, 0]
    elif img_raw.shape[2] == 3:
        img = (rgb2gray(img_raw) * 255).astype(int)
    elif img_raw.shape[2] == 4:
        img = (rgb2gray(img_raw[:, :, :3]) * (img_raw[:, :, 3])).astype(int)
    else:
        raise ValueError(f"invalid image shape: {img_raw.shape}")

    size = min(img.shape[0], img.shape[1])
    img = 1 - rescale(img, 512 / size)
    img = img - np.min(img)
    img = img / np.max(img)
    img = img.astype(np.float32)
    return img
