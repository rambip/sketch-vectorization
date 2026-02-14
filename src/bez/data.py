import numpy as np
from skimage import io, transform
from skimage.color import rgb2gray

D = 128


def load_rgb(path, only_alpha=True):
    img_raw = io.imread(path).astype(np.float32)

    if len(img_raw.shape) == 2:
        return img_raw

    n_channels = img_raw.shape[2]

    if only_alpha:
        if n_channels == 2:
            return img_raw[:, :, 1]
        elif n_channels == 4:
            return img_raw[:, :, 3]
        else:
            raise ValueError("this image has no alpha channel")

    if n_channels == 2:
        return img_raw[:, :, 0] * img_raw[:, :, 1] / 255.0
    elif n_channels == 3:
        return rgb2gray(img_raw)
    elif n_channels == 4:
        return rgb2gray(img_raw[:, :, :3]) * img_raw[:, :, 3]
    else:
        raise ValueError(f"invalid image shape: {img_raw.shape}")


def load_normalized(path, only_alpha=False, d=D):
    """
    load the image with values between 0 and 1
    """
    img = load_rgb(path, only_alpha)
    size = min(img.shape[0], img.shape[1])
    img = transform.rescale(img, d / size, anti_aliasing=True)
    if not only_alpha:
        img = 1 - img
    img = img - np.min(img)
    img = img / np.max(img)
    img = img.astype(np.float32)
    return img
