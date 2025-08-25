from scipy import ndimage as ndi

import numpy as np
from skimage.morphology import (
    dilation,
    erosion,
    remove_small_holes,
    remove_small_objects,
)
from skimage.morphology import disk
from skimage import transform
from skimage.filters import gaussian, threshold_local, threshold_otsu
from skimage.color import rgb2gray

# experimental code (not used)
"""
def drawing_size(img):
    fft_img = np.fft.fft2(img)
    power_spectrum = np.abs(np.fft.fftshift(fft_img)) ** 2

    # Create radial frequency array
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    r_grid = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_max = max(100, int(np.min(center)))
    r = np.arange(5, r_max, 2)
    radial_profile = []

    for i in r:
        mask = (r_grid >= i - 1) & (r_grid < i + 1)
        radial_profile.append(np.mean(power_spectrum[mask]) * i)

    return r[np.argmax(radial_profile)]
"""


def preprocess(img_raw):
    if len(img_raw.shape) == 2:
        img = img_raw / 256
    elif img_raw.shape[2] == 2:
        img = img_raw[:, :, 0] / 256
    elif img_raw.shape[2] == 3:
        img = rgb2gray(img_raw)
    elif img_raw.shape[2] == 4:
        img = rgb2gray(img_raw[:, :, :3]) * (img_raw[:, :, 3] / 256)
    else:
        raise ValueError(f"invalid image shape: {img_raw.shape}")


    # Calculate scale factor to make largest dimension 256
    max_dim = max(img.shape[:2])
    scale_factor = 512.0 / max_dim

    # Rescale the image
    img = 1-transform.rescale(img, scale_factor, order=1, anti_aliasing=False)
    return img.astype(np.float32)

    #size = min(img.shape[0], img.shape[1])

    #img = gaussian(img, sigma=size / 500)
    #block_size = 2 * (size // 50) + 1
    #t1 = threshold_local(img, block_size)
    #t2 = threshold_otsu(img)
    #img_binary = img < (t1 + t2) / 2

    #thicknesses = np.zeros_like(img_binary, dtype=int)

    #max_line_width_detection = img_binary.copy()
    #i = 0
    #while np.sum(max_line_width_detection) > 0:
    #    # iterate the erosion
    #    tmp = erosion(max_line_width_detection)
    #    # find the difference
    #    diff = max_line_width_detection ^ tmp > 0
    #    thicknesses[diff] = i
    #    max_line_width_detection = tmp
    #    i += 1

    #c = int(np.quantile(thicknesses[thicknesses > 0], 0.9))

    ## remove holes, objects and dilate
    #img_binary = remove_small_objects(img_binary, 4 * c * c)
    #img_binary = remove_small_holes(img_binary, 4 * c * c)
    #img_binary = dilation(img_binary, disk(c // 2))
    #img_binary = remove_small_holes(img_binary)
    #thicknesses = ndi.maximum_filter(thicknesses, footprint=disk(size / 100))
    #return img_binary, thicknesses
