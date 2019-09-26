import os
import numpy as np
import imageio


def read_mask(uri, expand_dims=True, rootpath=None, **kwargs):
    if rootpath is not None:
        uri = uri if uri.startswith(rootpath) else os.path.join(rootpath, uri)

    mask = imageio.volread(uri, **kwargs)
    if expand_dims and len(mask.shape) < 3:  # single grayscale image
        mask = np.expand_dims(mask, -1)

    mask = np.clip(mask, 0, 1)

    return mask
