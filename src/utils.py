import os
import numpy as np
import imageio


def maskread(uri, expand_dims=True, rootpath=None, **kwargs):
    """Read .tiff mask"""
    if rootpath is not None:
        uri = uri if uri.startswith(rootpath) else os.path.join(rootpath, uri)

    masks = imageio.mimread(uri, **kwargs)
    mask = np.clip(np.dstack(masks), 0, 1)

    if expand_dims and len(mask.shape) < 3:  # single grayscale image
        mask = np.expand_dims(mask, -1)

    return mask
