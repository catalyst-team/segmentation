import os


def has_image_ext(fname: str) -> bool:
    name, ext = os.path.splitext(fname)
    return ext.lower() in {".bmp", ".png", ".jpeg", ".jpg", ".tiff", ".tif"}


def find_in_dir(dirname: str):
    result = [
        os.path.join(dirname, fname)
        for fname in sorted(os.listdir(dirname))
    ]
    return result


def find_images_in_dir(dirname: str):
    return [fname for fname in find_in_dir(dirname) if has_image_ext(fname)]


def id_from_fname(fname: str):
    return os.path.splitext(os.path.basename(fname))[0]
