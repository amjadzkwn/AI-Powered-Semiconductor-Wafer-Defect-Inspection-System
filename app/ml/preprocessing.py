from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (64, 64)
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class InvalidWaferInput(ValueError):
    pass


def _normalize_wafer_values(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if array.ndim != 2:
        raise InvalidWaferInput("Wafer input must be a two-dimensional array or image.")

    unique = set(np.unique(array).tolist())
    if unique.issubset({0, 1, 2}):
        wafer = array.astype(np.uint8)
    else:
        grayscale = array.astype(np.float32)
        if grayscale.max() <= 1.0:
            grayscale *= 255.0
        wafer = np.zeros(grayscale.shape, dtype=np.uint8)
        wafer[(grayscale >= 43) & (grayscale < 170)] = 1
        wafer[grayscale >= 170] = 2

    return cv2.resize(wafer, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)


def decode_upload(content: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy":
        try:
            array = np.load(BytesIO(content), allow_pickle=False)
        except Exception as error:
            raise InvalidWaferInput("Invalid NPY file.") from error
        return _normalize_wafer_values(array)

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise InvalidWaferInput("Supported formats: PNG, JPG, JPEG, BMP, and NPY.")

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as error:
        raise InvalidWaferInput("The uploaded file is not a valid image.") from error
    return _normalize_wafer_values(np.asarray(image))


def wafer_to_tensor(wafer_map: np.ndarray) -> torch.Tensor:
    channels = np.stack(
        [(wafer_map == value).astype(np.float32) for value in (0, 1, 2)], axis=0
    )
    return torch.from_numpy(channels).unsqueeze(0).float()
