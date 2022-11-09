import cv2
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def crop(image: np.ndarray) -> np.ndarray:
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[80:-30, :, :]  # remove the sky and the car front


def resize(image: np.ndarray) -> np.ndarray:
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image: np.ndarray) -> np.ndarray:
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Combine all preprocess functions into one
    """
    image = crop(image=image)
    image = resize(image=image)
    image = rgb2yuv(image=image)

    if normalize:
        image = (image / 127.5) - 1.0

    return image
