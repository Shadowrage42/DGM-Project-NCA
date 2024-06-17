import os
import numpy as np
import imageio


def load_emoji(index, path="datasets/emoji/emoji.png"):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji


def load_image(path: str="datasets/emoji.png"):
    """
    Loads an entire image from the given path and normalizes its pixel values.

    Parameters:
    path (str): The path to the image. Defaults to "datasets/emoji.png".

    Returns:
    np.ndarray: A 2D array representing the image, normalized to the range [0, 1].
    """
    im = imageio.imread(path)
    image = np.array(im.astype(np.float32))
    image /= 255.0
    return image


def split_emojis(path: str="datasets/emoji.png", number: int=10):
    """
    Splits a sprite sheet image containing multiple emojis into individual emoji images.

    Parameters:
    path (str): The path to the sprite sheet image containing the emojis. Defaults to "datasets/emoji.png".

    The function assumes emojis to be 40 pixels wide.
    It saves each emoji as a separate PNG file in the sheet's directory.
    """
    savepath = os.path.dirname(path)
    im = imageio.imread(path)
    for index in range(number):
        emoji = im[:, index*40:(index+1)*40]
        imageio.imwrite(f'{savepath}/{index}.png', emoji)
        