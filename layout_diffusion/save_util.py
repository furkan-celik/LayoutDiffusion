import imageio
import numpy as np

from .dataset.util import image_unnormalize_batch, get_cropped_image


def imageio_save_image(img_tensor, path, unnormalize=False):
    """
    :param img_tensor: (C, H, W) torch.Tensor
    :param path:
    :param args:
    :param kwargs:
    :return:
    """
    if unnormalize:
        tmp_img = image_unnormalize_batch(img_tensor).clamp(0.0, 1.0)
    else:
        tmp_img = img_tensor

    imageio.imsave(
        uri=path,
        im=(tmp_img.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        ),  # (H, W, C) numpy
    )
