# -*- coding: utf-8 -*-
import kornia
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('img')
    args = parser.parse_args()
    img_file = args.img

    x_rgb: torch.tensor = torchvision.io.read_image(img_file)  # CxHxW / torch.uint8
    x_rgb = x_rgb.type(torch.float32)
    x_rgb = x_rgb[:3, :, :]
    x_rgb = x_rgb.unsqueeze(0)  # BxCxHxW
    print(x_rgb.shape)

    # out = kornia.filters.laplacian(x_rgb, 3)
    out = kornia.filters.sobel(x_rgb)
    print(out.shape)

    out_rgb: np.ndarray = kornia.tensor_to_image(out.byte())  # HxWxC / np.uint8
    int_rgb: np.ndarray = kornia.tensor_to_image(x_rgb.byte())  # HxWxC / np.uint8

    fig, axs = plt.subplots(1, 2, figsize=(32, 16))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].imshow(int_rgb)

    axs[1].axis('off')
    axs[1].imshow(out_rgb)

    plt.show()
