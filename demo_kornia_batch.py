# -*- coding: utf-8 -*-
import kornia
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from typing import Tuple
import cv2
import torch.cuda
from PIL import Image


def read_image(img_file):
    x_rgb: torch.tensor = torchvision.io.read_image(
        img_file)  # CxHxW / torch.uint8
    x_rgb = x_rgb.type(torch.float32)
    x_rgb = x_rgb.unsqueeze(0)  # BxCxHxW
    return x_rgb


def save_image(img, filename):
    out_rgb: np.ndarray = kornia.tensor_to_image(
        img.cpu().byte())  # HxWxC / np.uint8
    Image.fromarray(out_rgb).save(filename)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    n = 8
    with torch.no_grad(), ThreadPoolExecutor(max_workers=n) as executor:
        files_list = sorted(os.listdir(args.inp))
        for img_files in tqdm([files_list[i:i+n] for i in range(0, len(files_list), n)]):
            img_filepaths = [os.path.join(args.inp, x) for x in img_files]
            x_rgbs: Tuple[torch.tensor] = tuple(
                executor.map(read_image, img_filepaths))
            imgs = torch.zeros(
                (n, 3, x_rgbs[0].shape[1], x_rgbs[0].shape[2]), requires_grad=False)
            torch.cat(x_rgbs, out=imgs)
            imgs = imgs.to(device)
            # out = kornia.filters.laplacian(x_rgb, 3)
            outs = kornia.filters.sobel(imgs, normalized=False)

            basenames = [os.path.basename(x) for x in img_files]
            out_filenames = [os.path.join(args.out, basename)
                             for basename in basenames]
            results = executor.map(lambda args: save_image(
                *args), zip(outs, out_filenames))

            
