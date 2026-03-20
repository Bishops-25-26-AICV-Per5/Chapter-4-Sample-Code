"""
Author: TBSDrJ
Date: Spring 2025
Purpose: Pairs with Autoencoder that detects pandas in images to show results.
"""
from pprint import pprint
import random
import pathlib
import os

import torch
import numpy as np
import cv2

good_imgs = [9, 708, 864, 782, 163, 125, 40, 965, 341, 932]

def get_img() -> (np.ndarray, int):
    """Pick a random image from out of the pandas"""
    global good_imgs
    if good_imgs:
        r = good_imgs.pop(0)
    else:
        r = random.randrange(1, 1001)
    base_path = pathlib.Path('animals')
    panda_path = base_path / 'panda' / f'panda_{r:05}.jpg'
    panda = cv2.imread(panda_path)
    panda = cv2.resize(panda, (224, 224))
    return panda, r

def display_results() -> None:
    """Pass the image through the first convolution layer and display."""
    # Convert shape to channels first, so 3 x 224 x 224
    # Range settings based on checkpoints
    encoder = torch.load(f'saves/encoder_2_300.pt', weights_only=False)
    decoder = torch.load(f'saves/decoder_2_300.pt', weights_only=False)
    encoder.to(torch.device("mps"))
    decoder.to(torch.device("mps"))
    ch = 65
    while chr(ch).lower() != "q":
        img, r = get_img()
        img_ch_1st = np.moveaxis(img, -1, 0)
        img_tensor = torch.tensor(img_ch_1st, dtype=torch.float32, device="mps")
        result = encoder(img_tensor)
        result = decoder(result)
        result = result.detach().cpu().numpy()
        result = np.moveaxis(result, 0, -1)
        cv2.imshow("Original Image", img)
        cv2.imshow("Result", result)
        ch = cv2.waitKey(0)

def main():
    display_results()

if __name__ == "__main__":
    main()