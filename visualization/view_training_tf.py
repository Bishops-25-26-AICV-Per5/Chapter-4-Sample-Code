"""
Author: TBSDrJ
Date: Spring 2025
Purpose: Demonstrate how to see what your model is seeing
Dataset: https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda/data
# Panda 131 not a panda
"""
from pprint import pprint
import random
import pathlib
import os

import tensorflow as tf
import numpy as np
import cv2

IM_SZ = 224

def get_imgs() -> (np.ndarray, np.ndarray, np.ndarray):
    """Pick a random image from each of the three classes"""
    r = random.randrange(1, 1001)
    base_path = pathlib.Path('animals')
    cat_path = base_path / 'cat' / f'cats_{r:05}.jpg'
    cat = cv2.imread(cat_path)
    cat = cv2.resize(cat, (IM_SZ, IM_SZ))
    dog_path = base_path / 'dog' / f'dogs_{r:05}.jpg'
    dog = cv2.imread(dog_path)
    dog = cv2.resize(dog, (IM_SZ, IM_SZ))
    panda_path = base_path / 'panda' / f'panda_{r:05}.jpg'
    panda = cv2.imread(panda_path)
    panda = cv2.resize(panda, (IM_SZ, IM_SZ))
    return cat, dog, panda

def display_results(img: np.ndarray) -> None:
    """Pass the image through the first convolution layer and display."""
    img_display = img.copy()
    img = img / 255.
    img = img[None, :]
    # Range settings based on checkpoints
    for epoch in range(0, 101, 5):
        model = tf.keras.models.load_model(f'saves/model_tf_2026_03_17_11_33_{epoch:03}.keras', safe_mode=False)
        # Zero-Padding Layer
        padded = model.layers[0](img)
        # Convolution Layer (then discard batch dimension)
        result = model.layers[1](padded)[0]
        conv_wgts = model.layers[1].kernel.numpy()
        conv_bias = model.layers[1].bias.numpy()
        result = result.numpy()
        shape = result.shape
        # 4 x 6 array, with 20 px padding between
        mosaic = np.zeros((660 + 60, 990 + 100))
        print(f"Checkpoint epoch {epoch:03}")
        for i in range(4):
            for j in range(6):
                layer = 4*i + j
                new_result = result[:,:,layer]
                new_result = cv2.resize(new_result, (165, 165))
                # print(new_result.max())
                # print(new_result.min())
                # if (new_result.max() < 1e-6):
                #     print(conv_wgts[...,layer])
                #     print(sum(sum(sum(abs(conv_wgts[...,layer])))))
                #     print(conv_bias[layer])
                #     quit()
                # else:
                new_result -= new_result.min()
                new_result /= (new_result.max() - new_result.min())
                # print(new_result.max())
                # print(new_result.min())
                mosaic[185*i:185*i+165, 185*j:185*j+165] = new_result[:,:]
        cv2.imshow("Original Image", img_display)
        cv2.imshow("Filtered", mosaic)
        if epoch <= 10:
            cv2.waitKey(20000)
        else:
            cv2.waitKey(2500)

# tf.keras.utils.set_random_seed(314159)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

cat, dog, panda = get_imgs()
white = np.ones((224, 224, 3)) * 255.
# display_results(white)
display_results(panda)
display_results(dog)
display_results(cat)
