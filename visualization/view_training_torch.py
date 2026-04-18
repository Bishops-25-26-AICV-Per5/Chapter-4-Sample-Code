"""
Author: TBSDrJ
Date: Spring 2025
Purpose: Demonstrate how to see what your model is seeing
Dataset: https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda/data
Notes:
    -Panda 131 not a panda
    -The order of the layers in model.children() matches the order that
    they are added in the __init__, *not* the order that they are run.
    print(model.children()) will get you the list of layers in the order
    that they are stored so you can pick out the ones you need.
"""
from pprint import pprint
import random
import pathlib
import os

import torch
import numpy as np
import cv2

IM_SZ = 224

class Model(torch.nn.Module):
    def __init__(self, input_shape: (int, int, int)):
        super().__init__()
        self.input_shape = input_shape
        # Reusable layers
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2)

        # Now into the architecture
        # 3 x 224 x 224
        self.zp0 = torch.nn.ZeroPad2d((1, 2, 1, 2))
        # 3 x 227 x 227
        self.conv0 = torch.nn.Conv2d(3, 24, 11, stride=4)
        # (227 - 11)/4 + 1 = 216/4 + 1 = 54 + 1
        # 24 x 55 x 55
        # Maxpool, 3x3 stride=2: (55 - 3)/2 + 1 = 52/2 + 1 = 27
        # 24 x 27 x 27
        self.bn0 = torch.nn.BatchNorm2d(24)
        self.zp1 = torch.nn.ZeroPad2d((1, 1, 1, 1))
        # 24 x 29 x 29
        self.conv1 = torch.nn.Conv2d(24, 32, 3, stride=1)
        # (29 - 3)/1 + 1 = 27
        # 32 x 27 x 27
        # Maxpool, 3x3 stride=2: (27 - 3)/2 + 1 = 13
        # 32 x 13 x 13
        self.bn1 = torch.nn.BatchNorm2d(32)
        # re-use self.zp1
        # 32 x 15 x 15
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1)
        # (15 - 3)/1 + 1 = 13
        # 32 x 13 x 13
        # No maxpool this time around
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.flatten = torch.nn.Flatten()
        self.linear0 = torch.nn.Linear(5408, 1024)
        self.linear1 = torch.nn.Linear(1024, 256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # Block 0
        y = self.bn0(self.maxpool(self.relu(self.conv0(self.zp0(x)))))
        # Block 1
        y = self.bn1(self.maxpool(self.relu(self.conv1(self.zp1(y)))))
        # Block 2
        y = self.bn2(self.relu(self.conv2(self.zp1(y))))
        y = self.relu(self.linear0(self.flatten(y)))
        y = self.relu(self.linear1(y))
        y = self.relu(self.linear2(y))
        y = self.softmax(self.linear3(y))
        return y

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
    # print(panda_path)
    return cat, dog, panda

def display_results(img: np.ndarray) -> None:
    """Pass the image through the first convolution layer and display."""
    img_display = img.copy()
    img = img / 255.
    img = img[None, :]
    # Convert shape to channels first, so 3 x 224 x 224
    img = np.moveaxis(img, -1, 1)
    # Range settings based on checkpoints
    for epoch in range(0, 101, 5):
        model = torch.load(f'saves/model_torch_2026_03_17_13_32_{epoch:03}.pt', weights_only=False)
        img = torch.tensor(img, dtype=torch.float32, device="mps")
        layers = list(model.children()) # See note at top
        result = layers[2](img) # zero-pad
        result = layers[3](result) # conv
        result = layers[0](result) # relu
        result = result.detach().cpu().numpy()[0]
        # 4 x 6 array, with 20 px padding between
        mosaic = np.zeros((660 + 60, 990 + 100))
        print(f"Checkpoint epoch {epoch:03}")
        for i in range(4):
            for j in range(6):
                layer = 4*i + j
                new_result = result[layer,:,:]
                new_result = cv2.resize(new_result, (165, 165))
                # print(new_result.max())
                # print(new_result.min())
                new_result -= new_result.min()
                new_result /= (new_result.max() - new_result.min())
                # print(new_result.max())
                # print(new_result.min())
                mosaic[185*i:185*i+165, 185*j:185*j+165] = new_result[:,:]
        cv2.imshow("Original Image", img_display)
        cv2.waitKey()
        cv2.imshow("Filtered", mosaic)
        if epoch <= 10:
            cv2.waitKey(20000)
        else:
            cv2.waitKey(250)

def main():
    cat, dog, panda = get_imgs()
    display_results(panda)
    display_results(dog)
    display_results(cat)

if __name__ == "__main__":
    main()