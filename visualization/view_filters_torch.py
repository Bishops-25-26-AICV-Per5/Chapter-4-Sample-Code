import torch
import numpy as np
import cv2

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

def main():
    for epoch in range(0, 101, 5):
        print(f"Epoch {epoch}")
        model = torch.load(f'saves/model_torch_2026_03_17_13_25_{epoch:03}.pt', weights_only=False)
        layers = list(model.children())
        zp = layers[2]
        conv = layers[3]
        weights = conv.weight
        # print(weights.shape)
        # Duplicate each filter pixel to a nxn block to make it more visible
        n = 10
        mosaic = np.zeros((4*n*11 + 3*20, 6*n*11 + 5*20, 3))
        for i in range(4):
            for j in range(6):
                filter = weights[6*i + j,:,:,:,].detach().cpu().numpy()
                # print(filter.shape)
                filter = np.transpose(filter)
                # print(filter.shape)
                filter_img = np.repeat(filter, n, axis=0)
                filter_img = np.repeat(filter_img, n, axis=1)
                mosaic[11*n*i + i*20:11*n*(i+1) + i*20, 11*n*j + j*20:11*n*(j+1) + j*20, :] = filter_img
        # print(mosaic.min(), mosaic.max())
        # mosaic -= mosaic.min()
        mosaic /= mosaic.max()
        # print(mosaic.min(), mosaic.max())
        cv2.imshow("", mosaic)
        if epoch <= 10:
            cv2.waitKey(20000)
        else:
            cv2.waitKey(2000)

if __name__ == "__main__":
    main()