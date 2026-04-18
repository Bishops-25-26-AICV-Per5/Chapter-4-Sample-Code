import tensorflow.keras.models as models
import numpy as np
import cv2

def main():
    for epoch in range(0, 101, 5):
        model = models.load_model(f'saves/model_tf_2026_03_17_11_33_{epoch:03}.keras')
        conv_wgts = model.layers[1].kernel.numpy()
        # Duplicate each filter pixel to a nxn block to make it more visible
        n = 10
        mosaic = np.zeros((4*n*11 + 3*20, 6*n*11 + 5*20, 3))
        for i in range(4):
            for j in range(6):
                filter = conv_wgts[:,:,:,6*i + j]
                filter_img = np.repeat(filter, n, axis=0)
                filter_img = np.repeat(filter_img, n, axis=1)
                mosaic[11*n*i + i*20:11*n*(i+1) + i*20, 11*n*j + j*20:11*n*(j+1) + j*20, :] = filter_img
        # print(mosaic.min(), mosaic.max())
        # mosaic -= mosaic.min()
        mosaic /= mosaic.max()
        # print(mosaic.min(), mosaic.max())
        cv2.imshow("", mosaic)
        if epoch < 100:
            cv2.waitKey(20000)
        else:
            cv2.waitKey(200)

if __name__ == "__main__":
    main()