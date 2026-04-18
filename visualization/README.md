# Visualizing what your CNN is seeing

This one requires a little setup.  Also, this one is available in both Tensorflow and Torch.  To complete the setup:

1. You'll need the Cat - Dog - Panda dataset, which is available [here](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda/data) or on Blackbaud (Topics, Chapter 3).  Put the `animals` folder in the folder where all this code lives.
1. You'll also need a folder called `saves`. You can either download my [training results here](https://drive.google.com/drive/folders/1XDATc5zqyWRPeDp3vQykQm5tIiKsEtia?usp=sharing) or you can run `train_tf.py` or `train_torch.py` to create your own.  If you create your own, then, in the `view` codes, you'll need to change the filename to match your time stamp.

The programs `view_filters` will show you a visualization of the actual weights in the filters of the first convolution layer.  The programs `view_training` will show you the output after those filters are applied to the input image.

As we saw in class, somewhere around ¼ to ⅓ of the filters produce nothing but zeros after the `ReLU` layer. I'm not entirely sure what to do about that, or if one *should* in fact do something about that.