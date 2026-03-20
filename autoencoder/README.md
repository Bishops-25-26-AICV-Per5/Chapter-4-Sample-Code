# Chapter 4
## Autoencoder Example

The goal is to demonstrate how to build an autoencoder and to give you a bit of a sense of how/why you'd use one.  My encoder/decoder networks are quite simple, and the encoder only uses zero-padding and convolutions.  It is a bit unusual to have no linear layers between the last convolution and the latents.

To use this:
  -You'll need at least the panda class from the cat-dog-panda dataset.  There is a link in the header of `autoencoder.py` or, assuming you are in my class, it is posted the Chapter 3 Topic on the Topics page in Blackbaud.
  -The two files `autoencoder.py` and `autoencoder_2.py` are for training the autocoder network.  I have provided two pre-trained models so that you don't have to train them if you don't want to.  The only different is in the learning rate scheduler.
  -The files `autoencoder_printout.txt` and `autoencoder_printout_2.txt` are the loss values for each epoch in the training.
  -The files `saves/encoder_150.pt` and `saves/decoder_150.pt` are from my run of `autoencoder.py` for 150 epochs.
  -The files `saves/encoder_2_300.pt` and `saves/decoder_2_300.pt` are from my run of `autoecoder_2.py` for 300 epochs.
  -The file `see_image.py` will show the result of passing a picture through the autoencoder.  Right now, there is a list of 10 images that give results that are pretty good through my runs of the autoencoder, it will automatically run through those first.  You can change the loaded model on lines 34 & 35.

