# Chapter 4 Sample Code: RNN

So far: 
  - I have only a version that was built from [Dive Into Deep Learning](https://d2l.ai/) that avoids using its custom library.  This material is all in Chapter 9 of that text.  I'm continuing to work on building an RNN that looks more like the code I've provided all year.
  - In order to get Github to take the saved models, I had to zip them.  So, there are three files called `d2l_model_nnn.pt.zip` (where the `nnn` is a number = # of epochs of training to build this model).  You'll need to unzip them to use them.
  - This is set to load a model that I trained for 1000 epochs (and you'll need to unzip the model to get it to work).  If you want it to train, look at the last few lines.  Un-comment the line that says `trainer.fit()`, and comment the next two lines that load the pre-trained model.  It will print out the elapsed time after each epoch, and will show the results for a specific sentence fragment that I pulled from the text every 25 epochs.
  - I am using _Moby_ _Dick_, by Herman Melville, it is pretty easy to substitute other text or mess with other parameters of the model.
  - I have also provided saved models at 100 epochs and 200 epochs.
