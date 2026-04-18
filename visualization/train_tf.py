"""
Author: TBSDrJ
Date: Spring 2026
Purpose: Image classification with Cat/Dog/Panda dataset
Dataset: https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda/data
"""
import atexit
import time

# I am using TF 2.16.2
import tensorflow as tf 

TIME_STAMP = time.strftime("%Y_%m_%d_%H_%M")

def my_print(*args, **kwargs) -> None:
    with open("saves/printout_" + TIME_STAMP + ".txt", "a") as f:
        print(*args, **kwargs)
        if "end" in kwargs and kwargs["end"] == "\r":
            del kwargs["end"]
        print(*args, **kwargs, file=f)

# Save code
with open(__file__, "r") as f:
        this_code = f.read()
with open("saves/code_" + TIME_STAMP + ".py", "w") as f:
        print(this_code, file=f)
@atexit.register
def clean_up() -> None:
    model.model.save(f"saves/model_{TIME_STAMP}_{epoch:03d}.keras")

BATCH_SIZE = 32

# Set the random seed so that we get reproducible results
tf.keras.utils.set_random_seed(37)

train, validation = tf.keras.utils.image_dataset_from_directory(
    'animals',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'rgb',
    batch_size = BATCH_SIZE,
    image_size = (224, 224),
    shuffle = True,
    seed = 8008,
    validation_split = 0.2,
    subset = 'both',
)

train = train.map(lambda x, y: (x/255., y))
validation = validation.map(lambda x, y: (x/255., y))
# Do the flip first
flipped = train.map(lambda x, y: (tf.image.flip_left_right(x), y))
train = train.concatenate(flipped)
# Now brightness and hue also apply to the flipped images
brightness = train.map(lambda x, y: (
        tf.image.stateless_random_brightness(x, 0.25, seed=(3, 7)), y))
hue = train.map(lambda x, y: (
        tf.image.stateless_random_hue(x, 0.25, (3, 7)), y))
train = train.concatenate(brightness)
train = train.concatenate(hue)
# So we have 6x the images

train = train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

class Model:
    def __init__(self, input_shape):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.InputLayer(shape=input_shape))
        # Conv2D, 11x11 frame, stride of 4, z-p of ((1,2),(1,2))
        self.model.add(tf.keras.layers.ZeroPadding2D(
                padding=((1,2),(1,2)),
        )) # Size (227, 227, 3)
        self.model.add(tf.keras.layers.Conv2D(
                24, # This is your depth
                (11, 11), # Frame size
                strides=(4, 4),
                # kernel_initializer=tf.keras.initializers.RandomUniform(0.05, 0.15),
                activation=tf.keras.activations.relu,
        )) # Size (55, 55, 16)
        self.model.add(tf.keras.layers.MaxPooling2D(
                pool_size=(3,3),
                strides=2,
        )) # Size (27, 27, 16)
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ZeroPadding2D(
                padding=((1,1),(1,1)),
        )) # Size (29, 29, 16)
        self.model.add(tf.keras.layers.Conv2D(
                32,
                (3, 3),
                strides=(1,1),
                activation=tf.keras.activations.relu,
                # kernel_initializer=tf.keras.initializers.RandomUniform(0.05, 0.15),
        )) # Size (27, 27, 24)
        self.model.add(tf.keras.layers.MaxPooling2D(
                pool_size=(3,3),
                strides=2,
        )) # Size (13, 13, 24)  
        self.model.add(tf.keras.layers.ZeroPadding2D(
                padding=((1,1),(1,1)),
        )) # Size (15, 15, 24)
        self.model.add(tf.keras.layers.Conv2D(
                32,
                (3, 3),
                strides=(1,1),
                activation=tf.keras.activations.relu,
                # kernel_initializer=tf.keras.initializers.RandomUniform(0.05, 0.15),
        )) # Size (13, 13, 24)
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(
                1024, 
                activation=tf.keras.activations.relu,
                # kernel_initializer=tf.keras.initializers.RandomUniform(0.05, 0.15),
        ))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(
                256, 
                # kernel_initializer=tf.keras.initializers.RandomUniform(0.05, 0.15),
                activation=tf.keras.activations.relu,
        ))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(
                64, 
                # kernel_initializer=tf.keras.initializers.RandomUniform(0.05, 0.15),
                activation=tf.keras.activations.relu,
        ))
        self.model.add(tf.keras.layers.Dense(
                3, # Cat, Dog, Panda
                # kernel_initializer=tf.keras.initializers.RandomUniform(0.05, 0.15),
                activation=tf.keras.activations.softmax,
        ))
        self.lr_sch = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, 450, 0.95, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_sch)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.metrics = ['accuracy']
        # Compile with optimizer, loss and metrics using above variables
        self.model.compile(
                optimizer = self.optimizer,
                loss = self.loss,
                metrics = self.metrics,
        )

model = Model((224,224,3))
model.model.summary(print_fn=my_print)

epoch = 0

class SetEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, in_epoch, logs=None):
        global epoch    
        epoch = in_epoch
set_epoch = SetEpoch()

class SaveCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 4:
            model.model.save(
                f"saves/model_tf_{TIME_STAMP}_{epoch+1:03d}.keras")
save_checkpoint = SaveCheckpoint()

# Save a version of the model before we start any training.
model.model.save(f"saves/model_tf_{TIME_STAMP}_{epoch:03d}.keras")

callbacks = [
    set_epoch,
    save_checkpoint,
    tf.keras.callbacks.CSVLogger(
        f"saves/results_{TIME_STAMP}.csv", append=True),
]

model.model.fit(
        train,
        batch_size = BATCH_SIZE,
        epochs = 100,
        verbose = 1,
        validation_data = validation,
        callbacks = callbacks,
)
