"""
    Author: TBSDrJ
    Date: Spring 2023
    Purpose: Illustrate word embeddings along the lines of word2vec.
    Reference: https://www.tensorflow.org/text/guide/word_embeddings
    Uses dataset found at: 
        https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    The train and test each had 12 500 entries for each of positive and
        negative reviews.  I decided to combine these to get a single dataset
        with 25 000 entries each of positive and negative and do a random 
        train/validation split.  Note that when combining, some of the 
        filenames for the text files are the same, so one has to be careful
        to avoid overwriting those files in the combination process.
    My combined dataset can be found at:
        https://drive.google.com/file/d/1s-0zOF-FhdUwo2jq5QpIDxR64SFS9pLQ/view?usp=sharing
"""
import pickle

import tensorflow as tf
import numpy as np

BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.3
# Set this to some integer to limit the size of the vocabulary, or None
#   to use every word that is found.
VOCAB_SIZE = 10000
# I believe that this truncates the length of each review to 100 words.
SEQUENCE_LENGTH = 100
# Number of dimensions to capture the meaning of each word.  
#   word2vec used 300 dimensions.
EMBEDDING_DIM = 256

def get_datasets() -> (tf.data.Dataset, tf.data.Dataset):
    train, valid = tf.keras.utils.text_dataset_from_directory(
        'imdb/combined',
        batch_size = BATCH_SIZE,
        validation_split = VALIDATION_SPLIT,
        subset = 'both',
        seed = 37,
    )
    train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    valid.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    return train, valid

def clean_text(input_data):
    # Convert all to lower case
    # Get rid of everything that isn't a letter, a space or an apostrophe
    input_data = tf.strings.regex_replace(input_data, "[^a-z' ]", '')
    return input_data

def get_vectorization(train: tf.data.Dataset) -> tf.keras.layers.Layer:
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize = clean_text,
        max_tokens = VOCAB_SIZE,
        output_mode = 'int',
        output_sequence_length = SEQUENCE_LENGTH,
    )
    train_text = train.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)
    return vectorize_layer

def get_model(vectorize_layer: tf.keras.layers.Layer) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            vectorize_layer,
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name='embedding'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=10**(-3)),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model

def find_closest(vocab:list[str], weights: np.ndarray, word: str, n: int
    ) -> list[str]:
    """Given a word, find the n words nearest in the embedding."""
    find_key = vocab.index(word)
    weights_key = weights[find_key]
    closest = []
    for i, word in enumerate(vocab):
        wgts = weights[i]
        distances = [abs(weights_key[j] - wgts[j]) for j in range(EMBEDDING_DIM)]
        distance = sum(distances)
        if len(closest) < n + 1:
            closest.append((i, distance))
            closest.sort(key=lambda x: x[1])
        else:
            if distance < closest[n][1]:
                closest.pop(n)
                closest.append((i, distance))
                closest.sort(key=lambda x: x[1])
    return closest


def main():
    train, valid = get_datasets()
    vectorize_layer = get_vectorization(train)
    vocab = vectorize_layer.get_vocabulary()
    model = get_model(vectorize_layer)
    print(model.summary())
    history = model.fit(
        train,
        validation_data = valid,
        epochs = 3,
    )

    weights = model.get_layer('embedding').get_weights()[0]
    closest = find_closest(vocab, weights, 'excellent', 10)
    for i, dist in closest:
        print(vocab[i])
    print()
    closest = find_closest(vocab, weights, 'awful', 10)
    for i, dist in closest:
        print(vocab[i])

    model.save('imdb_model.keras')
    with open('train_epoch_data.dat', 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    main()