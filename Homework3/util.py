"""
Helper functions for data loading and preprocessing
"""

import tensorflow as tf
import tensorflow_datasets as tfds


# Parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 128
BATCH_SIZE = 64
DATA_DIR = './tensorflow-datasets/'


def load_data():
    """
    Load IMDB dataset and split into train/val/test

    Returns:
        tuple: (train_ds, val_ds, test_ds) - raw datasets
    """
    train_ds = tfds.load('imdb_reviews', split='train[:90%]', data_dir=DATA_DIR, shuffle_files=True)
    val_ds = tfds.load('imdb_reviews', split='train[-10%:]', data_dir=DATA_DIR, shuffle_files=True)
    test_ds = tfds.load('imdb_reviews', split='test', data_dir=DATA_DIR, shuffle_files=True)

    return train_ds, val_ds, test_ds


def create_vectorization_layer(train_ds):
    """
    Create and adapt TextVectorization layer on training data

    Args:
        train_ds: Training dataset

    Returns:
        TextVectorization layer
    """
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_LENGTH
    )

    # Adapt on training data
    vectorize_layer.adapt(train_ds.map(lambda x: x['text']))

    return vectorize_layer


def preprocess_datasets(train_ds, val_ds, test_ds, vectorize_layer):
    """
    Preprocess datasets with vectorization, batching, and prefetching

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        vectorize_layer: TextVectorization layer

    Returns:
        tuple: (train_ds, val_ds, test_ds) - preprocessed datasets
    """
    def preprocess(data):
        text = vectorize_layer(data['text'])
        label = data['label']
        return text, label

    train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def prepare_data():
    """
    Complete data preparation pipeline

    Returns:
        tuple: (train_ds, val_ds, test_ds) - preprocessed and ready datasets
    """
    print("Loading data...")
    train_ds, val_ds, test_ds = load_data()

    print("Creating vectorization layer...")
    vectorize_layer = create_vectorization_layer(train_ds)

    print("Preprocessing datasets...")
    train_ds, val_ds, test_ds = preprocess_datasets(train_ds, val_ds, test_ds, vectorize_layer)

    print("Data loaded and preprocessed!")
    print(f"Max length: {MAX_LENGTH}, Batch size: {BATCH_SIZE}")

    return train_ds, val_ds, test_ds
