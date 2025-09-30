import tensorflow as tf
import tensorflow_datasets as tfds

# load data
DATA_DIR = './tensorflow-datasets/'

def load_datasets(batch_size=32):
    # training = 90% of train, validation = 10%, test predefined
    train_ds = tfds.load('fashion_mnist', split='train[:90%]', data_dir=DATA_DIR, shuffle_files=True)
    val_ds = tfds.load('fashion_mnist', split='train[-10%:]', data_dir=DATA_DIR, shuffle_files=True)
    test_ds = tfds.load('fashion_mnist', split='test', data_dir=DATA_DIR, shuffle_files=True)

    # preprocessing function
    def preprocess(sample):
        # cast to float (will normalize to 0-1 scale in model)
        image = tf.cast(sample['image'], tf.float32)

        # one-hot encode labels for use of softmax/probabilty distribution
        label = tf.one_hot(sample['label'], depth=10)
      
        return image, label

    # apply preprocessing + batching
    train_ds = train_ds.map(preprocess).shuffle(10000).batch(batch_size)
    val_ds = val_ds.map(preprocess).batch(batch_size)
    test_ds = test_ds.map(preprocess).batch(batch_size)

    return train_ds, val_ds, test_ds
