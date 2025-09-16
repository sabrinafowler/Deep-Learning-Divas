import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

# Load datasets
DATA_DIR = './tensorflow-datasets/'

train_ds = tfds.load('fashion-mnist', split='train[:90%]', data_dir=DATA_DIR, shuffle_files=True)
val_ds = tfds.load('fashion-mnist', split='train[-10%:]', data_dir=DATA_DIR, shuffle_files=True)
test_ds = tfds.load('fashion-mnist', split='test', data_dir=DATA_DIR, shuffle_files=True)

# Shuffle and batch
train_ds = train_ds.shuffle(1024).batch(32)
val_ds = val_ds.shuffle(1024).batch(32)
test_ds = test_ds.shuffle(1024).batch(32)

# Train on the training data, set up an early stop based on validation data
#  Architecture 1
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(50, tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(20, tf.nn.relu))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adam()

val_accuracy = 0
val_interval = 50
patience = 3
stop_check = 0
early_stop = False
# Loop through fifteen epochs of data
for epoch in range(15): 
    print('Epoch', epoch) 
    n_batches = 0 
    for batch in tqdm(train_ds): 
        with tf.GradientTape() as tape: 
            # run network 
            x = tf.reshape(tf.cast(batch['image'], tf.float32)/255.0, [-1, 784]) 
            labels = batch['label'] 
            logits = model(x) 
            # calculate loss 
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) 
            scalar_loss = tf.reduce_mean(loss) 
            # increase batch number 
            n_batches+=1 
        # gradient update 
        grads = tape.gradient(scalar_loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        
        # validation check 
        if n_batches % val_interval == 0: 
            val_accuracy_values = [] 
            for batch in val_ds:  
                # run network 
                x = tf.reshape(tf.cast(batch['image'], tf.float32)/255.0, [-1, 784]) 
                labels = batch['label'] 
                logits = model(x) 
                # calculate accuracy 
                predictions = tf.argmax(logits, axis=1) 
                accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)) 
                val_accuracy_values.append(accuracy.numpy()) 
            accuracy_mean = np.mean(val_accuracy_values) 
                    
            # early stopping 
            if accuracy_mean - val_accuracy > 0.001: 
                val_accuracy = accuracy_mean 
                stop_check = 0 
            else: 
                stop_check += 1 
            if stop_check >= patience: 
                early_stop = True 
                break 
    if early_stop: 
        break

#######################################################################################################
# Report performance based on test data
loss_values = []
accuracy_values = []
batch_loss = []
# Loop through one epoch of data
for epoch in range(1):
    for batch in test_ds:
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32)/255.0, [-1, 784])
            labels = batch['label']
            logits = model(x)

            # calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
            scalar_loss = tf.reduce_mean(loss)
        loss_values.append(loss.numpy())
        batch_loss.append(scalar_loss.numpy())
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        accuracy_values.append(accuracy)

print(model.summary())

# print accuracy
print("Accuracy:", np.mean(accuracy_values))
# plot per-datum loss
loss_values = np.concatenate(loss_values)
print("Min. loss value: ", np.min(loss_values), "Max loss value: ", np.max(loss_values))
plt.hist(loss_values, density=True, bins=30)
plt.xlabel("Loss values")
plt.ylabel("frequency")
plt.show()


