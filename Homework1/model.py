import tensorflow as tf

# general build model function
def build_model(layers=[128], learning_rate=0.001, use_regularizer=False):
    model = tf.keras.Sequential()

    # normalize to 0-1 scale
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(28,28,1)))
    # flatten vector
    model.add(tf.keras.layers.Flatten())

    # regularize with L2 (penalize excessively large weights, but keep all features) if use_regularizer: 
    reg = tf.keras.regularizers.l2(0.001) if use_regularizer else None

    # add hidden layers
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer, activation="relu", kernel_regularizer=reg))

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # compile model with Adam 
    # categorical crossentropy used with one-hot encoded labels 
    # track accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
