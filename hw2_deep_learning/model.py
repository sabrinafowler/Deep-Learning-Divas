from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 100

# Model architectures
def build_model_a(input_shape=IMG_SHAPE, l2_reg=1e-4, dropout_rate=0.0):
    """Smaller CNN: two conv+pool blocks, one dense before softmax"""
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=reg),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def build_model_b(input_shape=IMG_SHAPE, l2_reg=5e-4, dropout_rate=0.4):
    """Deeper CNN with more filters and dropout"""
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=reg)(inputs)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Data augmentation
def make_data_generator():
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        zoom_range=0.08
    )
    return datagen


EXPERIMENTS = [
    ("ModelA", build_model_a, [
        # Run 1: Adam, smaller LR, moderate regularization
        # faster training, Adam good for baseline experiments, 0.001 lr for convergence/speed tradeoff
        {"optimizer": Adam(learning_rate=0.001), "l2_reg": 1e-4, "dropout": 0.2, "batch_size": 256, "label": "A-adam-lr1e-3-l2-1e-4-dr0.2"},
        
        # Run 2: SGD (with momentum), lower LR, slightly higher reg
        # SGD is better generlization, .9 momentum used to avoid osicllating, not as adaptive tho
        # uses slightly higher dropout for stabilization, since SGD is usually noisier
        {"optimizer": SGD(learning_rate=0.01, momentum=0.9), "l2_reg": 5e-4, "dropout": 0.3, "batch_size": 256, "label": "A-sgd-lr1e-2-l2-5e-4-dr0.3"}
    ]),
    
    ("ModelB", build_model_b, [
        # using stronger regularization and dropout, since these models have more params (dont want to memorize training data)
        # Run 1: Adam, slightly stronger regularization and dropout (deeper model)
        {"optimizer": Adam(learning_rate=0.001), "l2_reg": 5e-4, "dropout": 0.4, "batch_size": 256, "label": "B-adam-lr1e-3-l2-5e-4-dr0.4"},
        
        # Run 2: SGD with higher reg + dropout
        {"optimizer": SGD(learning_rate=0.01, momentum=0.9), "l2_reg": 1e-3, "dropout": 0.5, "batch_size": 256, "label": "B-sgd-lr1e-2-l2-1e-3-dr0.5"}
    ])
]