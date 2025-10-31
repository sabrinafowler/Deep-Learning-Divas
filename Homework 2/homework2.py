import os
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.datasets import cifar100
import seaborn as sns  # visualization helper (optional)

# Reproducibility to some degree
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Config / Hyperparams
RESULT_DIR = "/content/cifar100_exp_results"
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 100
BATCH_SIZE = 128
MAX_EPOCHS = 200  # will cut short by early stopping
PATIENCE = 5      # >= 5 as required


# Utility functions
def load_and_preprocess(validation_size=5000):
    """Load CIFAR-100 from Keras, normalize images, one-hot labels, split train/val/test."""
    (x_train_all, y_train_all), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    x_train_all = x_train_all.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train_all = to_categorical(y_train_all.flatten(), NUM_CLASSES)
    y_test = to_categorical(y_test.flatten(), NUM_CLASSES)

    # Shuffle training data
    idx = np.arange(x_train_all.shape[0])
    np.random.shuffle(idx)
    x_train_all = x_train_all[idx]
    y_train_all = y_train_all[idx]

    # Create validation split
    if validation_size >= x_train_all.shape[0]:
        raise ValueError("validation_size too large")
    x_val = x_train_all[:validation_size]
    y_val = y_train_all[:validation_size]
    x_train = x_train_all[validation_size:]
    y_train = y_train_all[validation_size:]
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def plot_history(history, title, savepath=None):
    """Plot accuracy and loss curves from a keras History object."""
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='train acc')
    plt.plot(history.history.get('val_accuracy', []), label='val acc')
    plt.title(f"{title} - Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='train loss')
    plt.plot(history.history.get('val_loss', []), label='val loss')
    plt.title(f"{title} - Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()

def plot_confusion_matrix(y_true_idx, y_pred_idx, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, savepath=None):
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap=cmap, norm=None if not normalize else None, cbar=True)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()

def compute_confidence_interval(acc, n, z=1.96):
    """Standard normal approximation 95% CI for accuracy (proportion)."""
    p = acc
    se = math.sqrt((p*(1-p))/n)
    lower = p - z * se
    upper = p + z * se
    # clip
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return lower, upper

def evaluate_model_on_test(model, x_test, y_test):
    """Evaluate and compute top-1, top-5, confusion matrix, CI."""
    # Keras evaluate returns loss and accuracy if compiled with accuracy
    results = model.evaluate(x_test, y_test, verbose=0)
    # find accuracy from results (assume accuracy is in metrics)
    y_pred_probs = model.predict(x_test, verbose=0)
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    top1 = np.mean(y_true_idx == y_pred_idx)
    # top 5
    top5 = np.mean([y_true_idx[i] in np.argsort(y_pred_probs[i])[-5:] for i in range(len(y_true_idx))])
    lower, upper = compute_confidence_interval(top1, len(y_test))
    return {
        "top1": top1,
        "top5": top5,
        "y_true_idx": y_true_idx,
        "y_pred_idx": y_pred_idx,
        "ci_95": (lower, upper)
    }

# Model architectures (equivalent to model.py)
def build_model_a(input_shape=IMG_SHAPE, l2_reg=1e-4, dropout_rate=0.0):
    """Smaller baseline CNN: two conv+pool blocks, one dense before softmax."""
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
    """Deeper CNN with more filters and dropout — richer model."""
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

# Experiments list: two models, two hyperparameter sets each
EXPERIMENTS = [
    ("ModelA", build_model_a, [
        {"optimizer": Adam(learning_rate=1e-3), "l2_reg": 1e-4, "dropout": 0.2, "batch_size": 128, "label": "A-adam-l2-1e-4-dr0.2"},
        {"optimizer": SGD(learning_rate=0.01, momentum=0.9), "l2_reg": 5e-4, "dropout": 0.3, "batch_size": 128, "label": "A-sgd-l2-5e-4-dr0.3"}
    ]),
    ("ModelB", build_model_b, [
        {"optimizer": Adam(learning_rate=3e-4), "l2_reg": 5e-4, "dropout": 0.4, "batch_size": 128, "label": "B-adam-l2-5e-4-dr0.4"},
        {"optimizer": SGD(learning_rate=0.01, momentum=0.9), "l2_reg": 1e-3, "dropout": 0.5, "batch_size": 128, "label": "B-sgd-l2-1e-3-dr0.5"}
    ])
]

# Main experiment
def run_experiments(experiments=EXPERIMENTS, use_augmentation=True):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess()
    datagen = make_data_generator() if use_augmentation else None
    results_summary = []

    for model_name, builder_fn, hp_list in experiments:
        for hp in hp_list:
            label = hp.get("label", f"{model_name}_run")
            print("\n" + "="*80)
            print(f"Running experiment: {label}")
            print("="*80)

            # Build model with hyperparams
            l2_reg = hp.get("l2_reg", 1e-4)
            dropout = hp.get("dropout", 0.3)
            model = builder_fn(l2_reg=l2_reg, dropout_rate=dropout)
            # Compile
            optimizer = hp.get("optimizer", Adam(learning_rate=1e-3))
            model.compile(optimizer=optimizer,
                          loss="categorical_crossentropy",
                          metrics=["accuracy", TopKCategoricalAccuracy(k=5, name="top_5_accuracy")])
            model.summary()

            # Callbacks: early stopping, reduce LR on plateau, checkpoint
            cb_early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
            cb_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
            ckpt_path = os.path.join(RESULT_DIR, f"{label}_best.h5")
            cb_ckpt = ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1)

            # Fit
            batch_size = hp.get("batch_size", BATCH_SIZE)
            initial_epoch = 0
            if datagen is not None:
                # Fit with data augmentation
                datagen.fit(x_train, augment=True, seed=SEED)
                train_gen = datagen.flow(x_train, y_train, batch_size=batch_size, seed=SEED)
                steps_per_epoch = math.ceil(x_train.shape[0] / batch_size)
                history = model.fit(train_gen,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=MAX_EPOCHS,
                                    validation_data=(x_val, y_val),
                                    callbacks=[cb_early, cb_reduce, cb_ckpt],
                                    verbose=2)
            else:
                history = model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=MAX_EPOCHS,
                                    validation_data=(x_val, y_val),
                                    callbacks=[cb_early, cb_reduce, cb_ckpt],
                                    verbose=2)

            # Save history plot
            hist_plot_path = os.path.join(RESULT_DIR, f"{label}_history.png")
            plot_history(history, title=label, savepath=hist_plot_path)

            # Load best weights
            if os.path.exists(ckpt_path):
                model.load_weights(ckpt_path)

            eval_info = evaluate_model_on_test(model, x_test, y_test)
            top1 = eval_info["top1"]
            top5 = eval_info["top5"]
            lower, upper = eval_info["ci_95"]
            print(f"Test Top-1 Accuracy: {top1:.4f}")
            print(f"Test Top-5 Accuracy: {top5:.4f}")
            print(f"95% CI for Top-1 Accuracy: ({lower:.4f}, {upper:.4f})")

            # Confusion matrix
            cm_path = os.path.join(RESULT_DIR, f"{label}_confmat.png")
            plot_confusion_matrix(eval_info["y_true_idx"], eval_info["y_pred_idx"],
                                  normalize=False, title=f"{label} Confusion Matrix (counts)", savepath=cm_path)

            # Print classification report for a subset of classes
            print("\nClassification report summary (macro avg):")
            try:
                from sklearn.metrics import precision_recall_fscore_support
                p, r, f1, _ = precision_recall_fscore_support(eval_info["y_true_idx"], eval_info["y_pred_idx"], average='macro', zero_division=0)
                print(f"Precision (macro): {p:.4f}, Recall (macro): {r:.4f}, F1 (macro): {f1:.4f}")
            except Exception as e:
                print("Could not compute precision/recall/f1:", e)

            # Save model final
            model_save_path = os.path.join(RESULT_DIR, f"{label}_final.h5")
            model.save(model_save_path)
            print(f"Saved model to {model_save_path}")

            # Record summary
            results_summary.append({
                "label": label,
                "model_name": model_name,
                "top1": float(top1),
                "top5": float(top5),
                "ci_95": (float(lower), float(upper)),
                "history": history.history,
                "saved_model": model_save_path,
                "history_plot": hist_plot_path,
                "confusion_plot": cm_path
            })

    return results_summary


# Execute experiments
if __name__ == "__main__":
    print("Starting CIFAR-100 experiments. Using GPU:", tf.config.list_physical_devices('GPU'))
    summary = run_experiments(EXPERIMENTS, use_augmentation=True)

    # Print concise table of results
    print("\n\nFINAL SUMMARY OF EXPERIMENTS")
    for res in summary:
        print("-"*60)
        print(f"Label: {res['label']}")
        print(f"  Model: {res['model_name']}")
        print(f"  Top-1 Accuracy: {res['top1']:.4f}")
        print(f"  Top-5 Accuracy: {res['top5']:.4f}")
        l,u = res['ci_95']
        print(f"  95% CI: ({l:.4f}, {u:.4f})")
        print(f"  Model file: {res['saved_model']}")
        print(f"  Plots: {res['history_plot']}, {res['confusion_plot']}")
    print("\nAll results saved under:", RESULT_DIR)
