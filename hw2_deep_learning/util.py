import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
import seaborn as sns 

NUM_CLASSES = 100

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
    """Standard normal approximation 95% CI for accuracy"""
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