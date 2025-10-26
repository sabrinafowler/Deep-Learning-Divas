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
import seaborn as sns 
from model import EXPERIMENTS, make_data_generator
from util import load_and_preprocess, plot_confusion_matrix, plot_history, compute_confidence_interval, evaluate_model_on_test

# Reproducibility to some degree
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Config / Hyperparams
RESULT_DIR = "cifar100_exp_results"
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 100
BATCH_SIZE = 256
MAX_EPOCHS = 100  
PATIENCE = 5  

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

print("done")