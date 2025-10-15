
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

from util import load_datasets
from model import build_model

# Load datasets
train_ds, val_ds, test_ds = load_datasets(batch_size=32)

def train_and_evaluate(layers, lr, use_reg, run_name):
    
    print(f"\n***** Training {run_name} *****")
    
    # compile model with given architecture
    model = build_model(layers, learning_rate=lr, use_regularizer=use_reg)

    #early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # train model
    # also collect loss/ accuracy cuvres
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=2,
        callbacks=[early_stop]
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"{run_name} Validation Accuracy: {val_acc:.4f}")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"{run_name} Test Accuracy: {test_acc:.4f}")

    # confusion matrix
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for x, y in test_ds], axis=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {run_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # confidence intervals
    n_test = len(y_true)
    p_error = 1 - test_acc
    ci = 1.96 * math.sqrt(p_error * (1 - p_error) / n_test)
    print(f"Generalization error: {p_error:.3f} ± {ci:.3f} (95% CI)")

    return model, history

# hidden layers
architectures = [[128], [256, 128]]

#Learning rates
learning_rates = [0.001, 0.0005]

# regularizer?
regularizers = [False, True]

# store model objects
results = {}

# run 8 total experiments
for arch in architectures:
    for lr in learning_rates:
        for reg in regularizers:
            run_name = f"Architecture: {arch}, Learning rate: {lr}, Regularize: {reg}"
            model, history = train_and_evaluate(arch, lr, reg, run_name)
            results[run_name] = model

# evaluate on test set
print("\n***** Test Set Evaluation *****")
for run_name, model in results.items():
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"{run_name} Test Accuracy: {test_acc:.4f}")
