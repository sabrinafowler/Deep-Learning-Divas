"""
Main training loop for sentiment analysis models
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import prepare_data
from model import create_lstm_model, create_transformer_model


def train_model(model, name, train_ds, val_ds, epochs=15):
    """
    Train model with early stopping

    Args:
        model: Keras model to train
        name: Name of the model for logging
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Maximum number of epochs

    Returns:
        Training history
    """
    print(f"\nTraining: {name}")
    print("=" * 60)

    # Early stopping with patience=5
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    print(f"\nBest val accuracy: {max(history.history['val_accuracy']):.4f}")
    return history


def plot_training_results(histories, names):
    """
    Plot training and validation accuracy curves

    Args:
        histories: List of training histories
        names: List of model names
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for history, name in zip(histories, names):
        axes[0].plot(history.history['accuracy'], label=name, alpha=0.7)
        axes[1].plot(history.history['val_accuracy'], label=name, alpha=0.7)

    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training curves saved!")


def evaluate_models(models, names, test_ds):
    """
    Evaluate models on test set and plot results

    Args:
        models: List of trained models
        names: List of model names
        test_ds: Test dataset

    Returns:
        List of result dictionaries
    """
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    results = []
    for model, name in zip(models, names):
        loss, acc = model.evaluate(test_ds, verbose=0)
        results.append({'name': name, 'loss': loss, 'accuracy': acc})
        print(f"{name:25s} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # Plot test results
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    accuracies = [r['accuracy'] for r in results]
    ax.bar(range(len(results)), accuracies, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Set Accuracy')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nEvaluation complete!")
    return results


def main():
    """
    Main training pipeline
    """
    # Load and preprocess data
    train_ds, val_ds, test_ds = prepare_data()

    # ========================================================================
    # EXPERIMENT 1: LSTM - CONFIGURATION 1
    # ========================================================================
    print("\n" + "=" * 60)
    print("LSTM WITH ADDITIVE ATTENTION - CONFIG 1")
    print("=" * 60)

    lstm_model_1 = create_lstm_model(
        embedding_dim=128,
        lstm_units=64,
        l2_reg=0.0001,
        learning_rate=0.001
    )

    history_lstm_1 = train_model(lstm_model_1, "LSTM Config 1", train_ds, val_ds)

    # ========================================================================
    # EXPERIMENT 2: LSTM - CONFIGURATION 2
    # ========================================================================
    print("\n" + "=" * 60)
    print("LSTM WITH ADDITIVE ATTENTION - CONFIG 2")
    print("=" * 60)

    lstm_model_2 = create_lstm_model(
        embedding_dim=256,
        lstm_units=128,
        l2_reg=0.001,
        learning_rate=0.0005
    )

    history_lstm_2 = train_model(lstm_model_2, "LSTM Config 2", train_ds, val_ds)

    # ========================================================================
    # EXPERIMENT 3: TRANSFORMER - CONFIGURATION 1
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRANSFORMER - CONFIG 1")
    print("=" * 60)

    transformer_model_1 = create_transformer_model(
        embedding_dim=128,
        num_heads=4,
        ff_dim=128,
        dropout_rate=0.3,
        learning_rate=0.001
    )

    history_transformer_1 = train_model(transformer_model_1, "Transformer Config 1", train_ds, val_ds)

    # ========================================================================
    # EXPERIMENT 4: TRANSFORMER - CONFIGURATION 2
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRANSFORMER - CONFIG 2")
    print("=" * 60)

    transformer_model_2 = create_transformer_model(
        embedding_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout_rate=0.2,
        learning_rate=0.0005
    )

    history_transformer_2 = train_model(transformer_model_2, "Transformer Config 2", train_ds, val_ds)

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    histories = [history_lstm_1, history_lstm_2, history_transformer_1, history_transformer_2]
    names = ['LSTM Config 1', 'LSTM Config 2', 'Transformer Config 1', 'Transformer Config 2']

    plot_training_results(histories, names)

    # ========================================================================
    # EVALUATION
    # ========================================================================
    models = [lstm_model_1, lstm_model_2, transformer_model_1, transformer_model_2]
    results = evaluate_models(models, names, test_ds)

    return results


if __name__ == "__main__":
    main()
