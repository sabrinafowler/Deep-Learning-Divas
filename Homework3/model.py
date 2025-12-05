"""
TensorFlow model definitions for sentiment analysis
"""

import tensorflow as tf
from util import VOCAB_SIZE, MAX_LENGTH


# ============================================================================
# ARCHITECTURE 1: LSTM WITH ADDITIVE ATTENTION
# ============================================================================

def create_lstm_model(embedding_dim=128, lstm_units=64, l2_reg=0.01, learning_rate=0.001):
    """
    LSTM model with AdditiveAttention and L2 regularization

    Args:
        embedding_dim: Dimension of embedding layer
        lstm_units: Number of LSTM units
        l2_reg: L2 regularization coefficient
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(MAX_LENGTH,))

    # Embedding
    x = tf.keras.layers.Embedding(
        VOCAB_SIZE,
        embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(inputs)

    # LSTM layer - return sequences for attention
    lstm_out = tf.keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)

    # Additive Attention
    query = lstm_out[:, -1:, :]  # Last timestep as query
    attention = tf.keras.layers.AdditiveAttention()([query, lstm_out])
    attention = tf.keras.layers.Flatten()(attention)

    # Output layer
    outputs = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(attention)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# ARCHITECTURE 2: TRANSFORMER
# ============================================================================

class TransformerBlock(tf.keras.layers.Layer):
    """Simple Transformer block - optimized for speed"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads  # More efficient key_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_transformer_model(embedding_dim=128, num_heads=4, ff_dim=128, dropout_rate=0.3, learning_rate=0.001):
    """
    Transformer model with Dropout regularization

    Args:
        embedding_dim: Dimension of embedding layer
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward network
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(MAX_LENGTH,))

    # Embedding
    x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Transformer block
    x = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout_rate)(x)

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
