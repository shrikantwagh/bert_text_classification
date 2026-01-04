"""
Model definition.

This model uses TF Hub modules for:
- preprocessing: raw text -> token ids + masks + segment ids
- encoder: BERT outputs (we use pooled_output)

Important compatibility notes (macOS + TF 2.16 + Py3.10):
- TF Hub BERT preprocessors depend on TensorFlow Text ops (e.g., CaseFoldUTF8).
  Importing `tensorflow_text` registers those ops.
- To avoid Keras 3 / TF Hub tracing issues, we build the Keras graph using `tf_keras`
  (the Keras 2 API compatible with TensorFlow).
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_hub as hub

# IMPORTANT: registers TF Text ops used by TF Hub preprocess models (e.g., CaseFoldUTF8)
import tensorflow_text as text  # noqa: F401

# IMPORTANT: use tf_keras (Keras 2) to avoid KerasTensor->NumPy issues with TF Hub
import tf_keras as keras

from .config import HubModelsConfig


class BertTextClassifierBuilder:
    """Builds a Keras model that includes TF Hub preprocess + encoder + classification head."""

    def __init__(self, hub_models: HubModelsConfig) -> None:
        self.hub_models = hub_models

    def build(self, dropout_rate: float = 0.1, trainable_encoder: bool = True) -> keras.Model:
        """
        Build the classifier model.

        Args:
            dropout_rate: Dropout applied on top of BERT pooled output.
            trainable_encoder: Whether to fine-tune the BERT encoder weights.

        Returns:
            A `tf_keras.Model` ready to be compiled and trained.
        """
        # Raw text input (a scalar string per example).
        text_input = keras.layers.Input(shape=(), dtype=tf.string, name="text")

        # Preprocessing model from TF Hub (tokenizes, builds masks, etc.).
        preprocessing_layer = hub.KerasLayer(
            self.hub_models.preprocess_handle,
            name="preprocessing",
        )
        encoder_inputs = preprocessing_layer(text_input)

        # BERT encoder from TF Hub.
        encoder = hub.KerasLayer(
            self.hub_models.encoder_handle,
            trainable=trainable_encoder,
            name="BERT_encoder",
        )
        outputs = encoder(encoder_inputs)

        # For classification tasks, the pooled_output is commonly used.
        net = outputs["pooled_output"]

        # A dropout layer helps regularization.
        net = keras.layers.Dropout(dropout_rate)(net)

        # Single sigmoid unit for binary sentiment.
        net = keras.layers.Dense(1, activation="sigmoid", name="classifier")(net)

        return keras.Model(inputs=text_input, outputs=net)
