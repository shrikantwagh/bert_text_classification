"""
Model definition.

function:

    def build_classifier_model(dropout_rate=0.1):
        text_input = Input(shape=(), dtype=tf.string, name="text")
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")
        outputs = encoder(encoder_inputs)
        net = outputs["pooled_output"]
        net = Dropout(dropout_rate)(net)
        net = Dense(1, activation="sigmoid", name="classifier")(net)

Key points:
- Input is raw string text.
- TF Hub preprocessing converts text -> token ids + masks + segment ids.
- TF Hub encoder produces BERT outputs, we use pooled_output for classification.
- A sigmoid head produces a scalar score in [0,1] (binary sentiment).
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_hub as hub

from .config import HubModelsConfig


class BertTextClassifierBuilder:
    """Builds a Keras model that includes TF Hub preprocess + encoder + classification head."""

    def __init__(self, hub_models: HubModelsConfig) -> None:
        self.hub_models = hub_models

    def build(self, dropout_rate: float = 0.1, trainable_encoder: bool = True) -> tf.keras.Model:
        """
        Build the classifier model.

        Args:
            dropout_rate: Dropout applied on top of BERT pooled output.
            trainable_encoder: Whether to fine-tune the BERT encoder weights.

        Returns:
            A compiled-uncompiled `tf.keras.Model` ready to be compiled and trained.
        """
        # Raw text input (a scalar string per example).
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")

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
        net = tf.keras.layers.Dropout(dropout_rate)(net)

        # Single sigmoid unit for binary sentiment.
        net = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(net)

        return tf.keras.Model(inputs=text_input, outputs=net)
