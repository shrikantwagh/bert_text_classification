"""
Training utilities.

This is compilation and training steps:

- Build optimizer using `official.nlp.optimization.create_optimizer` (AdamW + warmup)
- Compile with BinaryCrossentropy loss + BinaryAccuracy metric
- Fit on train_ds with validation_data=val_ds
- Evaluate on test_ds

We keep these steps in classes so the pipeline is reusable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import tensorflow as tf
from official.nlp import optimization  # AdamW optimizer helper

from .config import TrainingConfig


@dataclass
class TrainResult:
    """Holds training history and final evaluation metrics."""
    history: tf.keras.callbacks.History
    test_loss: float
    test_accuracy: float


class BertTrainer:
    """Encapsulates compile/train/evaluate logic for the BERT sentiment classifier."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def _create_optimizer(self, train_ds: tf.data.Dataset) -> tf.keras.optimizers.Optimizer:
        """
        Create AdamW optimizer with warmup.

        computed:
            steps_per_epoch = cardinality(train_ds)
            num_train_steps = steps_per_epoch * epochs
            num_warmup_steps = int(0.1 * num_train_steps)

        Args:
            train_ds: Training dataset (used to compute number of steps per epoch).

        Returns:
            A `tf.keras.optimizers.Optimizer` instance.
        """
        # NOTE: cardinality() will be known for datasets created via text_dataset_from_directory.
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        if steps_per_epoch <= 0:
            raise ValueError(
                "Could not determine steps_per_epoch from train_ds.cardinality(). "
                "Ensure the dataset has a finite cardinality."
            )

        num_train_steps = int(steps_per_epoch * self.config.epochs)
        num_warmup_steps = int(self.config.warmup_fraction * num_train_steps)

        optimizer = optimization.create_optimizer(
            init_lr=self.config.init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type="adamw",
        )
        return optimizer

    def compile(self, model: tf.keras.Model, train_ds: tf.data.Dataset) -> tf.keras.Model:
        """
        Compile the model with loss/metrics/optimizer.

        Returns:
            The same model (compiled).
        """
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]

        optimizer = self._create_optimizer(train_ds)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def fit(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ) -> tf.keras.callbacks.History:
        """Train the model."""
        return model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
        )

    def evaluate(self, model: tf.keras.Model, test_ds: tf.data.Dataset) -> Tuple[float, float]:
        """Evaluate on the test set and return (loss, accuracy)."""
        loss, accuracy = model.evaluate(test_ds)
        return float(loss), float(accuracy)

    def train_and_evaluate(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
    ) -> TrainResult:
        """
        End-to-end training + evaluation.

        Returns:
            TrainResult with training history and test metrics.
        """
        self.compile(model, train_ds)
        history = self.fit(model, train_ds, val_ds)
        test_loss, test_accuracy = self.evaluate(model, test_ds)
        return TrainResult(history=history, test_loss=test_loss, test_accuracy=test_accuracy)
