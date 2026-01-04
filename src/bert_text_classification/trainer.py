"""
Training utilities.

Compilation and training steps:

- Build AdamW optimizer with warmup + decay learning-rate schedule
- Compile with BinaryCrossentropy loss + BinaryAccuracy metric
- Fit on train_ds with validation_data=val_ds
- Evaluate on test_ds

We keep these steps in classes so the pipeline is reusable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import tensorflow as tf
import tf_keras as keras

from .config import TrainingConfig


@dataclass
class TrainResult:
    """Holds training history and final evaluation metrics."""
    history: keras.callbacks.History
    test_loss: float
    test_accuracy: float


class WarmupThenDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Step-based learning rate schedule:
      - linear warmup from 0 -> init_lr for warmup_steps
      - then polynomial decay from init_lr -> end_lr for remaining steps
    """

    def __init__(
        self,
        init_lr: float,
        total_steps: int,
        warmup_steps: int,
        end_lr: float = 0.0,
        power: float = 1.0,
        name: str = "WarmupThenDecay",
    ):
        super().__init__()
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps > total_steps:
            raise ValueError("warmup_steps must be <= total_steps")

        self.init_lr = float(init_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.end_lr = float(end_lr)
        self.power = float(power)
        self.name = name

        # Decay schedule begins after warmup.
        self._decay_steps = max(1, self.total_steps - self.warmup_steps)
        self._decay = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=self._decay_steps,
            end_learning_rate=self.end_lr,
            power=self.power,
        )

    def __call__(self, step):
        with tf.name_scope(self.name):
            step_f = tf.cast(step, tf.float32)

            if self.warmup_steps == 0:
                return self._decay(step_f)

            warmup_steps_f = tf.cast(self.warmup_steps, tf.float32)

            # Warmup: lr increases linearly from 0 to init_lr
            warmup_lr = self.init_lr * (step_f / warmup_steps_f)

            # Decay: apply polynomial decay with step offset by warmup
            decay_step = tf.maximum(0.0, step_f - warmup_steps_f)
            decay_lr = self._decay(decay_step)

            return tf.where(step_f < warmup_steps_f, warmup_lr, decay_lr)

    def get_config(self) -> Dict:
        return {
            "init_lr": self.init_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "end_lr": self.end_lr,
            "power": self.power,
            "name": self.name,
        }


class BertTrainer:
    """Encapsulates compile/train/evaluate logic for the BERT sentiment classifier."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def _create_optimizer(self, train_ds: tf.data.Dataset) -> keras.optimizers.Optimizer:
        """
        Create AdamW optimizer with warmup + decay schedule.

        steps_per_epoch = cardinality(train_ds)
        total_steps = steps_per_epoch * epochs
        warmup_steps = warmup_fraction * total_steps
        """
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        if steps_per_epoch is None or int(steps_per_epoch) <= 0:
            raise ValueError(
                "Could not determine steps_per_epoch from train_ds.cardinality(). "
                "Ensure the dataset has a finite cardinality."
            )

        total_steps = int(steps_per_epoch * self.config.epochs)
        warmup_steps = int(self.config.warmup_fraction * total_steps)

        lr = WarmupThenDecay(
            init_lr=self.config.init_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            end_lr=0.0,
        )

        # Optional weight decay if present in TrainingConfig; default 0.01
        weight_decay = float(getattr(self.config, "weight_decay", 0.01))

        return keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

    def compile(self, model: keras.Model, train_ds: tf.data.Dataset) -> keras.Model:
        """Compile the model with loss/metrics/optimizer."""
        loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy(name="binary_accuracy")]
        optimizer = self._create_optimizer(train_ds)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def fit(
        self,
        model: keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        callbacks: list | None = None,
    ) -> keras.callbacks.History:
        """Train the model."""
        callbacks = callbacks or []
        return model.fit(
            x=train_ds,
            validation_data=val_ds,
            callbacks=callbacks,
            epochs=self.config.epochs,
        )

    def evaluate(self, model: keras.Model, test_ds: tf.data.Dataset) -> Tuple[float, float]:
        """Evaluate on the test set and return (loss, accuracy)."""
        loss, accuracy = model.evaluate(test_ds)
        return float(loss), float(accuracy)

    def train_and_evaluate(
        self,
        model: keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
    ) -> TrainResult:
        """End-to-end training + evaluation."""
        self.compile(model, train_ds)
        history = self.fit(model, train_ds, val_ds)
        test_loss, test_accuracy = self.evaluate(model, test_ds)
        return TrainResult(history=history, test_loss=test_loss, test_accuracy=test_accuracy)
