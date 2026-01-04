"""
Export + local inference helpers.

exported a SavedModel with:
    classifier_model.save(EXPORT_PATH, include_optimizer=False)

Then reloaded with:
    reloaded_model = tf.saved_model.load(EXPORT_PATH)
    serving_results = reloaded_model.signatures["serving_default"](tf.constant(examples))
    serving_results = serving_results["classifier"]

This module provides equivalent helpers, wrapped in a class.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import tensorflow as tf

from .config import ExportConfig


@dataclass
class ExportResult:
    export_path: Path


class ModelExporter:
    """Exports a Keras model as a TensorFlow SavedModel."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config

    def export(self, model: tf.keras.Model) -> ExportResult:
        """
        Export the model to a timestamped folder.

        Returns:
            ExportResult containing the export_path.
        """
        base_path = self.config.export_base_path()
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        export_path = base_path / timestamp
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # include_optimizer=False .
        model.save(str(export_path), include_optimizer=False)
        return ExportResult(export_path=export_path)


class SavedModelPredictor:
    """
    Loads a SavedModel from disk and runs inference using the `serving_default` signature.

    The TF Hub model expects raw strings. The exported signature typically accepts a tensor of dtype string.
    """

    def __init__(self, export_path: Path) -> None:
        self.export_path = export_path
        self._loaded = tf.saved_model.load(str(export_path))
        self._infer = self._loaded.signatures["serving_default"]

    def predict_scores(self, texts: List[str]) -> List[float]:
        """
        Predict sentiment scores for input texts.

        Args:
            texts: List of input strings.

        Returns:
            List of floats in [0,1], where larger means more positive sentiment.
        """
    
        outputs = self._infer(tf.constant(texts))

        # The output key name is "classifier".
        scores = outputs["classifier"].numpy().reshape(-1).tolist()
        return [float(s) for s in scores]
