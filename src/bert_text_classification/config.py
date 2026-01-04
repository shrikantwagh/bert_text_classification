"""
Configuration objects for the BERT text classification pipeline.

This project is a class-based conversion 

This use:
- TensorFlow 2.x
- TensorFlow Hub (preprocess + Small BERT encoder)
- tf-models-official AdamW optimizer helper
- (Optional) Vertex AI upload/deploy for online predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HubModelsConfig:
    """
    TensorFlow Hub model handles.
    """
    # Small BERT encoder to fine-tune.
    encoder_handle: str = (
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    )

    # Matching preprocessing model (tokenization, masks, segment IDs).
    preprocess_handle: str = (
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )


@dataclass(frozen=True)
class DataConfig:
    """
    Dataset configuration.

    downloads Stanford's IMDB dataset and builds:
    - training dataset (80% of train split)
    - validation dataset (20% of train split)
    - test dataset (the provided test split)
    """
    # A directory that will contain the downloaded/unpacked dataset.
    data_dir: Path = Path("./data")

    # Batch size used by text_dataset_from_directory.
    batch_size: int = 32

    # Random seed for validation split.
    seed: int = 42

    # Validation fraction from the train split.
    validation_split: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    """
    Training hyperparameters.

   Uses AdamW with warmup using `official.nlp.optimization`.
    """
    epochs: int = 5
    init_lr: float = 3e-5
    warmup_fraction: float = 0.1

    # Dropout on top of BERT pooled output before the classifier head.
    dropout_rate: float = 0.15


@dataclass(frozen=True)
class ExportConfig:
    """
    SavedModel export configuration.

    """
    export_dir: Path = Path("./exports")
    dataset_name: str = "imdb"

    def export_base_path(self) -> Path:
        """Return the base directory for exports for this dataset."""
        # Example: ./exports/imdb_bert
        return self.export_dir / f"{self.dataset_name}_bert"


@dataclass(frozen=True)
class VertexConfig:
    """
    Vertex AI configuration (optional).

    If you don't plan to upload/deploy on Vertex AI, you can ignore this.
    """
    project: str
    region: str = "us-central1"

    # use a TensorFlow 2 CPU prebuilt serving container.
    serving_container_image_uri: str = (
        "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
    )

    # Default bucket naming  BUCKET = PROJECT
    bucket: str | None = None

    def resolved_bucket(self) -> str:
        """Return the bucket name to use (defaults to the project id)."""
        return self.bucket or self.project
