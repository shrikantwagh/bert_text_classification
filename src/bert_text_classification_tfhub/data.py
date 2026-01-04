"""
Dataset utilities.


1) Download Stanford IMDB sentiment dataset:
   https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

2) Remove the "unsup" folder from train split (unused)

3) Create TensorFlow datasets using `tf.keras.preprocessing.text_dataset_from_directory`,
   with an 80/20 train/validation split, plus the provided test split.

The functions here return `tf.data.Dataset` objects ready for model.fit().
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tensorflow as tf

from .config import DataConfig


IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@dataclass
class ImdbDatasets:
    """A convenience container for the three datasets and the class names."""
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: list[str]


class ImdbDataModule:
    """
    Responsible for downloading and preparing the IMDB dataset.

    Notes:
      In this repo, we default to `./data/`, but you can point it elsewhere via CLI flags.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    @property
    def aclimdb_root(self) -> Path:
        """Path to the dataset root folder (the folder containing `train/` and `test/`)."""
        return self.config.data_dir / "aclImdb"

    def download_if_needed(self) -> Path:
        """
        Download and unpack the IMDB dataset if not already present.

        Returns:
            Path to the root dataset directory (…/aclImdb).
        """
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # tf.keras.utils.get_file will cache/download to the directory you specify.
        # `untar=True` will unpack the .tar.gz and return the path of the archive file.
        archive_path = tf.keras.utils.get_file(
            fname="aclImdb_v1.tar.gz",
            origin=IMDB_URL,
            untar=True,
            cache_dir=str(self.config.data_dir),
            cache_subdir="",
        )

        # The unpacked dataset directory is placed alongside the archive.
        # Example: <data_dir>/aclImdb
        dataset_dir = Path(os.path.dirname(archive_path)) / "aclImdb"

        # Remove the "unsup" directory.
        remove_dir = dataset_dir / "train" / "unsup"
        if remove_dir.exists():
            shutil.rmtree(remove_dir)

        return dataset_dir

    def load(self) -> ImdbDatasets:
        """
        Create train/val/test tf.data pipelines from the local IMDB folder.

        Returns:
            ImdbDatasets with train, val, test and class_names.
        """
        dataset_dir = self.download_if_needed()

        autotune = tf.data.AUTOTUNE
        batch_size = self.config.batch_size
        seed = self.config.seed
        val_split = self.config.validation_split

        # text_dataset_from_directory on:
        #   path + "aclImdb/train"
        train_path = dataset_dir / "train"
        test_path = dataset_dir / "test"

        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            str(train_path),
            batch_size=batch_size,
            validation_split=val_split,
            subset="training",
            seed=seed,
        )

        class_names = list(raw_train_ds.class_names)

        # Cache + prefetch improves pipeline throughput.
        train_ds = raw_train_ds.cache().prefetch(buffer_size=autotune)

        val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            str(train_path),
            batch_size=batch_size,
            validation_split=val_split,
            subset="validation",
            seed=seed,
        )
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            str(test_path),
            batch_size=batch_size,
        )
        test_ds = test_ds.cache().prefetch(buffer_size=autotune)

        return ImdbDatasets(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            class_names=class_names,
        )
