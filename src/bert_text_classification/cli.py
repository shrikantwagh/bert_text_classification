"""
Command-line interface for the project.

Run:
  python -m bert_text_classification.cli --help

This provides subcommands:
- train: download IMDB, train, evaluate, export
- predict-local: load SavedModel from disk and predict
- vertex-upload-and-deploy: upload SavedModel to Vertex AI and deploy
- vertex-predict: send online prediction requests
- vertex-cleanup: undeploy and delete an endpoint
"""

from __future__ import annotations

import warnings

import warnings

# Google Cloud Python version support warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.api_core\._python_version_support",
)

# tensorflow_hub / pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import argparse
from pathlib import Path
from typing import List, Optional

import tensorflow as tf

from .config import DataConfig, ExportConfig, HubModelsConfig, TrainingConfig, VertexConfig
from .data import ImdbDataModule
from .exporter import ModelExporter, SavedModelPredictor
from .model import BertTextClassifierBuilder
from .trainer import BertTrainer
from .vertex import VertexAIManager


def _cmd_train(args: argparse.Namespace) -> None:
    # ----- Config -----
    data_cfg = DataConfig(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        seed=args.seed,
        validation_split=args.validation_split,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        init_lr=args.init_lr,
        warmup_fraction=args.warmup_fraction,
        dropout_rate=args.dropout_rate,
    )
    export_cfg = ExportConfig(export_dir=Path(args.export_dir), dataset_name="imdb")
    hub_cfg = HubModelsConfig(
        encoder_handle=args.hub_encoder,
        preprocess_handle=args.hub_preprocess,
    )

    # ----- Data -----
    data = ImdbDataModule(data_cfg).load()

    # Optional: show a couple of sample reviews.
    if args.print_samples:
        for text_batch, label_batch in data.train_ds.take(1):
            for i in range(min(3, int(text_batch.shape[0]))):
                print(f"Review: {text_batch.numpy()[i]}")
                label = int(label_batch.numpy()[i])
                print(f"Label : {label} ({data.class_names[label]})")
                print()

    # ----- Model -----
    model = BertTextClassifierBuilder(hub_cfg).build(
        dropout_rate=train_cfg.dropout_rate,
        trainable_encoder=True,
    )

    # Quick smoke-test inference.
    smoke = model(tf.constant(["this is such an amazing movie!"]))
    print("Smoke test model output:", smoke.numpy().reshape(-1).tolist())

    # ----- Train & Evaluate -----
    trainer = BertTrainer(train_cfg)
    result = trainer.train_and_evaluate(
        model=model,
        train_ds=data.train_ds,
        val_ds=data.val_ds,
        test_ds=data.test_ds,
    )

    print(f"Test Loss: {result.test_loss:.6f}")
    print(f"Test Accuracy: {result.test_accuracy:.6f}")

    # ----- Export -----
    exporter = ModelExporter(export_cfg)
    export_result = exporter.export(model)
    print("Exported SavedModel to:", str(export_result.export_path))


def _cmd_predict_local(args: argparse.Namespace) -> None:
    predictor = SavedModelPredictor(Path(args.export_path))
    scores = predictor.predict_scores(args.text)
    for t, s in zip(args.text, scores):
        print(f"input: {t:<60} score: {s:.6f}")


def _cmd_vertex_upload_and_deploy(args: argparse.Namespace) -> None:
    vertex_cfg = VertexConfig(
        project=args.project,
        region=args.region,
        bucket=args.bucket,
        serving_container_image_uri=args.serving_container_image_uri,
    )
    manager = VertexAIManager(vertex_cfg)

    model = manager.upload_model(
        export_path=Path(args.export_path),
        model_display_name=args.model_display_name,
    )
    deployment = manager.deploy_model(model, machine_type=args.machine_type)

    print("Uploaded model:", deployment.model_resource_name)
    print("Deployed endpoint:", deployment.endpoint_resource_name)
    print("Endpoint ID:", deployment.endpoint_id)


def _cmd_vertex_predict(args: argparse.Namespace) -> None:
    vertex_cfg = VertexConfig(
        project=args.project,
        region=args.region,
        bucket=args.bucket,
        serving_container_image_uri=args.serving_container_image_uri,
    )
    manager = VertexAIManager(vertex_cfg)
    response = manager.predict(endpoint_id=args.endpoint_id, texts=args.text)
    print("predictions:", response.predictions)


def _cmd_vertex_cleanup(args: argparse.Namespace) -> None:
    vertex_cfg = VertexConfig(
        project=args.project,
        region=args.region,
        bucket=args.bucket,
        serving_container_image_uri=args.serving_container_image_uri,
    )
    manager = VertexAIManager(vertex_cfg)
    manager.cleanup_endpoint(endpoint_id=args.endpoint_id)
    print("Cleaned up endpoint:", args.endpoint_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bert_text_classification",
        description="BERT text classification",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Download data, train/evaluate, and export a SavedModel.")
    p_train.add_argument("--data-dir", default="./data", help="Directory for dataset download/cache.")
    p_train.add_argument("--export-dir", default="./exports", help="Directory for SavedModel exports.")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--validation-split", type=float, default=0.2)
    p_train.add_argument("--init-lr", type=float, default=3e-5)
    p_train.add_argument("--warmup-fraction", type=float, default=0.1)
    p_train.add_argument("--dropout-rate", type=float, default=0.15)


    p_train.add_argument(
        "--hub-encoder",
        default=HubModelsConfig().encoder_handle,
        help="TF Hub handle for BERT encoder.",
    )
    p_train.add_argument(
        "--hub-preprocess",
        default=HubModelsConfig().preprocess_handle,
        help="TF Hub handle for preprocessing model.",
    )
    p_train.add_argument("--print-samples", action="store_true", help="Print a few sample reviews/labels.")

    p_train.set_defaults(func=_cmd_train)

    # ---- predict-local ----
    p_pred = sub.add_parser("predict-local", help="Run inference using an exported SavedModel on disk.")
    p_pred.add_argument("--export-path", required=True, help="Path to exported SavedModel folder.")
    p_pred.add_argument("--text", action="append", required=True, help="Input text (repeatable).")
    p_pred.set_defaults(func=_cmd_predict_local)

    # ---- vertex-upload-and-deploy ----
    p_vud = sub.add_parser("vertex-upload-and-deploy", help="Upload SavedModel to Vertex AI and deploy.")
    p_vud.add_argument("--export-path", required=True, help="Local SavedModel export path (timestamp folder).")
    p_vud.add_argument("--project", required=True, help="GCP project id.")
    p_vud.add_argument("--region", default="us-central1")
    p_vud.add_argument("--bucket", default=None, help="GCS bucket name (defaults to project id).")
    p_vud.add_argument("--model-display-name", required=True, help="Vertex model display name and GCS prefix.")
    p_vud.add_argument("--machine-type", default="n1-standard-4")
    p_vud.add_argument(
        "--serving-container-image-uri",
        default="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    )
    p_vud.set_defaults(func=_cmd_vertex_upload_and_deploy)

    # ---- vertex-predict ----
    p_vp = sub.add_parser("vertex-predict", help="Run online prediction against a Vertex endpoint.")
    p_vp.add_argument("--project", required=True, help="GCP project id.")
    p_vp.add_argument("--region", default="us-central1")
    p_vp.add_argument("--bucket", default=None)
    p_vp.add_argument("--endpoint-id", required=True, help="Endpoint id (the numeric part).")
    p_vp.add_argument("--text", action="append", required=True, help="Input text (repeatable).")
    p_vp.add_argument(
        "--serving-container-image-uri",
        default="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    )
    p_vp.set_defaults(func=_cmd_vertex_predict)

    # ---- vertex-cleanup ----
    p_vc = sub.add_parser("vertex-cleanup", help="Undeploy all and delete a Vertex endpoint.")
    p_vc.add_argument("--project", required=True, help="GCP project id.")
    p_vc.add_argument("--region", default="us-central1")
    p_vc.add_argument("--bucket", default=None)
    p_vc.add_argument("--endpoint-id", required=True)
    p_vc.add_argument(
        "--serving-container-image-uri",
        default="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    )
    p_vc.set_defaults(func=_cmd_vertex_cleanup)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
