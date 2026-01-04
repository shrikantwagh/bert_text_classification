"""
Vertex AI helpers (optional).

This module converts Vertex AI portion into reusable Python functions.

Logic (roughly):
- Determine PROJECT and BUCKET
- Create a GCS bucket if needed
- Copy local export folder to gs://BUCKET/MODEL_DISPLAYNAME
- Upload to Vertex AI Model registry with TF2 prebuilt serving container
- Deploy to an endpoint
- Call endpoint.predict(instances=[{"text": ["..."]}, ...])
- Cleanup: undeploy and delete endpoint

Here we:
- Use google-cloud-storage to ensure the bucket exists and upload artifacts
- Use google-cloud-aiplatform for model upload / endpoint deploy / predict / cleanup
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from google.cloud import aiplatform
from google.cloud import storage

from .config import VertexConfig


def _upload_directory_to_gcs(local_dir: Path, bucket_name: str, gcs_prefix: str) -> str:
    """
    Upload a local directory recursively to a GCS bucket.

    Args:
        local_dir: Directory containing the SavedModel artifacts.
        bucket_name: Target GCS bucket.
        gcs_prefix: Prefix under the bucket (e.g. "classification-bert-20260101...").

    Returns:
        The artifact URI: "gs://<bucket>/<gcs_prefix>"
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create bucket if missing.
    if not bucket.exists(client):
        # Location is not set here; caller should create bucket explicitly if they care about region.
        bucket = client.create_bucket(bucket_name)

    # Upload all files under the directory.
    for path in local_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(local_dir)
        blob_name = f"{gcs_prefix}/{rel.as_posix()}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(path))

    return f"gs://{bucket_name}/{gcs_prefix}"


@dataclass
class VertexDeploymentResult:
    model_resource_name: str
    endpoint_resource_name: str
    endpoint_id: str


class VertexAIManager:
    """Upload a SavedModel to Vertex AI and deploy it to an endpoint."""

    def __init__(self, config: VertexConfig) -> None:
        self.config = config

        # Initialize the Vertex AI client context.
        aiplatform.init(project=self.config.project, location=self.config.region)

    def ensure_bucket(self) -> None:
        """Ensure the bucket exists in the configured region."""
        bucket_name = self.config.resolved_bucket()
        client = storage.Client(project=self.config.project)
        bucket = client.bucket(bucket_name)
        if bucket.exists(client):
            return

        # Create bucket in the specified region gsutil mb -l ${REGION} ...).
        client.create_bucket(bucket, location=self.config.region)

    def upload_model(
        self,
        export_path: Path,
        model_display_name: str,
    ) -> aiplatform.Model:
        """
        Upload the SavedModel artifacts to GCS and register a Vertex AI model.

        Args:
            export_path: Local SavedModel directory (timestamp folder).
            model_display_name: Display name used in Vertex AI and also as GCS prefix.

        Returns:
            aiplatform.Model object representing the uploaded model.
        """
        self.ensure_bucket()
        bucket_name = self.config.resolved_bucket()

        # Upload SavedModel folder to GCS under gs://bucket/model_display_name/...
        artifact_uri = _upload_directory_to_gcs(
            local_dir=export_path,
            bucket_name=bucket_name,
            gcs_prefix=model_display_name,
        )

        uploaded_model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=self.config.serving_container_image_uri,
        )
        return uploaded_model

    def deploy_model(
        self,
        model: aiplatform.Model,
        machine_type: str = "n1-standard-4",
    ) -> VertexDeploymentResult:
        """
        Deploy an uploaded model to a new endpoint.

        Returns:
            VertexDeploymentResult with resource names and endpoint id.
        """
        endpoint = model.deploy(
            machine_type=machine_type,
            accelerator_type=None,
            accelerator_count=None,
        )
        # endpoint.resource_name ends with ".../endpoints/<id>"
        endpoint_id = endpoint.resource_name.split("/")[-1]

        return VertexDeploymentResult(
            model_resource_name=model.resource_name,
            endpoint_resource_name=endpoint.resource_name,
            endpoint_id=endpoint_id,
        )

    def predict(
        self,
        endpoint_id: str,
        texts: List[str],
    ):
        """
        Make an online prediction.

        The instances of shape:
            {"text": ["..."]}

        """
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
        instances = [{"text": [t]} for t in texts]
        return endpoint.predict(instances=instances)

    def cleanup_endpoint(self, endpoint_id: str) -> None:
        """Undeploy all models from an endpoint and delete the endpoint."""
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
        endpoint.undeploy_all()
        endpoint.delete()
