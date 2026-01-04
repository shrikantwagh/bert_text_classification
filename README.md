# BERT Text Classification


It trains a sentiment classifier on the **Stanford IMDB** dataset using a **TensorFlow Hub Small BERT** encoder + preprocessing model, exports a SavedModel, runs local inference, and (optionally) uploads/deploys the model on **Vertex AI** for online prediction.

## 1) Quickstart (local training + local inference)

### Prerequisites
- Python 3.10+ (recommended)
- `pip` / `venv`
- Enough disk space for the IMDB dataset (~80MB compressed, larger uncompressed)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train + export
This will download IMDB to `./data/aclImdb`, train, and export a SavedModel under `./exports/imdb_bert/<timestamp>/`:
```bash
python -m bert_text_classification.cli train \
  --data-dir ./data \
  --export-dir ./exports \
  --epochs 5
```

### Run local prediction from the exported model
```bash
python -m bert_text_classification.cli predict-local \
  --export-path ./exports/imdb_bert/<timestamp> \
  --text "this is such an amazing movie!" \
  --text "the movie was meh."
```

## 2) Optional: Vertex AI upload + deploy + predict

> These steps require:
> - a Google Cloud project with billing enabled
> - `gcloud` authenticated (`gcloud auth login`)
> - permissions for Vertex AI and GCS
> - the Python package `google-cloud-aiplatform` (already in `requirements.txt`)

### Upload + deploy
```bash
python -m bert_text_classification.cli vertex-upload-and-deploy \
  --export-path ./exports/imdb_bert/<timestamp> \
  --project <YOUR_GCP_PROJECT_ID> \
  --region us-central1 \
  --model-display-name classification-bert-$(date +%Y%m%d%H%M%S)
```

The command will:
1. Ensure a GCS bucket exists (by default, it uses a bucket named the same as your project id)
2. Copy the SavedModel artifacts to `gs://<bucket>/<model-display-name>/`
3. Upload the model to Vertex AI using the TensorFlow 2 CPU prebuilt serving container
4. Deploy the model to an endpoint

### Online predict
```bash
python -m bert_text_classification.cli vertex-predict \
  --project <YOUR_GCP_PROJECT_ID> \
  --region us-central1 \
  --endpoint-id <ENDPOINT_ID> \
  --text "I loved the movie and highly recommend it" \
  --text "I hated the movie"
```

### Cleanup (undeploy + delete endpoint)
```bash
python -m bert_text_classification.cli vertex-cleanup \
  --project <YOUR_GCP_PROJECT_ID> \
  --region us-central1 \
  --endpoint-id <ENDPOINT_ID>
```

## 3) Notes

- The default BERT models:
  - Encoder: `tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1`
  - Preprocess: `tensorflow/bert_en_uncased_preprocess/3`
- Training uses an AdamW optimizer from `tf-models-official` with linear warmup.
- The exported SavedModel signature is compatible with Vertex AI TensorFlow 2 prebuilt containers.
- If you run on macOS and TensorFlow installation is tricky, consider using:
  - a Linux environment (Docker / WSL), or
  - Apple silicon builds (if applicable).

## 4) Repo layout

- `src/bert_text_classification/`
  - `config.py` — configuration dataclasses
  - `data.py` — dataset download + tf.data pipelines
  - `model.py` — model builder (TF Hub preprocess + encoder + classifier head)
  - `trainer.py` — optimizer creation, compile/train/evaluate
  - `exporter.py` — SavedModel export + local inference helper
  - `vertex.py` — Vertex AI upload/deploy/predict/cleanup helpers
  - `cli.py` — command-line entrypoint

---

### One-liner help
```bash
python -m bert_text_classification.cli --help
```
