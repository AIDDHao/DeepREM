# Deep-clustering receptor-ligand reproduction

This repository contains the Python implementation, input data, and model parameter file for the receptor-ligand deep-clustering workflow.

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- pandas
- scikit-learn

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Repository layout

```text
data/
  Train.json                 training input
  Valid.json                 validation input
  Test.json                  prediction input
models/
  encoder_params_final.pth   encoder parameters for prediction
src/
  deep_cluster_model.py      encoder and decoder definitions
  train_deep_cluster.py      training entry point
  predict_neighbors.py       nearest-neighbor prediction
```

## Train

```bash
python src/train_deep_cluster.py \
  --train-json data/Train.json \
  --valid-json data/Valid.json \
  --output-dir models/retrained
```

The training command writes the encoder parameters, decoder parameters, training history, and metadata to the selected output directory.

## Predict

```bash
python src/predict_neighbors.py \
  --data data/Test.json \
  --encoder models/encoder_params_final.pth \
  --target-name <name> \
  --top-n 20 \
  --output results/top_neighbors.csv
```
