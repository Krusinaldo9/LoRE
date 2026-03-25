# Adaptive and Robust Graph Embeddings via Local Self-Reconstruction

This repository contains the official supplementary code for the locally self-reconstructive graph embedding model

**LoRE: Adaptive and Robust Graph Embeddings via Local Self-Reconstruction**

LoRE is a graph embedding framework that introduces local self-reconstruction as a new self-supervised training paradigm for graph neural networks. Instead of supervising embeddings through edge prediction, LoRE learns node representations by reconstructing base embeddings from their local neighborhood context. This node-centric formulation enforces structural consistency among equivalent nodes, avoids reliance on negative edge sampling, and enables efficient adaptation when graphs evolve.

---

## Installation

Set up the environment with:

```bash
conda env create -f environment.yml
conda activate lore
```

This installs all dependencies needed for serving LoRE via the Flask server. The environment is CUDA-ready but also runs on CPU (with increased runtimes).

---

## Data Format

Each dataset is provided as a zipped JSON file, for example:

```bash
data/<dataset>/graph.json.zip
```

The archive contains:

- **graph.json**: nodes, relations (a single relation for undirected graphs), and edges as directed triples `(v, u, m)`. For undirected graphs, both directions are included.
- **reconstruct.txt**: all node URIs belonging to the train, validation, and test splits.
- **train.tsv**, **val.tsv**, **test.tsv**: partitioned subsets for downstream evaluation.

When starting the server, you select the dataset via `--graph`, for example:

```bash
python server.py --graph data/aifb/graph.json.zip
```

---

## Running the Server

Start the Flask server with:

```bash
python server.py --graph data/aifb/graph.json.zip
```

Optionally, add `--config` to load a JSON configuration file for LoRE.

By default, the server binds to `0.0.0.0:5000`.

---

## API Endpoints

All POST endpoints require `Content-Type: application/json`.

### Health Probe
```
GET /
```
Returns a message and current LoRE configuration.

### Export Base Embeddings
```
GET /base
```
Exports current node/relation URIs and embeddings as JSON.

### Training
```
POST /train
```
Body (example):
```json
{
  "epochs": 10,
  "base_nodes": ["uri1", "uri2", "uri3"],
  "num_workers": 0
}
```
Streams training logs line by line (plain text).

### Reconstruction
```
POST /rec
```
Body:
```json
{ "uris": ["uri1", "uri2", "uri3"] }
```
Returns reconstructed embeddings for the given URIs.

### What-If Queries
```
POST /what-if
```
Body:
```json
{
  "neighbors": ["uri_a", "uri_b"],
  "relations": ["rel_x", "rel_y"],
  "inverse": [0, 1]
}
```
Returns a hypothetical neighborhood embedding.

### Updates
```
POST /update
```
Body:
```json
{
  "nodes": ["uri_a", "uri_b"],
  "relations": ["rel_x"],
  "updates": {
    "add": [[0, 1, 0]],
    "remove": []
  }
}
```
Applies graph updates and mirrors them in the change-log.

### Change-Log
```
GET /log?type=add
GET /log?type=remove
```
Retrieves the N3-encoded change-log of applied updates.

---

## Usage Notes

- Training resets the internal change-log.
- Updates affect both the in-memory graph and embeddings immediately.
- All vectors are returned as lists and can be used directly in downstream tasks.
