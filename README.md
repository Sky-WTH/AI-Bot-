# ⚡ Skynet (AI-Bot-)

**Train personalized neural networks from scratch, entirely on-device.**

Skynet is a lightweight Python SDK and CLI for building, training, and exporting small-to-medium neural networks — no cloud, no external APIs, just your own data and your own hardware.

> This repository currently contains two copies of the package: [`skynet/`](./skynet) and [`2.0 Model/`](./2.0%20Model), an in-progress iteration of the same library. Unless noted otherwise, everything below applies to both.

---

## Features

- 🧠 **Multiple architecture templates** — MLP, CNN, Transformer encoder, and Autoencoder, all registered through a common architecture registry.
- 📦 **Modality-aware data pipeline** — a single `DataPipeline` that routes to the right loader for tabular (CSV/JSON), image (folder-of-classes), or text (line-delimited / CSV) data, with automatic type detection and preprocessing.
- 🔁 **Incremental learning support** — a replay-buffer dataset wrapper to mitigate catastrophic forgetting across training rounds.
- ⚙️ **Config-driven training** — training and data settings are strictly validated [Pydantic](https://docs.pydantic.dev/) models, serializable to/from YAML or JSON.
- 💾 **Checkpointing & export** — resume interrupted runs, or export clean, portable model weights.
- 🖥️ **Device-aware** — auto-detects CPU, CUDA, or Apple Silicon (MPS), with optional mixed-precision training on CUDA.
- 🎛️ **Rich CLI** — a full `skynet` command-line tool built on [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/), so every operation is also available programmatically as a Python SDK.

## Installation

```bash
git clone https://github.com/Sky-WTH/AI-Bot-.git
cd AI-Bot-/skynet
pip install -e .
```

Requires **Python 3.9+**. Core dependencies (installed automatically): `torch`, `torchvision`, `typer`, `rich`, `pydantic`, `pyyaml`, `pillow`, `pandas`.

For running the test suite:

```bash
pip install -e ".[dev]"
```

## Quickstart

**1. Initialize a project.** This generates a starter `config.yaml` for a chosen architecture:

```bash
skynet init --arch mlp --config config.yaml
```

**2. Edit `config.yaml`** to point at your data and adjust hyperparameters:

```yaml
train:
  architecture: mlp
  learning_rate: 0.001
  batch_size: 32
  epochs: 10
  optimizer: adam
  loss_fn: cross_entropy
  device: auto

data:
  data_type: tabular
  data_path: ./data/my_dataset.csv
  target_column: label
  val_split: 0.2
```

**3. Train:**

```bash
skynet train --config config.yaml
```

**4. Evaluate, resume, or export:**

```bash
skynet evaluate --checkpoint checkpoints/best.pt --config config.yaml
skynet resume --checkpoint checkpoints/last.pt --config config.yaml
skynet export --checkpoint checkpoints/best.pt --output model.pth
```

## CLI Reference

| Command | Description |
|---|---|
| `skynet list-architectures` | List all registered architecture templates |
| `skynet init` | Scaffold a new project with a starter config file |
| `skynet train` | Start or resume training from a config file |
| `skynet resume` | Resume training from a specific checkpoint |
| `skynet evaluate` | Evaluate a checkpoint on a dataset |
| `skynet export` | Export clean model weights from a checkpoint |
| `skynet info` | Show metadata about a checkpoint file |
| `skynet device` | Show available compute devices |

Run `skynet <command> --help` for full options on any command.

## Architectures

| Name | Class | Best for |
|---|---|---|
| `mlp` | `MLP` | Tabular / structured data, general-purpose baseline |
| `cnn` | `CNN` | Image classification and feature extraction |
| `transformer` | Transformer encoder | Text classification and sequence modeling |
| `autoencoder` | `Autoencoder` | Unsupervised representation learning, dimensionality reduction, anomaly detection |

Architecture-specific hyperparameters (e.g. `hidden_dims`, `dropout`) are passed via the `arch_params` field in `TrainConfig`.

## Data Types

Set `data.data_type` to one of:

- **`tabular`** — CSV/JSON with automatic type detection; categorical features are label-encoded, numeric features are standardized.
- **`image`** — a directory of class-labeled subfolders (`root/class_a/*.png`, `root/class_b/*.jpg`, …).
- **`text`** — line-delimited `.txt` (optionally tab-separated label) or `.csv` with `text`/`label` columns.

## Project Structure

```
skynet/
├── skynet/
│   ├── cli.py              # Typer CLI (entry point: `skynet`)
│   ├── config.py           # TrainConfig / DataConfig / ExportConfig (Pydantic)
│   ├── architectures/       # mlp, cnn, transformer, autoencoder + registry
│   ├── data/                 # tabular, image, text, incremental loaders + pipeline
│   ├── trainer/               # training engine, checkpointing, callbacks
│   └── utils/                  # device detection, logging
├── tests/                # pytest test suite
└── pyproject.toml
```

## Testing

```bash
pytest
```

## License

Released under the [Apache License 2.0](./LICENSE).
