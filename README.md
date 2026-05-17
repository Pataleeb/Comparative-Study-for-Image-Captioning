[README.md](https://github.com/user-attachments/files/27863674/README.md)
# Comparative-Study-for-Image-Captioning# Image Captioning: LSTM vs Attention vs Transformer

A comparative study of three decoder architectures for image captioning on
MS COCO, all built on a frozen ResNet-101 visual encoder.

## Project Structure

```
image_captioning/
├── README.md
├── requirements.txt
├── setup_data.sh                # Downloads MS COCO 2017
├── config.py                    # All hyperparameters / paths
├── src/
│   ├── data/
│   │   ├── vocabulary.py        # Caption cleaning + vocab build
│   │   ├── dataset.py           # CocoCaptionDataset + transforms
│   │   └── loaders.py           # Train/val/test DataLoader factories
│   ├── models/
│   │   ├── baseline_lstm.py     # ResNet-101 + LSTM
│   │   ├── attention.py         # ResNet-101 + Spatial Attention LSTM
│   │   └── transformer.py       # ResNet-101 + Transformer Decoder
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_attention.py
│   │   └── train_transformer.py
│   ├── inference/
│   │   ├── greedy.py            # Greedy decode for all 3 models
│   │   └── beam_search.py       # Beam search for all 3 models
│   ├── evaluation/
│   │   ├── loss.py              # Test loss + perplexity
│   │   ├── bleu.py              # BLEU 1-4 (greedy + beam)
│   │   └── latency.py           # Inference latency (CUDA events)
│   └── utils/
│       ├── plotting.py          # Training curves + cross-model plots
│       └── helpers.py           # set_seed
└── scripts/
    ├── 01_prepare_data.py
    ├── 02_train_baseline_lstm.py
    ├── 03_train_attention.py
    ├── 04_train_transformer.py
    ├── 05_evaluate_models.py
    └── 06_data_efficiency_experiment.py
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download MS COCO 2017 (train + val + annotations, ~25 GB)
bash setup_data.sh
```

By default data lives at `/content/data/coco`. Override with:

```bash
export COCO_BASE_DIR=/path/to/coco
export RESULTS_DIR=/path/to/results
```

## Reproducing the Paper's Results

Run the scripts in order from the project root:

```bash
# Verify data + build vocabulary
python -m scripts.01_prepare_data

# Train all three models (each ~10 epochs)
python -m scripts.02_train_baseline_lstm
python -m scripts.03_train_attention
python -m scripts.04_train_transformer

# Evaluate on the held-out test set
python -m scripts.05_evaluate_models

# Cross-model data efficiency + latency experiment
python -m scripts.06_data_efficiency_experiment
```

All checkpoints, plots, and JSON results land in `$RESULTS_DIR`.

## Data

- Source: MS COCO `train2017` / `val2017` (118k / 5k images)
- 40,000 images sampled from train2017 (seed=42)
- Splits: 20,000 train / 2,000 val / 2,000 test (test drawn from val2017)
- Image preprocessing: resize 224x224, ImageNet normalization
- Vocabulary: 10,000 tokens (top frequencies + 4 special tokens)
- Captions: lowercased, non-alphanumeric stripped, whitespace tokenized
- `<start>` and `<end>` framing, max generation length = 20

## Architectures

All three models share a frozen ResNet-101 encoder. Only the decoder, the
embedding layer, and the linear classification head are trained.

| Model       | Decoder                                            | Hidden / d_model |
|-------------|----------------------------------------------------|------------------|
| Baseline    | Single-layer LSTM, global image feature            | 768              |
| Attention   | LSTMCell + Bahdanau soft attention over 14x14 grid | 768              |
| Transformer | nn.TransformerDecoder (3 layers, 8 heads)          | 256              |

## Reported Results (Greedy Decoding)

| Model       | Test Loss | Perplexity | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------------|-----------|------------|--------|--------|--------|--------|
| LSTM        | 2.902     | 18.21      | 0.343  | 0.198  | 0.121  | 0.077  |
| Attention   | 2.962     | 19.34      | 0.348  | 0.203  | 0.128  | 0.084  |
| Transformer | 2.950     | 19.11      | 0.347  | 0.198  | 0.113  | 0.067  |

