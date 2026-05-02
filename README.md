# ML SVG Project

This repository contains an SVG language-modeling project for studying scaling laws in decoder-only transformers and comparing standard parameterization against $\mu$P.

The core task is next-token prediction over cleaned SVG code. SVG files are treated as structured text, not as raster images. The main experimental pipeline is:

1. preprocess and normalize SVG data,
2. train a tokenizer on cleaned SVG strings,
3. encode the train/validation/test splits,
4. run one-epoch scaling studies across multiple model sizes,
5. fit scaling-law curves and compare standard parameterization with $\mu$P,
6. generate and evaluate SVG samples.

## Repository Structure

- `configs/`: preprocessing, tokenizer, and training configs
- `scripts/`: preprocessing, training, evaluation, fitting, and generation entrypoints
- `src/ml_svg_project/`: core library code
- `artifacts/`: tokenizer and encoded dataset artifacts
- `data/processed/`: cleaned dataset summaries
- `outputs/`: training runs, sweeps, fits, figures, and generation outputs
- `docs/`: assignment specification and related documents

For GitHub publication, the repository intentionally omits the bulk `data/` directory, the large `artifacts/` directory, and most model checkpoints. The included code and configs are still sufficient to reproduce those artifacts by rerunning preprocessing, tokenization, and training, provided that the required hardware and time are available.

## Report Breakdown

The final report uses an one-epoch protocol:

- dataset: `starvector/svg-icons-simple` + `75,000` rows from `starvector/svg-fonts-simple`
- train tokens: `133,464,477`
- tokenizer: Hugging Face BPE, vocab size `4096`
- context length: `1024`
- family sizes: `tiny`, `small`, `medium`, `large`, `xl`
- shared token batch for the final cloud runs: `196608`
- standard LR chosen by tiny sweep: `1e-3`
- $\mu$P LR chosen by tiny sweep: `1e-3`

Final family summaries used in the report:

- `outputs/standard_family/standard_family_h100_b196608/summary.json`
- `outputs/standard_family/mup_family_h100_b196608/summary.json`

Final scaling-fit summaries used in the report:

- `outputs/scaling_fits/standard_scaling_h100_b196608/fit_summary.json`
- `outputs/scaling_fits/mup_scaling_h100_b196608/fit_summary.json`
- `outputs/scaling_fits/standard_vs_mup_h100_b196608/comparison_summary.json`

## Important Notes

- Earlier exploratory runs exist in `outputs/`, but the final report only treats the exact H100 family runs as the core quantitative evidence.
- Some generation artifacts come from exploratory longer-run checkpoints and are used for qualitative failure analysis rather than for the main scaling comparison.
- `final_report/` is a self-contained export bundle for report writing and PDF generation; it is not part of the core training pipeline.
- The parameter sizes and batch sizes shown in the command examples are not universal defaults. They reflect a configuration that was runnable on a cloud H100 GPU. In practice, you should adjust these values to match your own hardware constraints and the amount of time available for training.
- The repository does not bundle the original datasets, encoded token arrays, packed arrays, or most checkpoints. If you want to rerun the full pipeline, you should expect to recreate those artifacts locally from the provided dataset sources and scripts.

## Environment Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main Python dependencies are listed in `requirements.txt`, including:

- `torch`
- `datasets`
- `tokenizers`
- `sentencepiece`
- `mup`
- `lxml`
- `cairosvg`
- `matplotlib`
- `scipy`

## Main Commands

### 1. Preprocess the active dataset

```bash
python scripts/preprocess_dataset.py \
  --config configs/preprocessing/icons_plus_fonts_75k.yaml
```

### 2. Train the tokenizer

```bash
python scripts/train_tokenizer.py \
  --preprocess-config configs/preprocessing/icons_plus_fonts_75k.yaml \
  --tokenizer-config configs/tokenizer/hf_bpe_4096.yaml
```

### 3. Run a tiny learning-rate sweep

Standard:

```bash
python scripts/run_lr_sweep.py \
  --preprocess-config configs/preprocessing/icons_plus_fonts_75k.yaml \
  --tokenizer-config configs/tokenizer/hf_bpe_4096.yaml \
  --training-config configs/training/tiny.yaml \
  --learning-rates 1e-4 3e-4 5e-4 8e-4 1e-3 \
  --sweep-name tiny_lr_sweep_h100_b196608 \
  --batch-size-tokens 196608
```

$\mu$P:

```bash
python scripts/run_lr_sweep.py \
  --preprocess-config configs/preprocessing/icons_plus_fonts_75k.yaml \
  --tokenizer-config configs/tokenizer/hf_bpe_4096.yaml \
  --training-config configs/training/tiny_mup.yaml \
  --learning-rates 1e-4 3e-4 5e-4 8e-4 1e-3 \
  --sweep-name tiny_mup_sweep_h100_b196608 \
  --batch-size-tokens 196608
```

### 4. Run the scaling families

Standard family:

```bash
python scripts/run_standard_family.py \
  --preprocess-config configs/preprocessing/icons_plus_fonts_75k.yaml \
  --tokenizer-config configs/tokenizer/hf_bpe_4096.yaml \
  --learning-rate 1e-3 \
  --family-name standard_family_h100_b196608 \
  --batch-size-tokens 196608
```

$\mu$P family:

```bash
python scripts/run_standard_family.py \
  --preprocess-config configs/preprocessing/icons_plus_fonts_75k.yaml \
  --tokenizer-config configs/tokenizer/hf_bpe_4096.yaml \
  --training-configs \
    configs/training/tiny_mup.yaml \
    configs/training/small_mup.yaml \
    configs/training/medium_mup.yaml \
    configs/training/large_mup.yaml \
    configs/training/xl_mup.yaml \
  --learning-rate 1e-3 \
  --family-name mup_family_h100_b196608 \
  --batch-size-tokens 196608
```

### 5. Fit scaling curves and compare

Standard fit:

```bash
python scripts/fit_scaling_curve.py \
  --family-summary outputs/standard_family/standard_family_h100_b196608/summary.json \
  --output-name standard_scaling_h100_b196608
```

$\mu$P fit:

```bash
python scripts/fit_scaling_curve.py \
  --family-summary outputs/standard_family/mup_family_h100_b196608/summary.json \
  --output-name mup_scaling_h100_b196608
```

Comparison:

```bash
python scripts/compare_scaling_fits.py \
  --standard-fit outputs/scaling_fits/standard_scaling_h100_b196608/fit_summary.json \
  --mup-fit outputs/scaling_fits/mup_scaling_h100_b196608/fit_summary.json \
  --output-name standard_vs_mup_h100_b196608
```

### 6. Generate and evaluate SVG samples

The repository keeps compact generation outputs used in the final report, but it does not ship the exploratory longer-run checkpoint itself. To rerun raw generation, you will need to point the commands below to a locally available trained checkpoint.

Raw generation:

```bash
python scripts/generate_samples.py \
  --model-path outputs/training_runs/xl_best_step7/model.pt \
  --tokenizer-path artifacts/tokenizer/starvector__svg_icons_simple__plus__starvector__svg_fonts_simple__maxrows_75000__stream__hf_tokenizers_4096/tokenizer.json \
  --output-name xl_best_uncond_t08 \
  --num-samples 3 \
  --temperature 0.8 \
  --top-k 40
```

Forced-structure diagnostic:

```bash
python scripts/generate_forced_svg_samples.py \
  --model-path outputs/training_runs/xl_best_step7/model.pt \
  --tokenizer-path artifacts/tokenizer/starvector__svg_icons_simple__plus__starvector__svg_fonts_simple__maxrows_75000__stream__hf_tokenizers_4096/tokenizer.json \
  --output-name xl_best_forced_path_t08 \
  --num-samples 6 \
  --max-new-tokens 192 \
  --temperature 0.8 \
  --top-k 40 \
  --mode path_attr
```

Evaluation:

```bash
python scripts/evaluate_generated_samples.py \
  --samples-dir outputs/generation/xl_best_uncond_t08
```

## Cloud Note

The final exact family runs were executed on a cloud H100 GPU because the literal one-epoch protocol becomes expensive on smaller hardware when the token batch must be shared across the whole model family.

Cloud execution is not required to understand or reuse the repository, but if you want to reproduce the final reported scaling results efficiently, a high-memory GPU is strongly recommended. The local scripts support batch-size overrides directly from the command line, so the same pipeline can be adapted to different hardware.

## Reference

The assignment specification is in:

- `docs/optional-project-spring26.pdf`
