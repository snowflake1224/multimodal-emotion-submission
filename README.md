# Blended Emotion Recognition — BlEmoRe @ IEEE FG 2026

> **First-Author Shared-Task Manuscript Repository (Not Published).**  
> Code and reproducible multimodal modeling notebooks for **CMAFNet** (Cross-Modal Attentive Fusion Network), built for the IEEE FG 2026 BlEmoRe Shared Task.

![CMAFNet Architecture](docs/diagrams/cmafnet-architecture.svg)

## Task & Problem Formulation

Blended emotions occur simultaneously in a clip with non-uniform intensity (`54.0%` of the 2,456 training clips combine two emotions across 6 basic classes: `ang`, `disg`, `fea`, `hap`, `sad`, `neu`). CMAFNet models the joint structure through two coordinated targets:

- **21-Class Emotion Combo Space:** `6` single emotions + `15` pairwise blended combinations $\binom{6}{2}$.
- **3-Class Salience Discretization:** Dominant split `{70/30, 50/50, 30/70}` decoded cleanly into the shared-task JSON submission format.

## Architecture (`blemore_cmaf_v2.ipynb`)

1. **Temporal Modality Pooling:** Mean + max temporal pooling across 4 pre-extracted modalities (`8,448` total features):
   - `VideoMAEv2` (`2,816-d`)
   - `VideoSwinTransformer` (`1,536-d`)
   - `WavLM-Large` (`2,048-d`)
   - `HuBERT-Large` (`2,048-d`)
2. **Modality Tokenization + Transformer Self-Attention:** Projects each modality into an independent embedding token alongside a learnable `[CLS]` token, applies multi-layer cross-modal self-attention, and aggregates with attention pooling.
3. **Training & Regularization:** Focal Cross-Entropy ($\gamma = 2.0$) with label smoothing, feature-space Mixup augmentation, modality dropout (`0.15`), Stochastic Weight Averaging (SWA), and sample reweighting across four architectural capacity variants (`cmaf_s`, `cmaf_m`, `cmaf_l`, `cmaf_xl`).

## 5-Fold True Out-of-Fold Cross-Validation Results

Evaluated strictly out-of-fold across all 5 stratified folds (`2,456` clips):

| System | Presence Acc ($\text{ACC}_{\text{presence}}$) | Salience Acc ($\text{ACC}_{\text{salience}}$) | Avg Score |
|--------|---|---|---|
| Trivial Baseline | 0.077 | 0.000 | 0.039 |
| HiCMAE Baseline | 0.298 | **0.180** | 0.239 |
| VideoMAEv2 + HuBERT | 0.332 | 0.138 | 0.235 |
| ImageBind + WavLM | 0.345 | 0.170 | **0.258** |
| **CMAFNet (Ours — 5-Fold OOF Ensemble)** | **0.3805** | 0.1307 | 0.2556 |

- Test submission generates predictions across `594` held-out clips (`submission.json` / `submission.zip`) using the top-K checkpoints across folds and variants.

## Files

- `blemore_cmaf_v2.ipynb` — CMAFNet cross-modal transformer fusion + Focal CE + Mixup + SWA + true OOF 5-fold validation & submission generation.
- `blemore_cmaf_v1.ipynb` — Initial baseline pipeline iteration.
- `docs/diagrams/cmafnet-architecture.{mmd,svg}` — Architecture & benchmark summary diagram.
