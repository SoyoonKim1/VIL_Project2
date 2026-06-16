# VIL Project 2 — Self-Supervised Learning on STL-10

Single-GPU SSL pretraining pipelines on **STL-10 unlabeled** (100k images, 96×96) with **ResNet-50** (no ImageNet pretrained weights). All methods share periodic **kNN / linear eval** on STL-10 labels during training, plus fixed downstream evaluators for fair comparison.

## SSL Methods

| Method | Entry script | Loss / idea | Notes |
|--------|--------------|-------------|-------|
| **SimCLR** | `pretrain_simclr_stl10.py` | NT-Xent contrastive loss on two augmented views | Simplest baseline; best STL downstream in this repo after tuning |
| **SimCLR Hybrid** | `pretrain_simclr_stl10_hybrid.py` | NT-Xent + EMA teacher distillation + light block masking | Experimental extension of SimCLR |
| **SwAV** | `pretrain_swav_stl10.py` | Online clustering with Sinkhorn assignments to prototypes | Strong CIFAR transfer; STL/CIFAR gap smaller than SimCLR |
| **SwAV-MCQ** | `pretrain_swav_stl10_mcq.py` | SwAV + multi-crop (global/local views) + optional queue | Closer to original SwAV recipe; supports LARS |
| **Barlow Twins** | `pretrain_barlowtwins_stl10.py` | Cross-correlation redundancy reduction | Uses LARS + gradient accumulation for large effective batch |

### SimCLR

- **Augmentation:** RandomResizedCrop, flip, color jitter, grayscale, Gaussian blur (see `simclr/augmentations.py`)
- **Head:** MLP projector `2048 → 2048 → D` (default `D=128`, best tuned run uses `D=256`)
- **Optimizer:** SGD with linear LR scaling `lr × (batch_size / 256)`

Best tuned config in this repo (`ab5g_simclr_no_lars_exp4_e500`): batch 320, temp 0.2, projection out 256, wd 1e-4, 500 epochs.

### SimCLR Hybrid

Adds on top of SimCLR:

- EMA teacher distillation (`--distill-weight`, `--ema-momentum`)
- Random block masking on input (`--mask-ratio`, `--mask-patch-size`)

### SwAV

- **Augmentation:** two global crops (same family as SimCLR)
- **Loss:** assign crops to learnable prototypes via Sinkhorn-Knopp
- Key hyperparameters: `--num-prototypes`, `--temperature`, `--sinkhorn-epsilon`

### SwAV-MCQ (Multi-Crop + Queue)

- **Augmentation:** 2 global + N local crops (`--local-crops-number`, `--local-crop-size`)
- **Optimizer:** LARS (default) or SGD
- Optional feature queue for additional negatives (`--queue-length`, `--queue-start-epoch`)

### Barlow Twins

- **Augmentation:** two distorted views (see `barlowtwins/augmentations.py`)
- **Loss:** Barlow Twins cross-correlation matrix objective
- **Optimizer:** LARS with optional gradient accumulation (`--grad-accum-steps`) for effective batch 1024

---

## Project Structure

```text
VIL_Project2/
  pretrain_simclr_stl10.py       # SimCLR
  pretrain_simclr_stl10_hybrid.py
  pretrain_swav_stl10.py         # SwAV (2-crop)
  pretrain_swav_stl10_mcq.py     # SwAV multi-crop
  pretrain_barlowtwins_stl10.py
  evaluation.py                  # fixed linear probing (STL-10 / CIFAR-10)
  evaluation_finetune.py         # fixed full fine-tuning
  visualize_tsne_stl10.py
  simclr/                        # SimCLR model, loss, aug, datasets
  swav/                          # SwAV model, loss
  barlowtwins/                   # Barlow Twins model, loss, aug
  scripts/                       # nohup launchers & eval scripts
  configs/simclr_stl10.yaml      # reference config
  outputs/                       # checkpoints & train_config.json
  logs/                          # training & eval logs
```

---

## Install

```bash
pip install -r requirements.txt
```

Requirements: PyTorch ≥ 2.1, torchvision, tensorboard, tqdm, matplotlib, scikit-learn.

---

## Quick Start — Pretrain (single GPU)

### SimCLR (recommended baseline)

```bash
python pretrain_simclr_stl10.py \
  --data-root ./data \
  --save-dir ./outputs/simclr_stl10_seed42 \
  --download \
  --seed 42 \
  --gpu-id 0 \
  --epochs 500 \
  --batch-size 320 \
  --temperature 0.2 \
  --projection-out-dim 256 \
  --lr 0.3 \
  --weight-decay 0.0001 \
  --warmup-epochs 10 \
  --num-workers 8 \
  --amp \
  --tensorboard \
  --eval-interval 25 \
  --linear-eval-epochs 100 \
  --linear-eval-lr 0.01 \
  --disable-tqdm
```

### SwAV

```bash
python pretrain_swav_stl10.py \
  --data-root ./data \
  --save-dir ./outputs/swav_stl10_seed42 \
  --download \
  --seed 42 \
  --gpu-id 0 \
  --epochs 400 \
  --batch-size 512 \
  --num-prototypes 512 \
  --temperature 0.1 \
  --lr 0.15 \
  --amp \
  --tensorboard \
  --eval-interval 25 \
  --disable-tqdm
```

### SwAV-MCQ

```bash
python pretrain_swav_stl10_mcq.py \
  --data-root ./data \
  --save-dir ./outputs/swav_stl10_mcq_seed42 \
  --download \
  --seed 42 \
  --gpu-id 0 \
  --epochs 200 \
  --batch-size 256 \
  --num-prototypes 3000 \
  --local-crops-number 4 \
  --optimizer lars \
  --amp \
  --tensorboard \
  --eval-interval 25 \
  --disable-tqdm
```

### Barlow Twins

```bash
python pretrain_barlowtwins_stl10.py \
  --data-root ./data \
  --save-dir ./outputs/barlowtwins_stl10_seed42 \
  --download \
  --seed 42 \
  --gpu-id 0 \
  --epochs 600 \
  --batch-size 128 \
  --grad-accum-steps 8 \
  --lr 0.6 \
  --optimizer lars \
  --amp \
  --tensorboard \
  --eval-interval 50 \
  --disable-tqdm
```

### SimCLR Hybrid

```bash
python pretrain_simclr_stl10_hybrid.py \
  --data-root ./data \
  --save-dir ./outputs/simclr_stl10_hybrid_seed42 \
  --download \
  --seed 42 \
  --gpu-id 0 \
  --epochs 200 \
  --amp \
  --tensorboard \
  --eval-interval 25 \
  --disable-tqdm
```

---

## Background Training (nohup)

Ready-made launch scripts live in `scripts/`. Examples:

```bash
# SimCLR best config, e500
bash scripts/run_ab5g_exp4_e500_gpu0_nohup.sh

# SimCLR follow-up ablations (GPU 0–3)
bash scripts/run_simclr_followup_exps_nohup.sh

# SimCLR 5-GPU ablation (temp, proj dim, epochs, batch)
bash scripts/run_ablation_5gpu_no_lars_nohup.sh

# SwAV
bash scripts/run_swav_gpu6_bs512_lr015_e400_nohup.sh

# SwAV 2-GPU ablation
bash scripts/run_swav_ablation_2gpu_nohup.sh

# SwAV-MCQ 7-GPU ablation
bash scripts/run_swav_mcq_ablation_7gpu_nohup.sh

# Barlow Twins
bash scripts/run_barlowtwins_b_gpu7_efflr06_nohup.sh
```

Logs: `./logs/<run_name>.log`  
PIDs: `./logs/<run_name>.log.pid`

---

## In-Training Evaluation

When `--eval-interval > 0`, each method runs on STL-10 labeled splits:

- **kNN top-1** — frozen backbone + weighted kNN
- **Linear top-1** — linear classifier on frozen train features

Logged to TensorBoard as `eval/knn_top1`, `eval/linear_top1`, `eval/time_sec`.

```bash
tensorboard --logdir ./outputs/<run_name>/tensorboard --port 6006
```

> In-training linear eval uses `--linear-eval-lr` (default 0.01 for SimCLR tuned runs). This is **not** the same recipe as `evaluation.py` below.

---

## Downstream Evaluation (fixed protocol)

Official comparison uses frozen features + fixed hyperparameters (do not change `evaluation.py` / `evaluation_finetune.py`).

| Script | Task | Protocol |
|--------|------|----------|
| `evaluation.py` | Linear probing | SGD lr=0.1, 100 epochs, batch 128, cosine schedule |
| `evaluation_finetune.py` | Full fine-tuning | ResNet-50 + linear head, same optimizer schedule |

**Datasets:** STL-10 (train 5k → test 8k), CIFAR-10 (train 50k → test 10k, upsampled to 96×96 with STL normalization for feature extraction).

### Single checkpoint

```bash
CKPT_PATH=./outputs/ab5g_simclr_no_lars_exp4_e500_seed42_e500/backbone_best.pt \
RUN_NAME=ab5g_simclr_no_lars_exp4_e500_seed42_e500 \
bash scripts/run_eval_linear_and_finetune.sh
```

### SimCLR follow-up runs (batch)

```bash
bash scripts/run_eval_simclr_followup.sh        # all 4 follow-up runs
bash scripts/run_eval_simclr_followup.sh exp1 # one run
```

### SwAV ablation pair

```bash
bash scripts/run_eval_swav_ab2g_pair_nohup.sh
```

Features are cached under `./outputs/eval_features/<run_name>/`.

---

## Checkpoints

Each run saves under `--save-dir`:

| File | Description |
|------|-------------|
| `backbone_best.pt` | Best backbone weights (use for downstream eval) |
| `backbone_epoch_XXXX.pt` | Periodic backbone snapshots |
| `checkpoint_best.pt` / `checkpoint_last.pt` | Full training state |
| `train_config.json` | All CLI arguments |

- Backbone: ResNet-50 without pretrained weights (`weights=None`)
- Projection / prototype heads are **discarded** for linear probing; only backbone features (2048-d) are used

---

## Utilities

```bash
# t-SNE visualization of backbone features
python visualize_tsne_stl10.py --checkpoint ./outputs/<run>/backbone_best.pt
```

---

## Tests

```bash
python -m pytest tests/
```

---

## Notes

- All pretraining uses **STL-10 `unlabeled`** split unless noted otherwise.
- **Single-GPU** only; SimCLR NT-Xent uses in-batch negatives (no cross-GPU queue).
- For nohup runs, pass `--disable-tqdm` to keep log files small.
- `configs/simclr_stl10.yaml` is a reference setting; actual runs are driven by CLI / shell scripts.
- STL downstream accuracy is typically **higher than CIFAR** because pretrain and eval share the same domain; long SimCLR training can widen this gap.
