# Soft Label Pruning and Quantization for Large-Scale Dataset Distillation (<ins>LPQLD</ins>)
[[`Paper`](https://ieeexplore.ieee.org/document/11395619/) | [`BibTex`](#citation) | [`Google Drive`](https://drive.google.com/drive/folders/1fZn0mH3_hQSh4FLzKeULAbkr9Avtys5X?usp=sharing)]

---

Official implementation of **LPQLD**: *Soft Label Pruning and Quantization for Large-scale Dataset Distillation* (IEEE TPAMI, 2026).

> This work extends [LPLD (NeurIPS'24)](https://github.com/he-y/soft-label-pruning-for-dataset-distillation) with two orthogonal improvements: **Label Quantization (Q)** for further storage compression, and **Dynamic Knowledge Reuse (DKR)** + **Calibrated Student-Teacher Alignment (CA)** for improved supervision diversity.

[Lingao Xiao](https://armandxiao.github.io/),&nbsp; [Yang He](https://he-y.github.io/)

> **Abstract**: Large-scale dataset distillation requires storing auxiliary soft labels that can be 30–40× (ImageNet-1K) or 200× (ImageNet-21K) larger than the condensed images, undermining the goal of dataset compression. We identify two fundamental issues: (1) *insufficient image diversity* and (2) *insufficient supervision diversity*. To address both, we propose **LPQLD**, which reduces soft label storage by **78×** on ImageNet-1K and **500×** on ImageNet-21K while improving accuracy by up to 7.2% and 2.8%, respectively. LPQLD combines Label Pruning (P) and Label Quantization (Q) for storage compression, with Dynamic Knowledge Reuse (DKR) and Calibrated Alignment (CA) for supervision diversity during training.

---

# Installation

Download repo:
```sh
git clone https://github.com/he-y/soft-label-pruning-quantization-for-dataset-distillation LPQLD
cd LPQLD
```

Create environment:
```sh
conda env create -f environment.yml
conda activate lpqld
```

## Download Datasets and Labels

### Method 1: Automatic Downloading
```sh
# sh download.sh [true|false]
sh download.sh true
```
- `true` (default): download recommended labels only (×80 for ImageNet-1K, ×400 for ImageNet-21K)
- `false`: download all label variants (**warning: very large, tens of GBs**)

### Method 2: Manual Downloading

Download labels from the tables below and place files in the following structure:
```
.
├── recover/
│   ├── model_with_class_bn/    ← reuse from LPLD (see below)
│   └── syn_data_LPLD/          ← reuse from LPLD (see below)
└── relabel_and_validate/
    └── syn_label_LPQLD/
        ├── FKD_cutmix_fp16_LPQLD_in1k_rn18_4k_ipc10_ratio9_topk10/
        ├── FKD_cutmix_fp16_LPQLD_in1k_rn18_4k_ipc10_ratio9_topk50/
        └── ...
```

---

### Model with Class-wise BN and Distilled Images

> **Reuse from LPLD.** Download from [LPLD Google Drive](https://drive.google.com/drive/folders/1_eFjyWmrFXtprslgAwjyMpvhfB_qTf7t?usp=sharing): the synthesized images and class-BN models are identical.

|    Dataset    | Model with Class-BN | Distilled Images |
| :-----------: | :-----------------: | :--------------: |
|  ImageNet-1K  | [50 MB](https://drive.google.com/file/d/1Vfou8nPp3x7m7YEG0wd7FcuQ_yE9jj34/view?usp=drive_link) | IPC10–200 ([0.15–2.98 GB](https://drive.google.com/drive/folders/1_eFjyWmrFXtprslgAwjyMpvhfB_qTf7t?usp=sharing)) |
| Tiny-ImageNet | [81 MB](https://drive.google.com/file/d/1sCArvJoHFthbSaBuWoUhDn67tsYtRPTn/view?usp=drive_link) | IPC50/100 ([21–40 MB](https://drive.google.com/drive/folders/1_eFjyWmrFXtprslgAwjyMpvhfB_qTf7t?usp=sharing)) |
| ImageNet-21K  | [446 MB](https://drive.google.com/file/d/1BuplTqBhXKzdfJqCKkTBg218Cbezef57/view?usp=drive_link) | IPC10/20 ([3–5 GB](https://drive.google.com/drive/folders/1_eFjyWmrFXtprslgAwjyMpvhfB_qTf7t?usp=sharing)) |

---


### Previous Soft Labels vs. Ours (LPQLD)

Each cell shows `storage / accuracy`. Click storage size to download.

**ImageNet-1K** — full original labels: [LPLD Google Drive](https://drive.google.com/drive/folders/1_eFjyWmrFXtprslgAwjyMpvhfB_qTf7t?usp=sharing)

| IPC | Previous (full) | LPLD ×40 | LPQLD ×40 | LPQLD ×80 | LPQLD ×200 |
| :-: | :-------------: | :------: | :-------: | :-------: | :--------: |
| 10 | 5.67 GB / 20.1% | [0.14 GB](https://drive.google.com/file/d/1Nf1piVIXIF-_v-jCEmaYGHdWTXsuQIkY/view?usp=drive_link) / 20.2% | [0.13 GB](https://drive.google.com/drive/folders/1Yu8UDYPZHTJKIkvaUScRvea80DKjglt8?usp=drive_link) / 29.6% | [0.07 GB](https://drive.google.com/drive/folders/1eiw9VJZDq3Eoba1gZ5ZToRI_F_E97KGm?usp=drive_link) / 27.3% | [0.03 GB](https://drive.google.com/drive/folders/1RPG8GmHsZYRA7cX-utgjmg4DuHCOJegU?usp=drive_link) / 20.0% |
| 20 | 11.33 GB / 33.6% | [0.29 GB](https://drive.google.com/file/d/1AdP44DJUadFlY1WCrYiE7F6slotk3Vx4/view?usp=drive_link) / 33.0% | [0.25 GB](https://drive.google.com/drive/folders/1q_xpjFzmA0mZP8UKDb-_EZsDeXwAjp_S?usp=drive_link) / 41.2% | [0.15 GB](https://drive.google.com/drive/folders/1HZo4cTvO2lFducodNcZ5p-aoQ4MlFDmY?usp=drive_link) / 38.6% | [0.06 GB](https://drive.google.com/drive/folders/1_TeP_l6GsbF4dwvz1SAMkFaOFvGNFpcw?usp=drive_link) / 30.5% |
| 50 | 28.33 GB / 46.8% | [0.71 GB](https://drive.google.com/file/d/1GnCY-Apg-dXgZe8BvDwDKqrQSAz1PAbs/view?usp=drive_link) / 46.7% | [0.63 GB](https://drive.google.com/drive/folders/1tpm3XD1LUYLLxAexH56mc9onlMn1XF13?usp=drive_link) / 51.3% | [0.37 GB](https://drive.google.com/drive/folders/1fphtxy9E5BAtOfGjZZk-mQQJ-d0bIBLf?usp=drive_link) / 49.6% | [0.14 GB](https://drive.google.com/drive/folders/1USH0vrNKFUELUZkg9eBV5PD7BMjnEEcy?usp=drive_link) / 43.0% |
| 100 | 56.66 GB / 52.8% | [1.43 GB](https://drive.google.com/file/d/12f6qUjsoN6AczK7iJz2ZAT8xNiX0W4bX/view?usp=drive_link) / 54.0% | [1.27 GB](https://drive.google.com/drive/folders/1L6d4nCfb_ROuelI0417TpSmVt7s6ga88?usp=drive_link) / 56.2% | [0.73 GB](https://drive.google.com/drive/folders/1mq17UpTpxpZtGmD6uLTbhCO8Db_uLmVI?usp=drive_link) / 54.9% | [0.29 GB](https://drive.google.com/drive/folders/1wmZuNjRM4mc1FDhOK6Dh78dJfnKD-TfJ?usp=drive_link) / 50.1% |
| 200 | 113.33 GB / 57.0% | [2.85 GB](https://drive.google.com/file/d/1mHWwOaB0yG7fP_lbDSZMmIHUrMh_nDWZ/view?usp=drive_link) / 59.6% | [2.54 GB](https://drive.google.com/drive/folders/13JWSrZj_8JmbI8zPnFAgLKswRQfGI_4l?usp=drive_link) / 59.9% | [1.47 GB](https://drive.google.com/drive/folders/1rd8ibLwQ21sF_0qDfPnznwvQVdteE0QB?usp=drive_link) / 58.8% | [0.58 GB](https://drive.google.com/drive/folders/1EHXNvN6AOHtXXy-Qurw1Ht8RFiNh6IJ8?usp=drive_link) / 55.4% |
> ×40 = MR-100 (top-100 logits); ×80 = MR-50 (top-50 logits); ×200 = MR-10 (top-10 logits).

**ImageNet-21K** — full original labels are too large to upload; compressed variants provided below.

| IPC | Previous (full) | LPLD ×40 | LPQLD ×40 | LPQLD ×400 |
| :-: | :-------------: | :------: | :-------: | :--------: |
| 10  | 643 GB / 18.5% | [16 GB](https://drive.google.com/file/d/1inuNAC7ApJWiuXaCsEwWU9_z7DOpMBzG/view?usp=drive_link) / 21.3% | [16 GB](https://drive.google.com/drive/folders/11R5SNhtXDdy79-_E8kUki-MEu8NYK1dQ?usp=drive_link) / 25.6% | [1.6 GB](https://drive.google.com/open?id=1dxCJzBLRNYlq-bxuaE42rwr1vbnMGPMG&usp=drive_copy) / 20.9% |
| 20  | 1286 GB / 20.5% | [32 GB](https://drive.google.com/file/d/1g52Lo2XoKHbJySkiLFo3Gsl6hnjffOEN/view?usp=drive_link) / 29.4% | [32 GB](https://drive.google.com/open?id=1sgqbTGvR5RY4aY-3KGmypVETlKFRaFAw&usp=drive_link) / 33.8% | [2.7 GB](https://drive.google.com/open?id=1Mj_RgnlkW9CFQvrUmf_DlXoGShU1-aHB&usp=drive_copy) / 24.1% |

> ×40 = label pruning only; ×400 = label pruning + quantization (MR-200).

---

## Necessary Modification for PyTorch

Modify `torch.utils.data._utils.fetch._MapDatasetFetcher` to support multi-processing loading of soft label data and mix configurations:

```python
class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            if hasattr(self.dataset, "G_VBSM") and self.dataset.G_VBSM:
                pass  # G_VBSM: uses self-decoding in the training script
            elif hasattr(self.dataset, "use_batch") and self.dataset.use_batch:
                mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config_by_batch_idx(possibly_batched_index[0])
            else:
                mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(possibly_batched_index[0])

        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]

        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            # NOTE: mix_index, mix_lam, mix_bbox can be None
            mix_index_cpu = mix_index.cpu() if mix_index is not None else None
            return self.collate_fn(data), mix_index_cpu, mix_lam, mix_bbox, soft_label.cpu()
        else:
            return self.collate_fn(data)
```

---

# Reproduce Main Results

## Step 1: Set Paths

Edit two files before running any scripts:

**1. Config YAML** — set the validation directory:
```yaml
# relabel_and_validate/cfg/reproduce/LPQLD_in1k_[4k].yaml (and LPQLD_in21k_[2k].yaml for ImageNet-21K)
validate:
  path:
    val_dir: /path/to/imagenet/val   # ← set this
```

**2. Reproduce script** — set the distilled image directory:
```bash
# relabel_and_validate/scripts/reproduce/lpqld_in1k.sh, line 28:
train_dir="/path/to/syn_data_LPLD/LPLD_in1k_rn18_4k_ipc${ipc}"  # ← set this

# relabel_and_validate/scripts/reproduce/lpqld_in21k.sh, line 29:
train_dir="/path/to/syn_data_in21k/sre2l_in21k_rn18_2K_ipc${ipc}"  # ← set this
```

## Step 2: Run Reproduce Scripts

```sh
cd relabel_and_validate
bash scripts/reproduce/lpqld_in1k.sh    # ImageNet-1K
bash scripts/reproduce/lpqld_in21k.sh   # ImageNet-21K
```

Alternatively, use downloaded labels directly — refer to [README_usage.md](./README_usage.md) for the three-stage pipeline.

---

# What's New in LPQLD vs LPLD

| Component | LPLD | LPQLD |
|-----------|:----:|:-----:|
| Class-wise BN synthesis (§III-B) | ✓ | ✓ |
| Label Pruning — P (§III-C) | ✓ | ✓ |
| Label Quantization — Q (§III-C) | ✗ | ✓ |
| Dynamic Knowledge Reuse — DKR (§III-D) | ✗ | ✓ |
| Calibrated Student-Teacher Alignment — CA (§III-E) | ✗ | ✓ |

**DKR** (`--temp_scheduler`): temperature annealing on stored pre-softmax logits extracts diverse supervisory signals across training epochs without additional storage.

**CA** (`--temp_stu_dynamic`): dynamically adjusts the student temperature as a calibrated ratio of the teacher temperature, aligning their probability distributions.

---

## Related Repos

- [LPLD](https://github.com/he-y/soft-label-pruning-for-dataset-distillation) — Are Large-scale Soft Labels Necessary for Large-scale Dataset Distillation?
- [SRe²L](https://github.com/VILA-Lab/SRe2L) — Squeeze, Recover and Relabel framework
- [ImageNet-21K Pretraining for the Masses](https://github.com/Alibaba-MIIL/ImageNet21K)

---

## Citation

```bibtex
@article{xiao2025lpqld,
  title   = {Soft Label Pruning and Quantization for Large-Scale Dataset Distillation},
  author  = {Lingao Xiao and Yang He},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2025}
}

@inproceedings{xiao2024lpld,
  title     = {Are Large-scale Soft Labels Necessary for Large-scale Dataset Distillation?},
  author    = {Lingao Xiao and Yang He},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year      = {2024}
}
```
