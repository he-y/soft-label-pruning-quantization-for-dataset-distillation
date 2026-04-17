## Module 1: Squeeze with Class BN ([Google Drive](https://drive.google.com/drive/folders/1a1cT5uq0LZZuf4aGn3AgMR5EMopyvKHj?usp=drive_link))

To obtain network (i.e., ResNet) with class-wise batch normalization statistics, a modified ResNet is being forwarded for one epoch.
- modified ResNet: it is exactly same as ResNet but with `Normal BatchNorm2d` replaced by `ClassAwareBatchNorm2d`. It additionally tracks per-class BN statistics and does not have any influence on the performance.
- modified ResNet is at `models/resnet_class.py`

To obtain model with class BN, run the following scripts or download from [google drive](https://drive.google.com/drive/folders/1a1cT5uq0LZZuf4aGn3AgMR5EMopyvKHj?usp=drive_link):
```sh   
cd recover
bash scripts/imagenet1k_forward.sh
bash scripts/tiny_forward.sh
```
- Training model on ImageNet-21K is based on [this repo](https://github.com/Alibaba-MIIL/ImageNet21K).




## Module 2: Recover with Class BN ([Google Drive](https://drive.google.com/drive/folders/1JELI-Sbmob4WjW8a52xxVuOYtWdDmQaq?usp=drive_link))
To recover LPLD-distilled images, run the following script or download from [google drive](https://drive.google.com/drive/folders/1JELI-Sbmob4WjW8a52xxVuOYtWdDmQaq?usp=drive_link):
```sh
cd recover
bash scripts/imagenet1k_recover.sh
bash scripts/tiny_recover.sh
bash scripts/in21k_recover.sh
```
- This training script requires the class-wise BN stats from [Module 1](#module-1-squeeze-with-class-bn-google-drive).

## Module 3: Relabel, Prune, Quantize, and Validate ([Google Drive](https://drive.google.com/drive/folders/1LIKrlcydyowSkw2lRjgrzfULHYZWTNh7?usp=drive_link))

### Module 3.1: Relabel and Prune (with optional Quantization)
Basic Usage:
```sh
cd relabel_and_validate
python generate_soft_label_pruning_batch.py \
    --cfg_yaml [config file] \
    --train_dir [image path] \
    --fkd_path [label path to save] \
    --prune_ratio [pruning ratio] \
    --label_quantization [MR-k or None]
```
- the `[config file]` should contain information about **batch size, augmentation, and etc.**
    - the `val_dir` in `[config file]` should be replaced with your own path to the dataset test set.
- `--prune_ratio` controls label pruning (P): e.g. `0.9` keeps 10% of augmentation epochs (~10× compression).
- `--label_quantization` controls label quantization (Q): e.g. `MR-100` stores top-100 logits per sample. Set to `None` to disable.
- Pre-generated labels are available in [Google Drive](#download-datasets-and-labels).

### Module 3.2: Validate

Basic Usage:
```sh
python train_FKD_LPQLD.py \
    --cfg_yaml [config file] \
    --model [model] \
    --prune_ratio [pruning ratio] \
    --train_dir [image path] \
    --fkd_path [label path to load] \
    --label_quantization [MR-k or None] \
    --temp_scheduler [cosine/step/none] \
    --temp_stu_dynamic [ratio]
```
- `--label_quantization`, `--temp_scheduler`, `--temp_stu_dynamic` should match the settings used in [relabel](#module-31-relabel-and-prune-with-optional-quantization).
- `--temp_scheduler` enables DKR (Dynamic Knowledge Reuse, §III-D).
- `--temp_stu_dynamic` enables CA (Calibrated Alignment, §III-E).

For convenience, run the following scripts to reproduce results of the main table in the paper:
```sh
cd relabel_and_validate
bash scripts/reproduce/lpqld_in1k.sh
bash scripts/reproduce/lpqld_in21k.sh
```

