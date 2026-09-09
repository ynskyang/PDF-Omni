# PDF-Omni: Poincaré Dual Disk Distortion Field-based Recurrent Update for Omnidirectional Stereo Matching

The PyTorch implementation of our paper

> **PDF-Omni: Poincaré Dual Disk Distortion Field-based Recurrent Update for Omnidirectional Stereo Matching**, ECCV 2026 ([paper](https://ynskyang.github.io/pdf-omni-page/static/paper/PDF-Omni.pdf), [project page](https://ynskyang.github.io/pdf-omni-page/))
>
> Yunseok Yang, Eunjin Son and Sang Jun Lee

## Quick start

The scripts under `scripts/` wrap the commands below:

```bash
bash scripts/setup_env.sh                 # conda env + dependencies
bash scripts/download_checkpoints.sh      # paper checkpoints into checkpoints/
bash scripts/prepare_data.sh /path/to/omnidata   # link the OmniMVS datasets under ../omnidata
bash scripts/eval.sh checkpoints/pdfomni_finetune.pth all --save_result
```

## Preparation

### Installation

Create the environment:
```bash
conda create -n pdf_omni python=3.10
conda activate pdf_omni
```

Install PyTorch (tested with 2.11.0 / CUDA 12.8, PyTorch >= 2.3 required):
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

Install other requirements:
```bash
pip install -r requirements.txt
```

### Datasets

Download OmniThings, OmniHouse, Sunny, Cloudy and Sunset from the [OmniMVS dataset page](https://rvlab.snu.ac.kr/research/omnistereo) and place them under `--db_root` (default `../omnidata`).
A lookup table is built inside each dataset folder on the first run.

## Training

### Train on OmniThings
```
python train.py --dbname omnithings --base_channel 32 --mixed_precision --ot_iter 3 --lec_weight 0.1 --use_ge --use_digru --ge_scale 1 --total_epochs 30
# use a single GPU
```

### Finetune on OmniHouse and Sunny
```
python train.py --dbname omnihouse sunny --base_channel 32 --mixed_precision --ot_iter 3 --lec_weight 0.1 --use_ge --use_digru --ge_scale 1 --total_epochs 16 --lr 0.0001 --pretrain_ckpt checkpoints/pdfomni_pretrain.pth
```

## Evaluation

Pretrained models: [pdfomni_pretrain.pth](https://github.com/ynskyang/PDF-Omni/releases/download/v1.0.0/pdfomni_pretrain.pth) (OmniThings) and [pdfomni_finetune.pth](https://github.com/ynskyang/PDF-Omni/releases/download/v1.0.0/pdfomni_finetune.pth) (finetuned on OmniHouse and Sunny).
These are the checkpoints used for the paper (pre-training epoch 28, fine-tuning epoch 14). Put them under `checkpoints/`.

```
python eval.py --dbname omnithings/omnihouse/sunny/cloudy/sunset/all --restore_ckpt checkpoints/pdfomni_finetune.pth --save_result
```

## Acknowledgements

This project borrows code from [RomniStereo](https://github.com/Insta360-Research-Team/RomniStereo), [S2M2](https://github.com/junhong-3dv/s2m2) and [Selective-Stereo](https://github.com/Windsrain/Selective-Stereo). We thank the original authors for their excellent work.

## License

This repository is released under [CC BY-NC 4.0](LICENSE) for non-commercial research use, since it adapts components from S2M2 (CC BY-NC 4.0). Third-party notices are included in `LICENSE`.

## Citation

```
@inproceedings{yang2026pdfomni,
  title     = {PDF-Omni: Poincar\'e Dual Disk Distortion Field-based Recurrent Update for Omnidirectional Stereo Matching},
  author    = {Yang, Yunseok and Son, Eunjin and Lee, Sang Jun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
