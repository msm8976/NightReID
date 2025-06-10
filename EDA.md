#  The Enhancement, Denoising, and Alignment (EDA) Framework

## Requirements

### Datasets

- [NightReID](https://drive.google.com/drive/folders/1i8MqrxvA3NWoyVHtRyY25152iZUU7TAJ?usp=sharing)
- [Night600](https://github.com/Alexadlu/IDF#nighttime-person-reid-dataset-download-link)

### Pre-trained model

- [ViT-B/16+ICS](https://github.com/damo-cv/TransReID-SSL#pre-trained-models)

## Training

We utilize 1 GPU for training.

```bash
# NightReID
python train.py --config_file configs/NightReID/EDA.yml

# Night600
python train.py --config_file configs/Night600/EDA.yml
```

## Evaluation

```bash
# NightReID
python test.py --config_file configs/NightReID/EDA.yml TEST.WEIGHT EDA_nightreid.pth

# Night600
python test.py --config_file configs/Night600/EDA.yml TEST.WEIGHT EDA_night600.pth
```

## Integrating into Your Own Tasks

### IED module

- Copy model/IED.py and model/Myloss.py into your project.

- Important: IED expects raw pixel values in [0 , 1].

    - We therefore disable `T.Normalize` in `datasets/make_dataloader.py` and perform manual normalization in `model/make_model.py`.

- Train IED jointly with your own tasks.

### DDA module

The DDA module is **not** involved in training.

```bash
python model/DDA.py /path/to/your/train_set
```

This script computes dataset-specific priors. You can replace the corresponding section in your `config.yml` file.
Several optional methods are provided; choose the one that best suits your dataset.

## Acknowledgement

- Our implementation is built on [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL).
- The IED design is inspired by [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) and [RED-Net](https://github.com/hitzhangyu/V2-Self-supervised-Image-Enhancement-Network-Training-With-Low-Light-Images-Only).

Many thanks to the authors of these excellent works!