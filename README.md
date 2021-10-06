# Step

This is an pytorch implementation of `STEP: OOD Detection in the Presence of Limited In-Distribution Labeled Data`

## Requirements

Install conda environment via `environment.yaml`.

## Out-of-distribution Dataset

Download out-of-distributin datasets provided by ODIN: [Google Drive](https://drive.google.com/drive/folders/1aPyNXDib0uUb9a0CUK1DhelqM5_TLX7u?usp=sharing)

For example, you can download `Imagenet.tar.gz` into `./data/` directory and run script `tar -xvzf Imagenet.tar.gz`.

## Pre-trained Model

For a quick start, you can download our pre-trained model to `./files/` directory. 

Download Link: [Google Drive](https://drive.google.com/drive/folders/1PaV6rn168sYDKZ8opI_F1Qkmw2AHyIEp?usp=sharing)


You can also run the following scripts to train your own pre-trained model.
```bash
python SimCLR.py --out-dataset=LSUN --in-dataset=Cifar10
```

## Usage


Choose the datasets you want and run the script: `python Step.py --out-dataset={LSUN, LSUN_resize, Imagenet, Imagenet_resize} --in-dataset={Cifar10, Cifar100}`. For example, you can run the following script: 
```
python Step.py --out-dataset=LSUN --in-dataset=Cifar10
```

When the training stage is over, the final model will be stored in `./files/`, and the result will be printed and stored in `./results/`.




