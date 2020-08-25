# CIFAR100_Mobilenet_variations

## Requirements

To install requirements:

```setup
conda env create -f requirements.yml
```

You can download the CIFAR100 tfrecords file here:

- [CIFAR100 tfrecords](https://drive.google.com/drive/folders/12RmORmC773Qv00z43c1Ly7db8J8MwRJ0?usp=sharing)

## Training

To train the model(s), run this command:

```train
python train.py <--se1> <--se2> <--se-plus> <--se-fusion> --suffix <model_dir_name>
```

> For example: 
> SE+MobileNet (after depthwise convs) : python train.py --se1
> SE-fusion+MobileNet (after both convs) : python train.py --se-fusion --se1 --se2
> Please specify the path to CIFAR100 tfrecords file in the config.json file.

## Evaluation

To evaluate the models on CIFAR100, run:

```eval
python eval.py <--se1> <--se2> <--se-plus> <--se-fusion> --suffix <model_dir>
```

## Results

Our model achieves the following performance on CIFAR-100:

| Model name         | Accuracy(std)% | FLOPs |
| ------------------ |------------------------ | ----- |
| MobileNet          |     62.21(0.42)              | 231.61M |
| SE+MobileNet (after depthwise convs)       |     62.03(0.67)              | 233.46M |
| SE+MobileNet (after pointwise convs)      |     63.70(0.32)              | 234.12M |
| SE+MobileNet (after both convs)  |     62.65(0.39)              | 235.97M |
| SE-plus+MobileNet (after depthwise convs)       |     63.07(0.21)              | 233.46M |
| SE-plus+MobileNet (after pointwise convs)      |     62.55(0.46)              | 234.12M |
| SE-plus+MobileNet (after both convs)  |     63.38(0.37)              | 235.97M |
| SE-fusion+MobileNet (after both convs)  |     63.71(0.18)              | 235.97M |
