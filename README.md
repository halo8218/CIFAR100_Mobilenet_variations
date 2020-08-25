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
python train.py <--se1> or <--se2> or <--se-plus> --suffix <model_dir_name>

```

> 📋Please specify the path to CIFAR100 tfrecords file in the config.json file.

## Evaluation

To evaluate the models on CIFAR100, run:

```eval
python eval.py <model_dir>
```

## Results

Our model achieves the following performance on CIFAR-100:

| Model name         | Classification Accuracy(std) | FLOPs |
| ------------------ |------------------------ | ----- |
| MobileNet          |     94.93%              | 00 |
| SE+MobileNet (after depthwise convs)       |     95.67%              | 00 |
| SE+MobileNet (after pointwise convs)      |     87.48%              | 00 |
| SE+MobileNet (after both convs)  |     88.33%              | 00 |
| SE-plus+MobileNet (after depthwise convs)       |     95.67%              | 00 |
| SE-plus+MobileNet (after pointwise convs)      |     87.48%              | 00 |
| SE-plus+MobileNet (after both convs)  |     88.33%              | 00 |
