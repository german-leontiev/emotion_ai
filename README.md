# Emotions classification

## Business application
There are a plenty of business applications requires emotions classification.
Some of them described in this [article](https://rb.ru/longread/emotion-ai/)

## Project structure:

`config.py` - configuration file for train and evaluate models
`create_dataset.py` - script fot dataset creation. Before executing, please extract [this archive](https://drive.google.com/file/d/1TG9P5B2k3eTbC4XDxDmEc07dyAORPC16/view?usp=sharing) to `source_dataset` folder.
`train_models.py` - trains CNN models for future evaluation and saves weights to weights folder
`evaluate_models.py` - evaluates trained models
`inference.py` - inference sctipt

## Project environment

This project uses anaconda env.
To recreate it, please use:
`conda env create -f environment.yml`
`conda activate emotion_ai`

## Execute scripts

\[IMPORTANT!\] Before starting, please extract [this archive](https://drive.google.com/file/d/1TG9P5B2k3eTbC4XDxDmEc07dyAORPC16/view?usp=sharing) to `source_dataset` folder.

1. Create splited dataset.
`python create_dataset.py`

2. Train models
`python train_models.py 1>log.out 2>&1`

3. Evaluate models
`python evaluate_models.py 1>evaluation_results.txt 2>&1`

## Metrics choice.
Just because this task and data from [this kaggle competition](https://www.kaggle.com/competitions/skillbox-computer-vision-project/overview) it's reasonable to use competitions metrics (Accuracy). 

## Evaluation results

|  Model      |  Accuracy  |
|  :---       |  :----:    |
|  alexnet    |  0.46      |
|  densenet   |  0.53      |
|  inception  |  0.53      |
|  resnet     |  0.52      |
|  squeezenet |  0.48      |
|  vgg        |  0.52      |

## Best model
As you seen above `densenet` is the best model (inception has bigger infernce time)

## Required hardware.
Model tested on NVIDIA GeForce RTX 2070 SUPER
Time of inference 0.21 s

## Inference
Script takes one argument - path to file.

`$ python inference.py test_image.jpg`
Emotion: happy




