# Devanagiri Character Recognition
Recognizes Devanagiri character in an image

This model is implemented using Convolutional Neural Network.

## Architecture

 In --> [[Conv2D-->relu]*3 --> MaxPool2D --> Dropout]*2 --> Flatten --> Dense --> Dropout --> Out

#### Train accuracy ~ 99%
#### Test accuracy ~ 99%

## Dataset 

Dataset is obtained from https://www.kaggle.com/rishianand/devanagari-character-set
