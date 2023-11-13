# printer_source
Using multiple variations of ResNet as the backbone architecture of classification model along with augmentation and data enhancement methods to classify print sources from microscopic images of ink dots. The data is microscopic printed images containing microscopic printing patterns from various source printers. 

# requirements
torch==1.12.0
torchvision==0.13.0
pytorch-metric-learning==1.5.1
numpy==1.19.4
Pillow==9.2.0
imageio==2.21.0
tqdm==4.19.9

# train model
python3 train.py -m {model}

# Backbones used
resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2
