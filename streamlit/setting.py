import os
from pathlib import Path
import sys


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the parent directory of the current file
root_path = os.path.dirname(current_file_path)

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(root_path)

# Get the relative path of the root directory with respect to the current working directory
ROOT = "E:/code/THESIS/printer_source/checkpoint"


# ML Model config 

resnet50 = os.path.join(ROOT, 'resnet50')
resnet101 = os.path.join(ROOT, 'resnet101')
resnet152 = os.path.join(ROOT, 'resnet152')
resnext50_32x4d = os.path.join(ROOT, 'resnext50_32x4d')
resnext101_32x8d = os.path.join(ROOT, 'resnext101_32x8d')
resnext101_64x4d = os.path.join(ROOT, 'resnext101_64x4d')
wide_resnet50_2 = os.path.join(ROOT, 'wide_resnet50_2')
wide_resnet101_2 = os.path.join(ROOT, 'wide_resnet101_2')



MODEL_DICT = {
    'Pattern 1': {
        'resnet50': (resnet50 + "_config1/best_model.pth"),
        'resnet101': (resnet101 + "_config1/best_model.pth"),
        'resnet152': (resnet152 + "_config1/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config1/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config1/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config1/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config1/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config1/best_model.pth")
    },
    'Pattern 2': {
        'resnet50': (resnet50 + "_config2/best_model.pth"),
        'resnet101': (resnet101 + "_config2/best_model.pth"),
        'resnet152': (resnet152 + "_config2/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config2/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config2/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config2/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config2/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config2/best_model.pth")
    },
    'Pattern 3': {
        'resnet50': (resnet50 + "_config3/best_model.pth"),
        'resnet101': (resnet101 + "_config3/best_model.pth"),
        'resnet152': (resnet152 + "_config3/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config3/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config3/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config3/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config3/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config3/best_model.pth")
    },
    'Pattern 4': {
        'resnet50': (resnet50 + "_config4/best_model.pth"),
        'resnet101': (resnet101 + "_config4/best_model.pth"),
        'resnet152': (resnet152 + "_config4/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config4/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config4/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config4/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config4/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config4/best_model.pth")
    },
    'Pattern 5': {
        'resnet50': (resnet50 + "_config5/best_model.pth"),
        'resnet101': (resnet101 + "_config5/best_model.pth"),
        'resnet152': (resnet152 + "_config5/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config5/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config5/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config5/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config5/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config5/best_model.pth")
    },
    'Pattern 6': {
        'resnet50': (resnet50 + "_config6/best_model.pth"),
        'resnet101': (resnet101 + "_config6/best_model.pth"),
        'resnet152': (resnet152 + "_config6/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config6/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config6/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config6/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config6/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config6/best_model.pth")
    },
    'Pattern 7': {
        'resnet50': (resnet50 + "_config7/best_model.pth"),
        'resnet101': (resnet101 + "_config7/best_model.pth"),
        'resnet152': (resnet152 + "_config7/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config7/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config7/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config7/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config7/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config7/best_model.pth")
    },
    'Pattern 8': {
        'resnet50': (resnet50 + "_config8/best_model.pth"),
        'resnet101': (resnet101 + "_config8/best_model.pth"),
        'resnet152': (resnet152 + "_config8/best_model.pth"),
        'resnext50_32x4d': (resnext50_32x4d + "_config8/best_model.pth"),
        'resnext101_32x8d': (resnext101_32x8d + "_config8/best_model.pth"),
        'resnext101_64x4d': (resnext101_64x4d + "_config8/best_model.pth"),
        'wide_resnet50_2': (wide_resnet50_2 + "_config8/best_model.pth"),
        'wide_resnet101_2': (wide_resnet101_2 + "_config8/best_model.pth")
    },

}


