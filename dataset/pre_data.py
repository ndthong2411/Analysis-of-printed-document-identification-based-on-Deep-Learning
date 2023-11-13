import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
class Dataset():
    def __init__(self, input_size, root, dataframe, mode='train'):
        self.input_size = input_size
        self.root = root
        self.dataframe= dataframe
        self.mode = mode

    def __getitem__(self, idx):
        
        img_path= os.path.join(self.root, self.dataframe.iloc[idx].path.replace('\\','/'))
        if os.path.exists(img_path):
            if self.mode == 'train':

                img, target = imageio.imread(img_path), self.dataframe.iloc[idx]['printer']
        
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')
                img = transforms.Resize(self.input_size + 16)(img)  # old 16
                img = transforms.RandomRotation(20)(img)
                img = transforms.RandomVerticalFlip()(img)
                img = transforms.RandomCrop(self.input_size)(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            else:
                img, target = imageio.imread(img_path), self.dataframe.iloc[idx]['printer']
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')
                img = transforms.Resize(self.input_size + 16)(img)  # old: 16
                img = transforms.CenterCrop(self.input_size)(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.dataframe)
        else:
            return len(self.dataframe)