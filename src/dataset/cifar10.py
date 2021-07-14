import os
import pickle
import torch

import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset

def get_dataset(args, is_training=False):
    return Cifar10(args.root,
                    transforms=None,
                    is_training=is_training)

def get_dataloader(args, is_training=False):
    dataset = get_dataset(args, is_training)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            drop_last=True,
                                            shuffle=True)
    return dataloader

class Cifar10(Dataset):
    def __init__(self, root, transforms=None, is_training=False):
        self.is_training = is_training
        self.transforms = transforms
        self.images, self.labels = self.get_cifar_data(root)
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def get_cifar_data(self, root):
        images, labels = [], []
        if self.is_training:
            filename = "data_batch"
        else:
            filename = "test_batch"

        batches = [os.path.join(root, databatch) 
                    for databatch in os.listdir(root) 
                    if filename in databatch]
        for batch in batches:
            with open(batch, 'rb') as f:
                datadict = pickle.load(f, encoding='latin1')
                images.append(datadict['data'])
                labels.extend(datadict['labels'])
        images = np.vstack(images).reshape(-1, 3, 32, 32)
        images = images.transpose((0, 2, 3, 1))
        return images, labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # get the image and label
        image = self.images[index]
        label = self.labels[index]
        # if training apply transforms
        if self.is_training:
            if self.transforms is None:
                self.transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            image = self.transforms(image)
        return {"image" : image, "label" : label, "class" : self.classes[label]}