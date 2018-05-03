from Pipeline.option import args
from Architecture.model import model

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms, utils
from torch.utils.data.dataset import random_split, ConcatDataset

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn



class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.data_frame.iloc[idx, 1:].as_matrix()
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        label = label

        return {'image': img, 'label': label}


class CenterCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size


        top = (h - new_h)//2
        left = (w - new_w)//2

        image = image[top: top + new_h,
                      left: left + new_w]

        label = label

        return {'image': image, 'label': label}


class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.FloatTensor([label[0].tolist()])}

def test():
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    for batch in test_loader:
        data = batch['image']
        target = batch['label'].view(-1)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float() , Variable(target).long()
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        _, pred = torch.max(output.data, 1)
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct, 100*correct / len(test_loader.dataset)

if args.model=='Alexnet':
    size = 224
else:
    size = 299


transformed_dataset_cw_1 = CustomDataset(csv_file='./Data/testing/im8_labels.csv',
                                           root_dir='./Data/testing/images8/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_2 = CustomDataset(csv_file='./Data/testing/im9_labels.csv',
                                           root_dir='./Data/testing/images9/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

test_dataset = ConcatDataset( [transformed_dataset_cw_1, transformed_dataset_cw_2] )

# load testing data loader
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, drop_last=False, **kwargs)


corr, acc = test()