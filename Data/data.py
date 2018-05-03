from Pipeline.option import args

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms, utils
from torch.utils.data.dataset import random_split, ConcatDataset

#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
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
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

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

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        label = label

        return {'image': img, 'label': label}


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

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

        #top = np.random.randint(0, h - new_h)
        #left = np.random.randint(0, w - new_w)

        top = (h - new_h)//2
        left = (w - new_w)//2

        image = image[top: top + new_h,
                      left: left + new_w]

        label = label

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.FloatTensor([label[0].tolist()])}


if args.model=='Alexnet':
    size = 224
else:
    size = 299


transformed_dataset_cw_1 = CustomDataset(csv_file='./Data/new_man/im1_labels.csv',
                                           root_dir='./Data/new_man/images1/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_2 = CustomDataset(csv_file='./Data/new_man/im2_labels.csv',
                                           root_dir='./Data/new_man/images2/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_3 = CustomDataset(csv_file='./Data/new_man/im3_labels.csv',
                                           root_dir='./Data/new_man/images3/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_4 = CustomDataset(csv_file='./Data/new_man/im4_labels.csv',
                                           root_dir='./Data/new_man/images4/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))


transformed_dataset_cw_5 = CustomDataset(csv_file='./Data/new_man/im5_labels.csv',
                                           root_dir='./Data/new_man/images5/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))


transformed_dataset_cw_6 = CustomDataset(csv_file='./Data/new_man/im6_labels.csv',
                                           root_dir='./Data/new_man/images6/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_7 = CustomDataset(csv_file='./Data/Manual_updated/man_ccw.csv',
                                           root_dir='./Data/Manual_updated/CCW_updt/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_8 = CustomDataset(csv_file='./Data/Manual_updated/new_dataset/man1_labels.csv',
                                           root_dir='./Data/Manual_updated/new_dataset/man1_res/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))
transformed_dataset_cw_9 = CustomDataset(csv_file='./Data/Manual_updated/new_dataset/man2_labels.csv',
                                           root_dir='./Data/Manual_updated/new_dataset/man2_res/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_10 = CustomDataset(csv_file='./Data/Manual_updated/new_dataset/cw_comp1.csv',
                                           root_dir='./Data/Manual_updated/new_dataset/cw_1_resize/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))

transformed_dataset_cw_11 = CustomDataset(csv_file='./Data/Manual_updated/new_dataset/cw_comp2.csv',
                                           root_dir='./Data/Manual_updated/new_dataset/cw_2_resize/',
                                           transform=transforms.Compose([
                                               Rescale(size),
                                               CenterCrop(size),
                                               ToTensor()
                                           ]))



if args.aug == 1:
  transformed_dataset_cw_1_aug = CustomDataset(csv_file='./Data/new_man/im1_labels.csv',
                                             root_dir='./Data/new_man/images1_aug/',
                                             transform=transforms.Compose([
                                                 Rescale(size),
                                                 CenterCrop(size),
                                                 ToTensor()
                                             ]))

  transformed_dataset_cw_2_aug = CustomDataset(csv_file='./Data/new_man/im2_labels.csv',
                                             root_dir='./Data/new_man/images2_aug/',
                                             transform=transforms.Compose([
                                                 Rescale(size),
                                                 CenterCrop(size),
                                                 ToTensor()
                                             ]))

  transformed_dataset_cw_3_aug = CustomDataset(csv_file='./Data/new_man/im3_labels.csv',
                                             root_dir='./Data/new_man/images3_aug/',
                                             transform=transforms.Compose([
                                                 Rescale(size),
                                                 CenterCrop(size),
                                                 ToTensor()
                                             ]))

  transformed_dataset_cw_4_aug = CustomDataset(csv_file='./Data/new_man/im4_labels.csv',
                                             root_dir='./Data/new_man/images4_aug/',
                                             transform=transforms.Compose([
                                                 Rescale(size),
                                                 CenterCrop(size),
                                                 ToTensor()
                                             ]))


  transformed_dataset_cw_5_aug = CustomDataset(csv_file='./Data/new_man/im5_labels.csv',
                                             root_dir='./Data/new_man/images5_aug/',
                                             transform=transforms.Compose([
                                                 Rescale(size),
                                                 CenterCrop(size),
                                                 ToTensor()
                                             ]))

  transformed_dataset_cw_6_aug = CustomDataset(csv_file='./Data/new_man/im6_labels.csv',
                                             root_dir='./Data/new_man/images6_aug/',
                                             transform=transforms.Compose([
                                                 Rescale(size),
                                                 CenterCrop(size),
                                                 ToTensor()
                                             ]))



if args.aug == 1:
  transformed_dataset = ConcatDataset( [transformed_dataset_cw_1, transformed_dataset_cw_2, transformed_dataset_cw_3, transformed_dataset_cw_4, transformed_dataset_cw_5, transformed_dataset_cw_6,
    transformed_dataset_cw_1_aug, transformed_dataset_cw_2_aug, transformed_dataset_cw_3_aug, transformed_dataset_cw_4_aug, transformed_dataset_cw_5_aug, transformed_dataset_cw_6_aug ] )

else:
  transformed_dataset = ConcatDataset( [transformed_dataset_cw_1, transformed_dataset_cw_2, transformed_dataset_cw_3, transformed_dataset_cw_4, transformed_dataset_cw_5, transformed_dataset_cw_6, transformed_dataset_cw_7, transformed_dataset_cw_8, transformed_dataset_cw_9, transformed_dataset_cw_10, transformed_dataset_cw_11 ] )

train_len = np.int( args.train_percent * len(transformed_dataset) )

train_data , test_data = random_split( transformed_dataset , [train_len , len(transformed_dataset)-train_len ] )

# load trainig data loader
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
train_loader = DataLoader(train_data,
                          batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)

# load testing data loader
test_loader = DataLoader(test_data,
                         batch_size=args.test_batch_size, shuffle=True, drop_last=False, **kwargs)
