from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from glob import glob
import numpy as np
import torch.nn.functional as F
from scipy import ndimage

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class BRATS_SYN(data.Dataset):
    """Dataset class for the BRATS dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and Load the BRATS dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.load_data()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def load_data(self):
        """Load BRATS dataset"""
        
        # Load test dataset
        prefix = self.mode
        test_neg = glob(os.path.join(self.image_dir, prefix, 'negative_img', '*jpg'))
        test_pos = glob(os.path.join(self.image_dir,  prefix, 'positive_img', '*jpg'))

        for filename in test_neg:
            mask_filename = filename.replace('negative_img', 'negative_lab')
            self.test_dataset.append([filename, [0], mask_filename])

        for filename in test_pos:
            mask_filename = filename.replace('positive_img', 'positive_lab')
            self.test_dataset.append([filename, [1], mask_filename])


        # Load train dataset
        train_neg = glob(os.path.join(self.image_dir, 'train', 'negative_img', '*jpg'))
        train_pos = glob(os.path.join(self.image_dir, 'train', 'positive_img', '*jpg'))
        
        for filename in train_neg:
            mask_filename = filename.replace('negative_img', 'negative_lab')
            self.train_dataset.append([filename, [0], mask_filename])

        for filename in train_pos:
            mask_filename = filename.replace('positive_img', 'positive_lab')
            self.train_dataset.append([filename, [1], mask_filename])

        print('Finished loading the BRATS dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label, mask_filename = dataset[index]
        name = mask_filename.split('/')[-1]
        image = Image.open(filename)
        mask = Image.open(mask_filename)
        Image2Tensor = T.ToTensor()
        mask = Image2Tensor(mask)
        
        pancreas_mask = (mask > 0).numpy()
        # origin_pancreas_mask = (mask == 1).numpy()
        struct = ndimage.generate_binary_structure(2, 2)
        pancreas_mask = ndimage.binary_dilation(pancreas_mask[0], struct,iterations= 5).astype(np.float32)
        image=image.convert('RGB')
        orig_image = self.transform(image)
        image = orig_image * torch.from_numpy(pancreas_mask[None,:,:])
        label_mask = torch.from_numpy((mask >= 3.0/255.0).numpy())
        pancreas_mask = torch.from_numpy((mask == 1.0/255.0).numpy())
        gt = orig_image.clone()

        gt[0, label_mask[0]] = 0
        gt[1, label_mask[0]] = 255
        gt[2, label_mask[0]] = 0
        gt[0, pancreas_mask[0]] = 255
        gt[1, pancreas_mask[0]] = 0
        gt[2, pancreas_mask[0]] = 0
        if self.mode == 'train':
            return image, torch.FloatTensor(label)
        else:
            return image, torch.FloatTensor(label), orig_image, gt, name

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    # transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'BRATS':
        dataset = BRATS_SYN(image_dir, transform, mode)
    elif dataset == 'Directory':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True, # (mode=='train'),
                                  num_workers=num_workers)
    return data_loader