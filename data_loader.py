# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:12:10 2019

@author: chxy
"""
from tqdm import tqdm
import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms


class PsuedoLabelledDataset(torch.utils.data.Dataset):
    """Psuedo labelled dataset."""

    def __init__(self,
                 data_dir,
                 batch_size=16,
                 cifar='100',
                 download=False,
                 train=True,
                 teachers=[],
                 cuda=False):
        """
        """
        self.named_dataset = []
        self.teacher_num = len(teachers)

        device = torch.device('cuda') if cuda else torch.device('cpu')

        # define transforms
        trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # get the correct CIFAR dataset
        cifar_dataset = getattr(datasets, f'CIFAR{cifar}')
        dataset = cifar_dataset(root=data_dir,
                                transform=trans,
                                download=download,
                                train=train)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size)

        if teachers:

            # prepare and assert the teachers behaviour
            sample_batch = next(iter(data_loader))[0].to(device)
            for teacher in teachers:
                teacher = teacher.to(device)
                teacher.eval()
                assert teacher(sample_batch).shape == (batch_size, int(cifar))

            # create the psuedo labels
            with tqdm(total=len(data_loader) * batch_size) as pbar:
                pbar.set_description(
                    f'Preparing dataset: creating the psuedo labels from {self.teacher_num} teachers')
                with torch.no_grad():
                    for images, labels in data_loader:
                        if cuda:
                            images = images.to(device)
                        psuedo_labels = torch.stack([teacher(images).cpu()
                                                     for teacher in teachers], -1)
                        if cuda:
                            images = images.cpu()
                            psuedo_labels = psuedo_labels.cpu()
                        for image, label, psuedo_label in zip(images, labels, psuedo_labels):
                            self.named_dataset.append(
                                tuple([image, label, psuedo_label]))
                        pbar.update(batch_size)

        # else,
        else:
            with tqdm(total=len(data_loader) * batch_size) as pbar:
                pbar.set_description(
                    f'Preparing datatset')
                # to let it be compatible with the collate_fn for naming tensors
                dummy_psuedo_label = torch.empty(int(cifar), 1)
                for images, labels in data_loader:
                    for image, label in zip(images, labels):
                        self.named_dataset.append(
                            tuple([image, label, dummy_psuedo_label]))
                    pbar.update(batch_size)

    def __len__(self):
        return len(self.named_dataset)

    def __getitem__(self, idx):
        return self.named_dataset[idx]


def get_train_loader(data_dir,
                     batch_size,
                     cifar='100',
                     download=False,
                     fold='train',
                     teachers=[],
                     cuda=False,
                     random_seed=2020,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    train iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    Returns
    -------
    - data_loader: train set iterator.
    """
    if fold == 'train':
        train_bool = True
    elif fold == 'valid':
        train_bool = False
    else:
        raise ValueError("fold must be either 'train' or 'valid'")

    # load dataset
    dataset = PsuedoLabelledDataset(data_dir=data_dir,
                                    batch_size=batch_size,
                                    cifar=cifar,
                                    download=download,
                                    train=train_bool,
                                    teachers=teachers,
                                    cuda=cuda)

    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    return train_loader


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    trans = transforms.Compose([
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])  # 归一化
    ])

    # load dataset
    dataset = datasets.CIFAR100(
        data_dir, train=False, download=False, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
