import inspect
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from utils import isnotebook

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class PsuedoLabelledDataset(torch.utils.data.Dataset):
    """Psuedo labelled dataset."""

    def __init__(self,
                 data_loader,
                 num_classes,
                 trans,
                 batch_size,
                 train,
                 model_num,
                 teachers=[],
                 cuda=False):

        self.named_dataset = []
        self.teacher_num = len(teachers)

        device = torch.device('cuda') if cuda else torch.device('cpu')

        if teachers:
            # prepare and assert the teachers behaviour
            sample_images = next(iter(data_loader))[0].to(device)
            for teacher in teachers:
                teacher = teacher.to(device)
                teacher.eval()
                assert teacher(sample_images).shape == (
                    batch_size, num_classes)

            # create the psuedo labels
            with tqdm(total=len(data_loader)) as pbar:
                with torch.no_grad():
                    for images, labels in data_loader:
                        if cuda:
                            images = images.to(device)
                        psuedo_labels = torch.stack(
                            [teacher(images).cpu()
                             for teacher in teachers], -1
                        )
                        if cuda:
                            images = images.cpu()
                            psuedo_labels = psuedo_labels.cpu()
                        for image, label, psuedo_label in zip(images,
                                                              labels,
                                                              psuedo_labels):
                            self.named_dataset.append(
                                tuple([image, label, psuedo_label]))
                        pbar.update(1)

        else:
            with tqdm(total=len(data_loader)) as pbar:
                dummy_psuedo_label = torch.empty(num_classes, model_num)
                for images, labels in data_loader:
                    for image, label in zip(images, labels):
                        self.named_dataset.append(
                            tuple([image, label, dummy_psuedo_label]))
                    pbar.update(1)

    def __len__(self):
        return len(self.named_dataset)

    def __getitem__(self, idx):
        return self.named_dataset[idx]


def get_train_loader(data_loader,
                     batch_size,
                     img_size,
                     padding,
                     padding_mode,
                     num_classes,
                     model_num=3,
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
    # define transforms
    trans = transforms.Compose([
        transforms.RandomCrop(
            size=img_size - padding,
            padding=padding,
            padding_mode=padding_mode),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print('Preparing the training loader...')
    dataset = PsuedoLabelledDataset(data_loader=data_loader,
                                    trans=trans,
                                    batch_size=batch_size,
                                    train=True,
                                    model_num=model_num,
                                    num_classes=num_classes,
                                    teachers=teachers,
                                    cuda=cuda)

    if shuffle:
        np.random.seed(random_seed)

    data_laoder = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_laoder


def get_test_loader(data_loader,
                    batch_size,
                    img_size,
                    num_classes,
                    teachers=[],
                    model_num=3,
                    cuda=False,
                    random_seed=2020,
                    shuffle=True,
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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])
    ])

    print('Preparing the testing loader...')
    dataset = PsuedoLabelledDataset(data_loader=data_loader,
                                    trans=trans,
                                    batch_size=batch_size,
                                    train=False,
                                    model_num=model_num,
                                    num_classes=num_classes,
                                    teachers=teachers,
                                    cuda=cuda)

    if shuffle:
        np.random.seed(random_seed)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader
