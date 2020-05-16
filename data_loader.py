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
                 data_dir,
                 trans,
                 pad,
                 pad_mode,
                 batch_size,
                 train,
                 model_num,
                 cifar='100',
                 download=False,
                 teachers=[],
                 cuda=False):
        """
        """
        self.named_dataset = []
        self.teacher_num = len(teachers)

        device = torch.device('cuda') if cuda else torch.device('cpu')
        padding = [pad] * 4

        # get the correct CIFAR dataset
        cifar_dataset = getattr(datasets, f'CIFAR{cifar}')
        dataset = cifar_dataset(root=data_dir,
                                transform=trans,
                                download=download,
                                train=train)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size)

        # get the fold by inspecting the function that instantiates the class
        caller = inspect.stack()[1][3]
        fold = 'train' if caller == 'get_train_loader' else 'test'
        if teachers:
            # prepare and assert the teachers behaviour
            sample_images = next(iter(data_loader))[0].to(device)
            sample_images = F.pad(input=sample_images,
                                  pad=padding,
                                  mode=pad_mode)
            for teacher in teachers:
                teacher = teacher.to(device)
                teacher.eval()
                assert teacher(sample_images).shape == (batch_size, int(cifar))

            # create the psuedo labels
            with tqdm(total=len(data_loader) * batch_size) as pbar:
                pbar.set_description(
                    f'Preparing {fold}set: creating the psuedo labels')
                with torch.no_grad():
                    for images, labels in data_loader:
                        images = F.pad(
                            input=images,
                            pad=padding,
                            mode=pad_mode
                        )
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
                        pbar.update(batch_size)

        else:
            with tqdm(total=len(data_loader) * batch_size) as pbar:
                pbar.set_description(
                    f'Preparing {fold}set')
                dummy_psuedo_label = torch.empty(int(cifar), model_num)
                for images, labels in data_loader:
                    images = F.pad(
                        input=images,
                        pad=padding,
                        mode=pad_mode
                    )
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
                     pad,
                     pad_mode,
                     model_num,
                     cifar='100',
                     download=False,
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
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PsuedoLabelledDataset(data_dir=data_dir,
                                    trans=trans,
                                    pad=pad,
                                    pad_mode=pad_mode,
                                    batch_size=batch_size,
                                    cifar=cifar,
                                    download=download,
                                    train=True,
                                    model_num=model_num,
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


def get_test_loader(data_dir,
                    batch_size,
                    pad,
                    pad_mode,
                    model_num,
                    cifar='100',
                    download=False,
                    teachers=[],
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

    # load dataset
    dataset = PsuedoLabelledDataset(data_dir=data_dir,
                                    trans=trans,
                                    pad=pad,
                                    pad_mode=pad_mode,
                                    batch_size=batch_size,
                                    cifar=cifar,
                                    download=download,
                                    train=False,
                                    model_num=model_num,
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
