import torch
import random
import numpy as np
from utils import isnotebook, get_devices, accuracy, RunningAverageMeter
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class PsuedoLabelledDataset(torch.utils.data.Dataset):
    """Psuedo labelled dataset."""

    def __init__(self,
                 dataset,
                 num_classes,
                 batch_size,
                 train,
                 model_num,
                 teachers=[],
                 unlabel_split=.0,
                 use_gpu=False,
                 progress_bar=True):

        self.data = []
        self.teacher_num = len(teachers)

        devices = get_devices(model_num, use_gpu)

        if unlabel_split:
            len_ds = len(dataset)
            all_indices = range(len_ds)
            subset_indices = set(random.sample(
                all_indices, int(len_ds * unlabel_split)))
            counter = 0

        if progress_bar:
            pbar = tqdm(total=len(dataset), smoothing=.005)

        if teachers:
            accuracies = []
            # prepare and assert the teachers behaviour
            _batch = dataset[0][0].unsqueeze(0)
            for teacher, device in zip(teachers, devices):
                teacher = teacher.to(device)
                batch = _batch.clone().to(device)
                teacher.eval()
                sample_classes = teacher(batch).shape[1]
                assert sample_classes == num_classes, f"Num classes of the output is {sample_classes}, {num_classes} required"
                accuracies.append(RunningAverageMeter())

            # create the psuedo labels
            with torch.no_grad():
                for image, label in dataset:
                    unlabelled = 1
                    if unlabel_split:
                        if counter in subset_indices:
                            unlabelled = 0
                        counter += 1
                    _psuedo_labels = []
                    for i, (teacher, device) in enumerate(zip(teachers, devices)):
                        if use_gpu:
                            image = image.to(device)
                        # add dimension to comply with the desired input dimension (batch of single image)
                        pred = teacher(image.unsqueeze(0)).cpu()
                        _psuedo_labels.append(pred)
                    image = image.cpu()
                    psuedo_labels = torch.stack(_psuedo_labels, -1).squeeze(0)

                    self.data.append((image, label, psuedo_labels, unlabelled))

                    acc_at_1 = accuracy(
                        pred,
                        torch.tensor([[label]]),
                        topk=(1,)
                    )[0]
                    accuracies[i].update(acc_at_1.item())

                    if progress_bar:
                        pbar.update(1)

            print(
                f"Accurcies of loaded models are {' '.join([round(acc.avg, 2) for acc in accuracies])}, respectively")

        else:
            dummy_psuedo_label = torch.empty(num_classes, model_num)
            for image, label in dataset:
                unlabelled = 1
                if unlabel_split:
                    if counter in subset_indices:
                        unlabelled = 0
                    counter += 1
                self.data.append(
                    (image, label, dummy_psuedo_label, unlabelled)
                )
                if progress_bar:
                    pbar.update(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_train_loader(dataset,
                     batch_size,
                     num_classes,
                     model_num=3,
                     teachers=[],
                     unlabel_split=.0,
                     use_gpu=False,
                     random_seed=2020,
                     shuffle=True,
                     num_workers=4,
                     progress_bar=True):
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
    - dataset: train set iterator.
    """
    _dataset = PsuedoLabelledDataset(dataset=dataset,
                                     batch_size=batch_size,
                                     train=True,
                                     model_num=model_num,
                                     num_classes=num_classes,
                                     teachers=teachers,
                                     unlabel_split=unlabel_split,
                                     use_gpu=use_gpu,
                                     progress_bar=progress_bar)

    def _worker_init_fn(x):
        seed = random_seed + x
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return

    data_loader = torch.utils.data.DataLoader(_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=use_gpu,
                                              worker_init_fn=_worker_init_fn)

    return data_loader


def get_test_loader(dataset,
                    batch_size,
                    num_classes,
                    teachers=[],
                    model_num=3,
                    use_gpu=False,
                    random_seed=2020,
                    shuffle=True,
                    num_workers=4,
                    progress_bar=True):
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
    - dataset: test set iterator.
    """
    _dataset = PsuedoLabelledDataset(dataset=dataset,
                                     batch_size=batch_size,
                                     train=False,
                                     model_num=model_num,
                                     num_classes=num_classes,
                                     teachers=teachers,
                                     unlabel_split=.0,
                                     use_gpu=use_gpu,
                                     progress_bar=progress_bar)

    def _worker_init_fn(x):
        seed = random_seed + x
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return

    data_loader = torch.utils.data.DataLoader(_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=use_gpu,
                                              worker_init_fn=_worker_init_fn)

    return data_loader
