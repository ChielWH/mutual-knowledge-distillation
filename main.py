# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:07:46 2019

@author: chxy
"""

import torch

from trainer import Trainer
from config import get_config
from utils import save_config, load_teachers
from data_loader import get_test_loader, get_train_loader


def main(config):
    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if not config.disable_cuda and torch.cuda.is_available():
        use_gpu = True
        torch.cuda.manual_seed_all(config.random_seed)
        kwargs = {'num_workers': config.num_workers,
                  'pin_memory': config.pin_memory}
    else:
        use_gpu = False

    teachers = load_teachers(config, use_gpu, 40)

    # instantiate data loaders
    test_data_loader = get_test_loader(data_dir=config.data_dir,
                                       batch_size=config.batch_size,
                                       pad=config.padding,
                                       pad_mode=config.pad_mode,
                                       cuda=use_gpu,
                                       teachers=teachers,
                                       model_num=len(config.model_names),
                                       **kwargs)

    if config.is_train:
        train_data_loader = get_train_loader(data_dir=config.data_dir,
                                             batch_size=config.batch_size,
                                             pad=config.padding,
                                             pad_mode=config.pad_mode,
                                             random_seed=config.random_seed,
                                             shuffle=config.shuffle,
                                             model_num=len(config.model_names),
                                             teachers=teachers,
                                             cuda=use_gpu,
                                             **kwargs)

        data_loader = (train_data_loader, test_data_loader)
    else:
        data_loader = test_data_loader

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
