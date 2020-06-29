import sys
import torch
import numpy
import random
from trainer import Trainer
from config import get_config
from utils import save_config, load_teachers, get_dataset, get_devices, copy_first_level
from data_loader import get_test_loader, get_train_loader


def main(config):
    if config.previous_level_from != 'self':
        copy_first_level(
            src_exp=config.previous_level_from,
            dst_exp=config.experiment_name
        )
        sys.exit(
            f'First level copied from {config.previous_level_from}, done for the first level...')

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    numpy.random.seed(config.random_seed)
    random.seed(config.random_seed)

    if not config.disable_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_gpu = True
    else:
        use_gpu = False
    devices = get_devices(len(config.model_names), use_gpu)

    # instantiate data load_teachers
    data_dict = get_dataset(config.dataset, config.data_dir,
                            'test', config.padding, config.padding_mode)
    config.num_classes = data_dict['num_classes']
    print('Preparing the test loader...')
    test_loader = get_test_loader(
        dataset=data_dict['dataset'],
        num_classes=data_dict['num_classes'],
        batch_size=config.batch_size,
        use_gpu=use_gpu,
        progress_bar=config.progress_bar
    )

    print('Preparing the validation loader...')
    if 'cifar' in config.dataset:
        valid_loader = test_loader

    else:
        data_dict = get_dataset(
            config.dataset, config.data_dir, 'val', config.padding, config.padding_mode)
        valid_loader = get_test_loader(
            dataset=data_dict['dataset'],
            num_classes=data_dict['num_classes'],
            batch_size=config.batch_size,
            use_gpu=use_gpu,
            progress_bar=config.progress_bar
        )

    if config.train:
        data_dict = get_dataset(
            config.dataset, config.data_dir, 'train', config.padding, config.padding_mode)
        teachers = load_teachers(config, devices, data_dict['img_size'])
        print('Preparing the training loader...')
        train_loader = get_train_loader(dataset=data_dict['dataset'],
                                        num_classes=data_dict['num_classes'],
                                        batch_size=config.batch_size,
                                        random_seed=config.random_seed,
                                        shuffle=config.shuffle,
                                        model_num=len(config.model_names),
                                        teachers=teachers,
                                        unlabel_split=config.unlabel_split,
                                        use_gpu=use_gpu,
                                        progress_bar=config.progress_bar)
    else:
        train_loader = None

    # instantiate trainer
    trainer = Trainer(config, train_loader, valid_loader, test_loader)

    if config.train:
        save_config(config)
        trainer.train()

    if config.test:
        trainer.test(config)
        trainer.test(config, best=True)


if __name__ == '__main__':
    config = get_config()
    main(config)
