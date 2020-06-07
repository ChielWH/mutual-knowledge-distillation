import torch
from trainer import Trainer
from config import get_config
from utils import save_config, load_teachers, get_dataset, get_devices
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
    devices = get_devices(len(config.model_names), use_gpu)

    # instantiate data loaders
    data_dict = get_dataset(config.dataset, config.data_dir, 'test')
    kwargs.update(data_dict)
    config.num_classes = data_dict['num_classes']
    test_loader = get_test_loader(
        batch_size=config.batch_size,
        **kwargs
    )

    if 'cifar' in config.dataset:
        valid_loader = test_loader

    else:
        valid_loader = get_test_loader(
            batch_size=config.batch_size,
            **kwargs
        )

    if config.is_train:
        data_dict = get_dataset(config.dataset, config.data_dir, 'train')
        teachers = load_teachers(config, devices, data_dict['img_size'])
        kwargs.update(data_dict)
        train_loader = get_train_loader(batch_size=config.batch_size,
                                        padding=config.padding,
                                        padding_mode=config.padding_mode,
                                        random_seed=config.random_seed,
                                        shuffle=config.shuffle,
                                        model_num=len(config.model_names),
                                        teachers=teachers,
                                        cuda=use_gpu,
                                        **kwargs)
    else:
        train_loader = None

    # instantiate trainer
    trainer = Trainer(config, train_loader, valid_loader, test_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    # else:
    trainer.test(config)
    trainer.test(config, best=True)


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
