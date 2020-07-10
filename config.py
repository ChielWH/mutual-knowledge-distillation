import os
import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='Mutual Knowledge Distillation')


def str2bool(v):
    return v.lower() in {'true', '1'}


def str2list(v):
    if ':' in v:
        return v.split(':')
    return v


def str2listorbool(v):
    if ':' in v:
        return v.split(':')
    return v.lower() in {'true', '1'}


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--dataset', type=str, default='cifar100',
                      help='The set of images used to train, validate and test the model',
                      choices=['cifar10', 'cifar100', 'tiny-imagenet-200'])
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--padding', type=int, default=4,
                      help='number of pixels to pad around the image, see: https://pytorch.org/docs/stable/torchvision/transforms.html -> torchvision.transforms.Pad')
data_arg.add_argument('--padding_mode', type=str, default='reflect',
                      help='padding mode (constant is filled with 0),  see: https://pytorch.org/docs/stable/torchvision/transforms.html -> torchvision.transforms.Pad',
                      choices=['constant', 'reflect', 'edge', 'symmetric'])
data_arg.add_argument('--num_workers', type=int, default=0,
                      help='# of subprocesses to use for data loading, guideline is 4 * #GPUs (0 is adviced if reproducability is desirable)', )
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train indices')
data_arg.add_argument('--unlabel_split', type=float, default=.0,
                      help='Fraction of dataset to discard the labels from')
data_arg.add_argument('--discard_unlabelled', type=str2bool, default=False,
                      help='Fraction of dataset to discard the labels from')


# Training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum value')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.1,
                       help='Initial learning rate value')
train_arg.add_argument('--weight_decay', type=float, default=5e-4,
                       help='value of weight dacay for regularization')
train_arg.add_argument('--nesterov', type=str2bool, default=True,
                       help='Whether to use Nesterov momentum')
train_arg.add_argument('--gamma', type=float, default=0.1,
                       help='value of learning rate decay')
train_arg.add_argument('--lr_step', type=int, default=60,
                       help='number of epochs after which the lr is multiplied with gamma')
train_arg.add_argument('--lambda_a', type=float, default=0.5,
                       help='balance between sl signal and the additional signals, 1.0 is solely SL signal, 0.0 is no signal (not recommanded)')
train_arg.add_argument('--lambda_b', type=float, default=0.5,
                       help='balance between DML signal and the KD signals, 1.0 is no KD signal, 0.0 is no DML signal')
train_arg.add_argument('--temperature', type=float, default=3,
                       help='softmax temperature')

# Miscellaneous params
misc_arg = add_argument_group('Misc.')
train_arg.add_argument('--train', type=str2bool, default=True,
                       help='Whether to train the model or not')
train_arg.add_argument('--test', type=str2bool, default=True,
                       help='Whether to test the model or not')
misc_arg.add_argument('--test_script', type=str2bool, default=False)
misc_arg.add_argument('--hp_search', type=str2listorbool, default=False,
                      help='Wether or not this experiment is part of a hyperparameter search, provide at least on of the choices, the experiment level is named accordingly (level_1_hp1=value_hp2=value_etc) (default: %(default)s)')
misc_arg.add_argument('--hp_search_from_static', type=str2bool, default=False,
                      help="Whether or not to use the `sweep` hp search from wandb (default: %(default)s)")
misc_arg.add_argument('--use_sweep', type=str2bool, default=False,
                      help="Whether or not to use the `sweep` hp search from wandb (default: %(default)s)")
misc_arg.add_argument('--disable_cuda', type=str2bool, default=False,
                      help="Whether disable the GPU, if False, GPU will be utilized if available (default: %(default)s)")
misc_arg.add_argument('--best', type=str2bool, default=False,
                      help='Load best model or most recent for testing (default: %(default)s)')
misc_arg.add_argument('--random_seed', type=int, default=0,
                      help='Seed to ensure reproducibility (default: %(default)s)')
misc_arg.add_argument('--data_dir', type=str, default='/scratch/cifar100',
                      help='Directory in which data is stored (default: %(default)s)')
misc_arg.add_argument('--use_wandb', type=str2bool, default=False,
                      help='Whether to use Weights and Biases for visualization (default: %(default)s)')
misc_arg.add_argument('--progress_bar', type=str2bool, default=False,
                      help='Whether to log the progress of the training phase (default: %(default)s)')
misc_arg.add_argument('--experiment_name', type=str, default='test_experiment',
                      help="Name of the experiment, used to store all logs and checkpoints (default: %(default)s)")
misc_arg.add_argument('--experiment_level', type=int, default=1,
                      help='Level in the experiment, only succesive levels can be passed in here, level 1 must exist before level 2 can be build (default: %(default)s)')
misc_arg.add_argument('--previous_level_from', type=str, default='self',
                      help='Wether or not to start from the second level and copy the first level from another experiment (default: %(default)s)',
                      choices=set(os.listdir('experiments') + ['self']).difference(set(['.DS_Store', 'README.md'])))
misc_arg.add_argument('--model_names', type=str2list, default='RN14:MN20:CN2',
                      help='The abbreviation of the model name with size indicator (default: %(default)s)')


def make_level_name(config):
    level_name = f'level_{config.experiment_level}'
    if config.hp_search:
        level_name += '_' + ','.join(
            [f'{hp}={getattr(config, hp)}' for hp in config.hp_search])
    return level_name


def get_config():
    config, unparsed = parser.parse_known_args()

    # assert that the passed arguments are correct and will not conflict later
    unparsed_args = [arg for arg in unparsed if arg.startswith('--')]
    if unparsed_args:
        raise ValueError(
            f'Unkown arguments: {unparsed_args}, correct the arguments to proceed.')

    assert any([config.train, config.test]), \
        'At least one of `--train` or `--test` must be set to True'
    if config.use_sweep:
        assert config.use_wandb and config.hp_search

    # for testing purposes, will remove later
    if config.test_script:
        config.epochs = 3
        config.use_wandb = 0

    config.level_name = make_level_name(config)

    return config


if __name__ == '__main__':
    config = get_config()
    print(config)
