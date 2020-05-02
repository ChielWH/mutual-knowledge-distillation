import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='Mutual Knowledge Distillation')


def str2bool(v):
  return v.lower() in ('true', '1')


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--num_classes', type=int, default=100,
                      help='Number of classes to classify')
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--padding', type=int, default=4,
                      help='number of pixels to pad around the image')
data_arg.add_argument('--pad_mode', type=str, default='constant',
                      help='padding mode')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--pin_memory', type=str2bool, default=True,
                      help='whether to copy tensors into CUDA pinned memory')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train indices')
data_arg.add_argument('--download', type=str2bool, default=False,
                      help='Whether to download the dataset')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
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
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--gamma', type=float, default=0.1,
                       help='value of learning rate decay')
train_arg.add_argument('--lr_step', type=int, default=60,
                       help='number of epochs after which the lr is multiplied with gamma')
train_arg.add_argument('--lambda_a', type=float, default=0.5,
                       help='balance between sl signal and the additional signals')
train_arg.add_argument('--lambda_b', type=float, default=0.5,
                       help='balance between kd signal and the dml signals')
train_arg.add_argument('--temperature', type=float, default=3,
                       help='softmax temperature')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--disable_cuda', type=str2bool, default=False,
                      help="Whether disable the GPU, if False, GPU will be utilized if available")
misc_arg.add_argument('--best', type=str2bool, default=False,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/cifar100',
                      help='Directory in which data is stored')
misc_arg.add_argument('--use_wandb', type=str2bool, default=False,
                      help='Whether to use Weights and Biases for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--experiment_name', type=str, default='test_experiment',
                      help='Name of the experiment, used to store all logs and checkpoints')
misc_arg.add_argument('--experiment_level', type=int, default=1,
                      help='Level in the experiment, only succesive levels can be passed in here, level 1 must exist before level 2 can be build')
misc_arg.add_argument('--model_names', nargs='+', default=['RN14', 'MN20', 'EFB0'],
                      help='The abbreviation of the model name with size indicator',
                      choices=['EFB0', 'EFB1', 'EFB2', 'EFB3', 'EFB4', 'EFB5', 'EFB6', 'EFB7',
                               'MN20', 'MN30', 'MN40', 'MN50', 'MN60', 'MN70', 'MN85', 'MN100',
                               'RN14', 'RN20', 'RN32', 'RN44', 'RN50', 'RN110', 'RN200', 'RN302',
                               'CN2', 'CN4', 'CN6', 'CN28', 'CN10'])


def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
