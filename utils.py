import os
import sys
import yaml
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_factories import (
    efficientnet_factory,
    mobilenetv2_factory,
    resnet_factory,
    plain_cnn_factory)


class MovingAverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def update(self, val):
        try:
            self.avg = self.alpha * self.avg + (1 - self.alpha) * val
        except AttributeError:
            # initialize average with the first provided value
            self.avg = val


class RunningAverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def prepare_dirs(config, return_dir=True):
    experiment_name = config.experiment_name.lower().replace(' ', '_')
    level = f'level_{config.experiment_level}'
    level_path = os.path.join('experiments', experiment_name, level)
    ckpt_path = os.path.join(level_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if return_dir:
        return level_path


def save_config(config):
    experiment_dir = prepare_dirs(config)
    if config.use_wandb:
        import wandb
        print(
            f'[*] Config file is stored in ./{wandb.run.dir}/config.yaml')
    else:
        param_path = os.path.join(experiment_dir, 'config_params.yaml')
        with open(param_path, 'w') as fp:
            yaml.dump(config.__dict__, fp)
        print(f'[*] Saved config file to ./{param_path}')


def print_epoch_stats(model_names, train_losses, train_accs, valid_losses, valid_accs):
    avg_tl, avg_ta, avg_vl, avg_va = [], [], [], []
    print("Epoch statistics:")
    print("model     |train loss| train acc|  val loss|   val acc|")
    print("=" * 55)
    model_stats = "{model_name:10}|{train_loss:10.3f}|{train_acc:10.3f}|{valid_loss:10.3f}|{valid_acc:10.3f}|"
    for i, model_name in enumerate(model_names):
        print(
            model_stats.format(
                model_name=model_name,
                train_loss=train_losses[i].avg,
                train_acc=train_accs[i].avg,
                valid_loss=valid_losses[i].avg,
                valid_acc=valid_accs[i].avg
            )
        )
        avg_tl.append(train_losses[i].avg)
        avg_ta.append(train_accs[i].avg)
        avg_vl.append(valid_losses[i].avg)
        avg_va.append(valid_accs[i].avg)

    print("-" * 55)
    print(
        model_stats.format(
            model_name="Average",
            train_loss=np.array(avg_tl).mean(),
            train_acc=np.array(avg_ta).mean(),
            valid_loss=np.array(avg_vl).mean(),
            valid_acc=np.array(avg_va).mean()
        )
    )


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell in ['ZMQInteractiveShell', 'Shell']:
            return True   # Jupyter notebook, Colab or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def infer_input_size(batch):
    return batch[0].shape[2]


def infer_model_desc(model_name):
    name_dict = {'EF': 'EfficientNet',
                 'MN': 'MobileNetV2',
                 'RN': 'ResNet',
                 'CN': 'VGG like CNN'
                 }
    mn_size_dict = {
        str(size): f'scale={int(size)/100}' for size in range(10, 305, 5)}

    arch = model_name[:2]
    architecture = name_dict[arch]
    size_indicator = model_name[2:]
    if arch == 'MN':
        size_indicator = mn_size_dict[size_indicator]

    return architecture, size_indicator


def load_teachers(config, devices, input_size):
    level = config.experiment_level
    if level < 1:
        raise ValueError('experiment_level must be an integer of 1 or higher')
    elif level == 1:
        return []
    elif level > 1:
        experiment_name = config.experiment_name.lower().replace(' ', '_')
        prev_dir_name = f'experiments/{experiment_name}/level_{level - 1}/ckpt/'
        assert os.path.exists(
            prev_dir_name), "Can only load teacher models from levels of which all previous levels have been run."
        ckpts = os.listdir(prev_dir_name)
        bests = list(sorted([path for path in ckpts if 'best' in path]))
        teachers = []
        model_names = [f_name.split('_')[1] for f_name in bests]
        assert len(model_names) == len(
            config.model_names), f"Can not find all model checkpoints, found {model_names} to teach {config.model_names}"
        models = model_init_and_placement(
            model_names,
            devices,
            input_size,
            config.num_classes
        )
        assert len(models) == len(
            config.model_names), f'{len(models)} != {len(config.model_names)}'
        for i, f_name in enumerate(bests):
            print(f_name)
            state_dict = torch.load(prev_dir_name + f_name)
            model = models[i]
            model.load_state_dict(state_dict['model_state'])
            teachers.append(model)
    return teachers


def _correct_out_features(model, out_features):
    import torch
    for idx, module in reversed(list(enumerate(model.modules()))):
        if type(module) == torch.nn.modules.linear.Linear:
            in_features = module.in_features
            bias = type(module.bias) == torch.nn.parameter.Parameter
            setattr(
                model,
                list(model.named_modules())[idx][0],
                torch.nn.Linear(in_features,
                                out_features,
                                bias=bias)
            )


def model_init_and_placement(model_names, devices, input_size, num_classes):
    model_architectures = [model_name[:2] for model_name in model_names]
    size_indicators = [model_name[2:] for model_name in model_names]
    nets = []
    for model_architecture, size_indicator, device in zip(model_architectures, size_indicators, devices):
        if model_architecture == 'EF':
            net = efficientnet_factory.create_model(size_indicator)
            _correct_out_features(net, num_classes)
        elif model_architecture == 'MN':
            net = mobilenetv2_factory.create_model(size=size_indicator,
                                                   input_size=input_size,
                                                   num_classes=num_classes)
        elif model_architecture == 'RN':
            net = resnet_factory.create_model(size_indicator, num_classes)
            _correct_out_features(net, num_classes)
        elif model_architecture == 'CN':
            net = plain_cnn_factory.create_model(
                size_indicator, input_size, num_classes)
        else:
            raise ValueError(
                f'Model architecture {model_architecture} not known')
        net.to(device)
        nets.append(net)
    return nets


def get_devices(model_num, use_gpu):
    if use_gpu:
        # collect all possible gpu's
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        # duplicate the list of devices to get at least enough devices
        devices = devices * model_num
        # take a slice of length model_num
        devices = devices[:model_num]
    else:
        devices = ['cpu'] * model_num
    return devices


def get_dataset(name, data_dir, fold):
    assert name.lower() in {'cifar10', 'cifar100', 'tiny-imagenet-200'}, \
        "Only 'cifar10', 'cifar100', 'tiny-imagenet-200' are valid names fot he supported datasets"

    assert fold in {'train', 'val', 'test'}

    if not os.path.exists(data_dir):
        out = input(
            f'Could not find the data directory {data_dir}, want to create it?\n(yes/no): ')
        if out.lower() == 'yes':
            os.makedirs(data_dir)
        else:
            sys.exit('Aborting script...')

    if name.lower() == 'cifar10':
        train = fold == 'train'
        try:
            dataset = datasets.CIFAR10(
                root=data_dir,
                transform=transforms.ToTensor(),
                download=False,
                train=train,
            )

        except RuntimeError:
            out = input(
                'Could not find the dataset, want to downloading it?\n(yes/no')
            if out.lower() == 'yes':
                dataset = datasets.CIFAR10(
                    root=data_dir,
                    transform=transforms.ToTensor(),
                    download=True,
                    train=train
                )

        data_dict = {
            'data_loader': DataLoader(dataset),
            'img_size': 32,
            'num_classes': 10
        }

    elif name.lower() == 'cifar100':
        train = fold == 'train'
        try:
            dataset = datasets.CIFAR100(
                root=data_dir,
                transform=transforms.ToTensor(),
                download=False,
                train=train
            )

        except RuntimeError:
            out = input(
                'Could not find the dataset, want to downloading it?\n(yes/no')
            if out.lower() == 'yes':
                dataset = datasets.CIFAR100(
                    root=data_dir,
                    transform=transforms.ToTensor(),
                    download=True,
                    train=train
                )

        data_dict = {
            'data_loader': DataLoader(dataset),
            'img_size': 32,
            'num_classes': 100
        }

    elif name.lower() == 'tiny-imagenet-200':
        try:
            dataset = datasets.ImageFolder(
                data_dir + f'/{fold}', transform=transforms.ToTensor())
        except FileNotFoundError:
            out = input(
                f'Dataset not found at {data_dir}, want to download it?\n(yes/no): ')
            if out.lower() == 'yes':
                import tiny_imagenet_download
                tiny_imagenet_download.run()
            else:
                sys.exit(
                    'Aborting script, provide correct data_dir argument or download the dataset next time to proceed using tiny-imagenet-200.')

        data_dict = {
            'data_loader': DataLoader(dataset),
            'img_size': 64,
            'num_classes': 200
        }

    else:
        print("Only 'cifar10', 'cifar100', 'tiny-imagenet-200' are valid names fot he supported datasets")
        sys.exit('Aborting script...')

    return data_dict
