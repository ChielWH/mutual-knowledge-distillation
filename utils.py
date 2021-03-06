import os
import sys
import yaml
import shutil
import torch
import numpy as np
from torchvision import datasets, transforms
from model_factories import (
    mobilenetv2_factory,
    resnet_factory,
    plain_cnn_factory)


class AverageMeterBase(object):
    def __lt__(self, other):
        return self.avg < other

    def __le__(self, other):
        return self.avg <= other

    def __eq__(self, other):
        return self.avg == other

    def __ne__(self, other):
        return self.avg != other

    def __gt__(self, other):
        return self.avg > other

    def __ge__(self, other):
        return self.avg >= other

    def __add__(self, other):
        if type(other) in {int, float}:
            return self.avg + other
        return self.avg + other.avg

    def __sub__(self, other):
        if type(other) in {int, float}:
            return self.avg - other
        return self.avg + other.avg

    def __mul__(self, other):
        if type(other) in {int, float}:
            return self.avg - other
        return self.avg * other.avg

    def __truediv__(self, other):
        if type(other) in {int, float}:
            return self.avg - other
        return self.avg + other.avg

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def __repr__(self):
        return repr(self.avg)

    def __str__(self):
        return str(self.avg)


class MovingAverageMeter(AverageMeterBase):
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


class RunningAverageMeter(AverageMeterBase):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
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


def prepare_dirs(experiment_name, experiment_level, return_dir=True):
    level_path = os.path.join('experiments', experiment_name, experiment_level)
    ckpt_path = os.path.join(level_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if return_dir:
        return level_path


def copy_first_level(src_exp, dst_exp):
    for directory in ['ckpt', 'wandb']:
        src_path = os.path.join('experiments', src_exp, 'level_1', directory)
        dst_path = os.path.join('experiments', dst_exp, 'level_1', directory)
        if os.path.exists(dst_path) and any([file.endswith('pth.tar') for file in os.listdir(dst_path)]):
            out = input(
                f'Found checkpoint(s) of {dst_exp} at ./{dst_path}, want to overwrite it?\n(yes/no): ')
            if out.lower() == 'yes':
                shutil.rmtree(dst_path)
            else:
                continue
                # sys.exit(f'{out} is interpred as no, aborting script...')
        elif directory == 'wandb':
            run_dir = os.listdir(src_path)
            runs = [run for run in run_dir if run.startswith('run')]
            last_run = sorted(runs)[-1]
            src_path = os.path.join(src_path, last_run)
            dst_path = os.path.join(dst_path, last_run)
        shutil.copytree(src_path, dst_path, symlinks=True)


def save_config(config):
    experiment_dir = prepare_dirs(
        config.experiment_name,
        config.level_name
    )
    if config.use_wandb:
        import wandb
        print(
            f'[*] Config file is stored in ./{wandb.run.dir}/config.yaml')
    else:
        param_path = os.path.join(experiment_dir, 'config_params.yaml')
        with open(param_path, 'w') as fp:
            yaml.dump(config.__dict__, fp)
        print(f'[*] Saved config file at ./{param_path}')


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


def split_loader(data_loader, cut_off=5000):
    dataset = data_loader.dataset
    first_split = len(dataset) - cut_off
    datasets = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[first_split, cut_off]
    )
    return datasets


def load_teachers(config, devices, input_size):
    level = config.experiment_level
    if level < 1:
        raise ValueError('experiment_level must be an integer of 1 or higher')
    elif level == 1:
        return []
    elif level > 1:
        if config.hp_search_from_static:
            pre_level_name = f'level_{config.experiment_level - 1}'
        else:
            pre_level_name = config.level_name[:6] \
                + str(int(config.level_name[6]) - 1) \
                + config.level_name[7:]
        experiment_name = config.experiment_name.replace(' ', '_')
        prev_dir_name = f'experiments/{experiment_name}/{pre_level_name}/ckpt/'
        print(prev_dir_name)
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
            print(f'Loading {f_name} as teacher {i + 1}...')
            try:
                state_dict = torch.load(prev_dir_name + f_name)
            except RuntimeError:
                state_dict = torch.load(
                    prev_dir_name + f_name,
                    map_location=torch.device('cpu'))
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


def _get_transforms(train, img_size, padding, padding_mode):
    if train:
        trans = transforms.Compose([
            transforms.RandomCrop(
                size=img_size,
                padding=padding,
                padding_mode=padding_mode),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                 0.229, 0.224, 0.225])
        ])
    return trans


def get_dataset(name, data_dir, fold, padding=4, padding_mode='reflect'):
    name = name.lower()
    assert name in {'cifar10', 'cifar100', 'tiny-imagenet-200'}, \
        "Only 'cifar10', 'cifar100', 'tiny-imagenet-200' are valid names fot he supported datasets"

    assert fold in {'train', 'val', 'test'}
    train = fold == 'train'

    dataset_properties = {
        'cifar10': {
            'img_size': 32,
            'num_classes': 10
        },
        'cifar100': {
            'img_size': 32,
            'num_classes': 100
        },
        'tiny-imagenet-200': {
            'img_size': 64,
            'num_classes': 200
        }

    }
    trans = _get_transforms(
        train=train,
        img_size=dataset_properties[name]['img_size'],
        padding=padding,
        padding_mode=padding_mode
    )

    if not os.path.exists(data_dir):
        out = input(
            f'Could not find the data directory {data_dir}, want to create it?\n(yes/no): ')
        if out.lower() == 'yes':
            os.makedirs(data_dir)
        else:
            sys.exit('Aborting script...')

    if name == 'cifar10':
        try:
            dataset = datasets.CIFAR10(
                root=data_dir,
                transform=trans,
                download=False,
                train=train,
            )

        except RuntimeError:
            out = input(
                'Could not find the dataset, want to downloading it?\n(yes/no')
            if out.lower() == 'yes':
                dataset = datasets.CIFAR10(
                    root=data_dir,
                    transform=trans,
                    download=True,
                    train=train
                )

    elif name == 'cifar100':
        try:
            dataset = datasets.CIFAR100(
                root=data_dir,
                transform=trans,
                download=False,
                train=train
            )

        except RuntimeError:
            out = input(
                'Could not find the dataset, want to downloading it?\n(yes/no')
            if out.lower() == 'yes':
                dataset = datasets.CIFAR100(
                    root=data_dir,
                    transform=trans,
                    download=True,
                    train=train
                )

    elif name == 'tiny-imagenet-200':
        try:
            dataset = datasets.ImageFolder(
                data_dir + f'/{fold}', transform=trans)
        except FileNotFoundError:
            out = input(
                f'Dataset not found at {data_dir}, want to download it?\n(yes/no): ')
            if out.lower() == 'yes':
                import tiny_imagenet_download
                tiny_imagenet_download.run()
            else:
                sys.exit(
                    'Aborting script, provide correct data_dir argument or download the dataset next time to proceed using tiny-imagenet-200.')

    else:
        print("Only 'cifar10', 'cifar100', 'tiny-imagenet-200' are valid names fot he supported datasets")
        sys.exit('Aborting script...')

    data_dict = {
        'dataset': dataset,
        **dataset_properties[name]
    }

    return data_dict


if __name__ == '__main__':
    # d = get_dataset('cifar100', './data/cifar100/', 'train', 4, 'reflect')
    # print(next(iter(d['data_loader']))[0].shape)
    a = MovingAverageMeter()
    b = MovingAverageMeter()
    a.update(1)
    b.update(2)
    print('\n', 'answer:', a + b)
    print('\n', 'answer:', a <= b)
