import os
import yaml
import torch
import itertools
import numpy as np
from collections import Counter, defaultdict
from model_factories import (
    efficientnet_factory,
    mobilenetv2_factory,
    resnet_factory,
    plain_cnn_factory)
from PIL import Image


def denormalize(T, coords):
    return (0.5 * ((coords + 1.0) * T))


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


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype='float32')
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype='float32')
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')


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


def correct_out_features(model, out_features):
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


def infer_input_size(batch):
    return batch[0].shape[2]


def uniquify(model_names):
    names = []
    count_state = defaultdict(lambda: 1)
    counter = Counter(model_names)
    for model_name in model_names:
        if counter[model_name] == 1:
            names.append(model_name)
        else:
            names.append(model_name + f'({count_state[model_name]})')
            count_state[model_name] += 1
    return names


def infer_model_desc(model_name):
    name_dict = {'EF': 'EfficientNet',
                 'MN': 'MobileNetV2',
                 'RN': 'ResNet',
                 'CN': 'VGG like CNN'
                 }
    mn_size_dict = {
        str(size): f'scale={int(size)/100}' for size in range(10, 255, 5)}

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
            prev_dir_name), "Can only carry out levels that are the level 1 or of which all previous levels have been run."
        ckpts = os.listdir(prev_dir_name)
        bests = list(sorted([path for path in ckpts if 'best' in path]))
        teachers = []
        model_names = [f_name.split('_')[1] for f_name in bests]
        models = model_init_and_placement(
            model_names,
            devices,
            input_size,
            config.num_classes
        )
        for i, f_name in enumerate(bests):
            state_dict = torch.load(prev_dir_name + f_name)
            model = models[i]
            model.load_state_dict(state_dict['model_state'])
            teachers.append(model)
    return teachers


def model_init_and_placement(model_names, devices, input_size, num_classes):
    model_architectures = [model_name[:2] for model_name in model_names]
    size_indicators = [model_name[2:] for model_name in model_names]
    for model_architecture in model_architectures:
        assert model_architecture in {'EF', 'MN', 'RN', 'CN'}, \
            "Model architecture abbreviation must be in [EF, MN, RN, CN]"
    nets = []
    for model_architecture, size_indicator, device in zip(model_architectures, size_indicators, itertools.cycle(devices)):
        if model_architecture == 'EF':
            net = efficientnet_factory.create_model(size_indicator)
            correct_out_features(net, num_classes)
        elif model_architecture == 'MN':
            net = mobilenetv2_factory.create_model(size=size_indicator,
                                                   input_size=input_size,
                                                   num_classes=num_classes)
        elif model_architecture == 'RN':
            net = resnet_factory.create_model(size_indicator)
            correct_out_features(net, num_classes)
        elif model_architecture == 'CN':
            net = plain_cnn_factory.create_model(
                size_indicator, input_size, num_classes)

        net.to(device)
        nets.append(net)
    return nets
