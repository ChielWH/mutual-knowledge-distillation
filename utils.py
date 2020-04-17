import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model_factories import efficientnet_factory, mobilenetv2_factory, resnet_factory

from PIL import Image


def denormalize(T, coords):
    return (0.5 * ((coords + 1.0) * T))


class AverageMeter(object):
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


def prepare_dirs(config):
    for path in [config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = config.save_name
    filename = model_name + '_params.json'
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def correct_out_features(model, out_features):
    import torch
    for idx, module in reversed(list(enumerate(model.modules()))):
        if type(module) == torch.nn.modules.linear.Linear:
            in_features = module.in_features
            bias = type(module.bias) == torch.nn.parameter.Parameter
            setattr(model, list(model.named_modules())[idx][0], torch.nn.Linear(
                in_features, out_features, bias=bias))


def infer_input_size(batch):
    return batch[0].shape[2]


def model_init(model_name, use_gpu, input_size, num_classes):
    model_architecture = model_name[:2]
    assert model_architecture in {'EF', 'MN', 'RN'}
    size_indicator = model_name[2:]
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
    if use_gpu:
        net.cuda()
    return net


if __name__ == '__main__':
    model_init('EFB6')
    model_init('RN302')
    model_init('MN35')
