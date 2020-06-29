import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import loss_landscapes
import loss_landscapes.metrics
from utils import get_dataset, model_init_and_placement


def plot_loss_landscape(model, steps, distance, normalization, save_path):
    loss_data = loss_landscapes.random_plane(
        model=model,
        metric=metric,
        distance=distance,
        steps=steps,
        normalization=normalization,
    )

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    fig.savefig(save_path)
    print(f'Saved figure at {save_path}')


matplotlib.rcParams['figure.figsize'] = [18, 12]
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Loss landscape visualization')

    def larger_than_one(x):
        x = int(x)
        if x < 1:
            raise argparse.ArgumentTypeError(
                f"Minimum number of plots is 1, {x} was provided")
        return x

    parser.add_argument('--model_path', '-p', type=str,
                        help='path to the models state dict')
    parser.add_argument('--save_path', '-sp', type=str,
                        help='path to the models state dict')
    parser.add_argument('--normalization', '-n', type=str, default='filter',
                        help='type of normalization',
                        choices=['model', 'layer', 'filter'])
    parser.add_argument('--steps', '-s', type=int, default=75,
                        help='number of steps in the trajectory, more steps is larger loss surface but takes longer to compute')
    parser.add_argument('--distance', '-ss', type=int, default=10,
                        help='size of the steps in the trajectory, larger stepsize is larger loss surface but less specific')
    parser.add_argument('--n_plots', '-np', type=larger_than_one, default=1,
                        help='number of times to plot it, different initializations result in different surfaces')
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        help='name of the dataset',
                        choices=['cifar-10', 'cifar-100', 'tiny-imagenet-200'])
    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help='path to the models state dict')
    parser.add_argument('--loss_function', '-lf', type=str, default='CrossEntropyLoss',
                        help='loss function that is part of the torch.nn.modules.loss module')
    parser.add_argument('--data_dir', '-dd', type=str, default='./data/',
                        help='path to the datasets')

    config, unparsed = parser.parse_known_args()

    # set criterion
    criterion = getattr(torch.nn.modules.loss, config.loss_function)()

    # get the data
    data_dict = get_dataset(
        name=config.dataset,
        data_dir=os.path.join(config.data_dir, config.dataset),
        fold='test'
    )

    data_loader = iter(torch.utils.data.DataLoader(
        dataset=data_dict['dataset'],
        batch_size=config.batch_size
    ))

    # initialize model
    model_name = config.model_path.split('/')[-1].split('_')[1]
    model = model_init_and_placement(
        model_names=[model_name],
        devices=['cpu'],
        input_size=data_dict['img_size'],
        num_classes=data_dict['num_classes']
    )[0]

    # load the weights
    state_dict = torch.load(
        f=config.model_path,
        map_location=torch.device('cpu')
    )
    model.load_state_dict(state_dict['model_state'])

    # create the plot(s)
    if config.n_plots == 1:
        x, y = next(data_loader)
        metric = loss_landscapes.metrics.Loss(criterion, x, y)
        plot_loss_landscape(
            model=model,
            steps=config.steps,
            distance=config.distance,
            normalization=config.normalization,
            save_path=config.save_path
        )
    else:
        save_path = '.'.join(config.save_path.split('.')[:-1])
        save_extension = config.save_path.split('.')[-1]
        for i in range(config.n_plots):
            x, y = next(data_loader)
            metric = loss_landscapes.metrics.Loss(criterion, x, y)
            plot_loss_landscape(
                model=model,
                steps=config.steps,
                distance=config.distance,
                normalization=config.normalization,
                save_path=save_path + f'_{i+1}.' + save_extension
            )
