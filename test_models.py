import os
import yaml
import json
import pickle
import argparse
import torch
import pandas as pd
from typing import Set
from trainer import Trainer
from data_loader import get_test_loader, get_train_loader
from utils import get_dataset


def get_experiment_config(experiment_name: str, experiment_level: int):
    """Summary

    Args:
        experiment_name (str): Name of the experiment (must be an experiment of which best and last checkpoints are stored at ./experiments/{experiment_name}/level_i/ckpt/)
        experiment_level (int): Level in the experiment (must be stored at ./experiments/{experiment_name}/)

    Returns:
        dic: Description
    """
    experiment_path = f'experiments/{experiment_name}/level_{experiment_level}/wandb/'
    last_run = list(sorted(os.listdir(experiment_path)))[-1]
    config_path = os.path.join(experiment_path, last_run, 'config.yaml')
    with open(config_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def config_dict_to_namespace(config_dict: dict):
    for key, val in config_dict.items():
        try:
            config_dict[key] = val['value']
        except (KeyError, TypeError):
            pass
    return argparse.Namespace(**config_dict)


def get_experiment_dataloaders(experiment_name: str,
                               data_dir: str = './data/'):

    first_level = get_num_levels(experiment_name)[0]
    config = get_experiment_config(experiment_name, first_level)
    config = config_dict_to_namespace(config)
    config.data_dir = os.path.join(data_dir, config.dataset)

    torch.manual_seed(config.random_seed)
    kwargs = {}
    if not config.disable_cuda and torch.cuda.is_available():
        use_gpu = True
        torch.cuda.manual_seed_all(config.random_seed)
        kwargs = {'num_workers': config.num_workers,
                  'pin_memory': config.pin_memory}
    else:
        use_gpu = False

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
        teachers = []
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

    return train_loader, valid_loader, test_loader


def test_experiment_models(experiment_name: str,
                           experiment_level: int,
                           train_loader: torch.utils.data.dataloader.DataLoader,
                           valid_loader: torch.utils.data.dataloader.DataLoader,
                           test_loader: torch.utils.data.dataloader.DataLoader,
                           best: bool = True,
                           experiment_dir: str = './experiments'):
    """Summary

    Args:
        experiment_name (str): Description
        experiment_level (int): Description
        train_loader (torch.utils.data.dataloader.DataLoader): Description
        valid_loader (torch.utils.data.dataloader.DataLoader): Description
        test_loader (torch.utils.data.dataloader.DataLoader): Description
        best (bool, optional): Description
        experiment_dir (str, optional): Description

    Returns:
        defaultdict(dict): Nested dictionary with the test results
    """
    config = get_experiment_config(experiment_name, experiment_level)
    config = config_dict_to_namespace(config)
    config.use_wandb = False

    # instantiate trainer
    trainer = Trainer(config, train_loader, valid_loader, test_loader)
    test_result = trainer.test(config, best=best, return_results=True)

    return test_result


def get_num_levels(experiment_name: str, experiment_dir: str = './experiments'):
    """Summary

    Args:
        experiment_name (str): Name of the experiment (must be an experiment of which best and last checkpoints are stored at ./experiments/{experiment_name}/level_i/ckpt/)
        experiment_dir (str, optional): path to the directory where the experiments are stored
            default: ./experiments

    Returns:
        List[int]: list with levels stored at ./{experiment_dir}/{experiment_name}/
    """
    experiment_path = os.listdir(os.path.join(experiment_dir, experiment_name))
    return sorted([int(d.split('_')[-1]) for d in experiment_path])


def transform_results_to_rows(result_dict: dict, sideways: bool = False):
    """Summary

    Args:
        result_dict (dict): dict with the restresults, typically the output of test_experiment_models

    Returns:
        dic: nested dictionary with the test results, ready to be put into a pandas.DataFrame
    """
    # exp_dic = {}
    # for level, models in result_dict.items():
    #     for model_name, results in models.items():
    #         architecture = model_name[:2]
    #         level_dic = {
    #             'size': model_name[2:]
    #         }
    #         level_dic.update(results)
    #         exp_dic[(level, architecture)] = level_dic
    # return exp_dic
    exp_dic = {}
    for level, models in result_dict.items():
        for model_name, results in models.items():
            architecture = model_name[:2]
            size = model_name[2:]
            if sideways:
                exp_dic[(f'{level} ({architecture})', size)] = results
            else:
                # get size in front to get te correct column order later
                results = {'size': model_name[2:], **results}
                exp_dic[(level, architecture)] = results
    return exp_dic


def rowsdict_to_df(rowsdict: dict):
    """Summary

    Args:
        rowsdict (dict): results of transform_results_to_rows

    Returns:
        pandas.DataFrame: table showing the test loss, acc @ 1 and acc @ 5 for each of the models (rowcount is # models * # levels, typically 3 * 3)
    """
    return pd.DataFrame.from_dict(rowsdict).T


def get_experiment_results(
    experiment_name: str,
    sideways: bool = False,
    best: bool = True,
    data_dir: str = './data',
    experiment_dir: str = './experiments',
    save_df: Set[str] = {'return'}
):
    """


    Args:
        experiment_name (str): Name of the experiment (must be an experiment of which best and last checkpoints are stored at ./experiments/{experiment_name}/level_i/ckpt/)
        best (bool, optional): Wether or not the checkpoints of the epoch with the highest validation accuracy are loaded to be tested
            default: True
        data_dir (str, optional): path to the directory where the data is stored
            default: ./data
        experiment_dir (str, optional): path to the directory where the experiments are stored
            default: ./experiments
        save_df (str, optional): name of the format of which the results must be stored (results are stored at ./{cwd}/experiment_results/)
            options:
                - return (returns a pandas.DataFrame for usage in interactive shells)
                - dataframe (pikcles a pandas.DataFrame)
                - csv
                - json
                - yaml
            default: return

    Returns (if save_df is set to return):
        pandas.DataFrame: table showing the test loss, acc @ 1 and acc @ 5 for each of the models (rowcount is # models * # levels, typically 3 * 3)
    """
    assert save_df.issubset({'return', 'dataframe', 'csv', 'json', 'yaml'})
    train_loader, valid_loader, test_loader = get_experiment_dataloaders(
        experiment_name,
        data_dir)
    levels = get_num_levels(experiment_name)
    results_dict = {}
    for experiment_level in levels:
        test_result = test_experiment_models(
            experiment_name=experiment_name,
            experiment_level=experiment_level,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            best=best,
            experiment_dir=experiment_dir
        )
        results_dict[f'level {experiment_level}'] = test_result

    state = 'best' if best else 'last'
    if 'return' not in save_df:
        result_path = os.path.join('experiment_results', experiment_name)
        os.makedirs(result_path, exist_ok=True)

    if 'dataframe' in save_df:
        rows = transform_results_to_rows(results_dict, sideways)
        result_df = rowsdict_to_df(rows)
        with open(f'{result_path}/{config.experiment_name}_{state}.pkl', 'wb') as f:
            pickle.dump(result_df, f)
    if 'csv' in save_df:
        rows = transform_results_to_rows(results_dict, sideways)
        result_df = rowsdict_to_df(rows)
        result_df.to_csv(f'{result_path}/{config.experiment_name}_{state}.csv')
    if 'json' in save_df:
        with open(f'{result_path}/{config.experiment_name}_{state}.json', 'wb') as f:
            json.dump(results_dict, f)
    if 'yaml' in save_df:
        with open(f'{result_path}/{config.experiment_name}_{state}.json', 'wb') as f:
            yaml.dump(results_dict, f)
    if 'return' in save_df:
        rows = transform_results_to_rows(results_dict, sideways)
        result_df = rowsdict_to_df(rows)
        return result_df


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in {'true', '1'}
    parser = argparse.ArgumentParser(
        description='Get the test results of an experiment')

    parser.add_argument('--experiment_name', type=str,
                        help='')

    parser.add_argument('--sideways', type=str2bool, default=False,
                        help='')

    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='')

    parser.add_argument('--experiment_dir', type=str, default='./experiemnts/',
                        help='')

    parser.add_argument('--save_df', type=str, default=['dataframe'],
                        nargs='+', help='',
                        choices=['return', 'dataframe', 'csv', 'json', 'yaml'])

    config, unparsed = parser.parse_known_args()
    experiment_name = config.experiment_name
    data_dir = config.data_dir
    experiment_dir = config.experiment_dir
    save_df = set(config.save_df)

    get_experiment_results(
        experiment_name=experiment_name,
        best=True,
        data_dir=data_dir,
        experiment_dir=experiment_dir,
        save_df=save_df
    )

    get_experiment_results(
        experiment_name=experiment_name,
        best=False,
        data_dir=data_dir,
        experiment_dir=experiment_dir,
        save_df=save_df
    )
