import os
import shutil
import warnings
from copy import deepcopy
from datetime import datetime
# from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import wandb

import utils
from utils import accuracy, RunningAverageMeter, MovingAverageMeter, get_dataset
from data_loader import get_test_loader

if utils.isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# get rid of the torch.nn.KLDivLoss(reduction='batchmean') warning
warnings.filterwarnings(
    action="ignore",
    message="reduction: 'mean' divides the total loss by both the batch size and the support size."
)


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the MobileNet Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, train_loader, valid_loader, test_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """

        # DATA PARAMS
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.test_script = config.test_script
        if self.test_script:
            try:
                if config.train:
                    self.train_loader = DataLoader(
                        self.train_loader.sampler.data_source.data[
                            :250 * config.batch_size],
                        batch_size=config.batch_size
                    )

                self.valid_loader = DataLoader(
                    self.valid_loader.sampler.data_source.data[
                        :3 * config.batch_size],
                    batch_size=config.batch_size
                )
            except AttributeError:
                if config.train:
                    self.train_loader = DataLoader(
                        self.train_loader.sampler.data_source.dataset.data[
                            :10 * config.batch_size],
                        batch_size=config.batch_size
                    )

                self.valid_loader = DataLoader(
                    self.valid_loader.sampler.data_source.dataset.data[
                        :3 * config.batch_size],
                    batch_size=config.batch_size
                )

            self.test_loader = self.valid_loader

        self.num_train = len(
            self.train_loader.dataset) if train_loader else 0
        self.num_valid = len(
            self.valid_loader.dataset) if valid_loader else 0
        self.num_test = len(self.test_loader.dataset) if test_loader else 0
        self.num_classes = config.num_classes
        _batch = next(iter(self.test_loader))
        self.input_size = utils.infer_input_size(_batch)

        # TRAINING PARAMS
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.nesterov = config.nesterov
        self.gamma = config.gamma
        self.lr_step = config.lr_step
        self.lambda_a = config.lambda_a
        self.lambda_b = config.lambda_b
        self.temperature = config.temperature
        self.scale_dml = config.scale_dml

        # MISCELLANEOUS PARAMS
        self.model_num = len(config.model_names)
        self.counter = 0
        self.use_wandb = config.use_wandb
        self.use_sweep = config.use_sweep
        self.progress_bar = config.progress_bar
        self.experiment_level = config.experiment_level
        self.experiment_name = config.experiment_name
        self.level_name = config.level_name
        self.unlabelled = bool(config.unlabel_split)
        self.discard_unlabelled = config.discard_unlabelled

        # MODELS AND MODEL ATTRIBUTES
        self.model_names = config.model_names
        self.indexed_model_names = [
            f'({i})_{model_name}' for i, model_name in enumerate(self.model_names, 1)
        ]
        self.use_gpu = (not config.disable_cuda and torch.cuda.is_available())
        self.devices = utils.get_devices(self.model_num, self.use_gpu)
        self.nets = utils.model_init_and_placement(
            self.model_names,
            self.devices,
            self.input_size,
            self.num_classes)

        # LOSS FUNCTIONS
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss(reduction='none')

        # KEEP TRACK OF THE VALIDATION ACCURACY
        self.best_valid_accs = [-0.1] * self.model_num
        self.best_mean_valid_acc = -0.1

        # LEARNING SIGNAL CONDITIONNS ADN FRACTIONS
        # if lambda b = 1. (which it always should be for the first level)
        # the kd part is disabled and should therefore not be logged
        self.kd_condition = all([
            bool(1 - self.lambda_b),
            self.experiment_level > 1
        ])
        self.kd_fraction = (1 - self.lambda_b) * self.lambda_a

        # deep mutual learning can only be done on a set of models
        # the dml is therefore disabled for model_num <= 1 (for a kd experiment for instance)
        self.dml_condition = any([
            self.model_num >= 2,
            bool(self.lambda_b),
            bool(self.lambda_a)
        ])
        self.dml_fraction = self.lambda_a * self.lambda_b

        # if both previous conditions are False, the sl_signal is equal to the overall loss,
        # therefore we do not need the sl_signal explicitely
        self.sl_condition = any([self.dml_condition, self.kd_condition])
        self.sl_fraction = 1 - self.lambda_a

        # CONFIGURE WEIGHTS & BIASES LOGGING AND SAVE DIR
        self.experiment_dir = utils.prepare_dirs(
            self.experiment_name,
            self.level_name
        )
        if self.use_wandb:
            wandb.init(
                name=self.level_name,
                project=self.experiment_name,
                dir=self.experiment_dir,
                config=config,
                id=self.level_name,
                tags=list(set(config.model_names))
            )
            wandb.log({
                'sl fraction': self.sl_fraction,
                'kd fraction': self.kd_fraction,
                'dml fraction': self.dml_fraction
            })
            if self.use_sweep:
                _config = wandb.config
                for param in config.hp_search:
                    print(param, getattr(_config, param))
                    setattr(self, param, getattr(_config, param))

        # INITIALIZE OPTIMIZER & SCHEDULER AND LOG THE MODEL DESCRIPTIONS
        model_stats = []
        self.optimizers = []
        self.schedulers = []
        for i, net in enumerate(self.nets):

            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
            )

            self.optimizers.append(optimizer)

            # set learning rate decay
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizers[i],
                step_size=self.lr_step,
                gamma=self.gamma,
                last_epoch=-1
            )

            self.schedulers.append(scheduler)

            model_name = self.model_names[i]
            architecture, size_indicator = utils.infer_model_desc(model_name)
            params = sum([p.data.nelement() for p in net.parameters()])
            print('[*] Number of parameters of {} model: {:,}'.format(
                model_name,
                params)
            )
            model_stats.append(
                [model_name,
                 architecture,
                 size_indicator,
                 f'{params:,}'.replace(',', '.')]
            )
        if self.use_wandb:
            wandb.log({"examples": wandb.Table(data=model_stats, columns=[
                "Model name", "Architecture", "Size indicator", "# params"])})

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        # if self.resume:
        #     self.load_checkpoint(best=False)
        print(
            "[*] Training started at {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print("[*] Train on {} samples, validate on {} samples, test on {} samples".format(
            self.num_train, self.num_valid, self.num_test)
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f} - Current time: {}'.format(
                    epoch + 1,
                    self.epochs,
                    self.optimizers[0].param_groups[0]['lr'],
                    datetime.now().strftime('%H:%M:%S')
                )
            )

            # train for 1 epoch
            train_metrics = self.one_iteration(train_mode=True)

            # evaluate on validation set
            with torch.no_grad():
                valid_metrics = self.one_iteration(train_mode=False)

            # print the epoch stats to the console
            utils.print_epoch_stats(
                model_names=self.indexed_model_names,
                train_losses=train_metrics['train loss'],
                train_accs=train_metrics['train acc @ 1'],
                valid_losses=valid_metrics['valid loss'],
                valid_accs=valid_metrics['valid acc @ 1']
            )

            mean_valid_acc = np.array(
                [acc.avg for acc in valid_metrics['valid acc @ 1']]).mean()
            if self.best_mean_valid_acc < mean_valid_acc:
                self.best_mean_valid_acc = mean_valid_acc
                if self.use_wandb:
                    wandb.run.summary["best_mean_valid_acc"] = mean_valid_acc

            # log all metrics to Weights and Biases
            if self.use_wandb:
                log_dict = {'Average validation accuracy': mean_valid_acc}
                for i, model_name in enumerate(self.indexed_model_names):
                    for metric_dict in [train_metrics, valid_metrics]:
                        for metric_name, avg_meter in metric_dict.items():
                            log_dict[f'{model_name} {metric_name}'] = avg_meter[i].avg
                wandb.log(log_dict)

            # save epoch state
            for i, net in enumerate(self.nets):
                is_best = False
                if self.best_mean_valid_acc < valid_metrics['valid acc @ 1'][i].avg:
                    is_best = True
                    self.best_valid_accs[i] = valid_metrics['valid acc @ 1'][i].avg

                self.save_checkpoint(
                    self.indexed_model_names[i],
                    {'epoch': epoch + 1,
                     'model_state': net.state_dict(),
                     'optim_state': self.optimizers[i].state_dict(),
                     'current_valid_acc': valid_metrics['valid acc @ 1'][i].avg,
                     'best_valid_acc': self.best_valid_accs[i]},
                    is_best,
                    self.use_wandb
                )

            for scheduler in self.schedulers:
                scheduler.step()

        # if self.test_script:
        #     shutil.rmtree('/'.join(self.experiment_dir.split('/')[:-1]))

    def one_iteration(self, train_mode):
        if train_mode:
            AverageMeter = MovingAverageMeter
            data_loader = self.train_loader
        else:
            AverageMeter = RunningAverageMeter
            data_loader = self.valid_loader

        losses = []
        accs_at_1 = []
        accs_at_5 = []
        if self.sl_condition:
            sl_signals = []
        if self.kd_condition:
            kd_signals = []
        if self.dml_condition:
            dml_signals = []

        for net in self.nets:
            if train_mode:
                net.train()
            else:
                net.eval()
            losses.append(AverageMeter())
            accs_at_1.append(AverageMeter())
            accs_at_5.append(AverageMeter())
            if self.sl_condition:
                sl_signals.append(AverageMeter())
            if self.kd_condition:
                kd_signals.append(AverageMeter())
            if self.dml_condition:
                dml_signals.append(AverageMeter())

        if train_mode and self.progress_bar:
            pbar = tqdm(total=self.num_train)

        for batch_i, batch in enumerate(data_loader):

            # unpack batch
            _images, _true_labels, _psuedo_labels, _unlabelled = batch
            if self.discard_unlabelled:
                if sum(_unlabelled) == 0:
                    if train_mode and self.progress_bar:
                        pbar.update(len(_unlabelled))
                    continue

                _unlabelled = _unlabelled == 1
                _images = _images[_unlabelled]
                _true_labels = _true_labels[_unlabelled]
                _psuedo_labels = _psuedo_labels[_unlabelled]

            # place the data to the possibly multiple devices
            images, true_labels, psuedo_labels, unlabelled_mask = [], [], [], []
            for i, device in zip(range(self.model_num), self.devices):
                images.append(
                    Variable(_images.clone().to(device)))
                true_labels.append(
                    Variable(_true_labels.clone().to(device)))
                psuedo_labels.append(
                    _psuedo_labels[:, :, i].to(device))
                if self.unlabelled:
                    unlabelled_mask.append(
                        _unlabelled.to(device))

            # forward pass
            outputs = []
            for i, net in enumerate(self.nets):
                outputs.append(net(images[i]))

            # CALCULATE AGGREGATED LOSSES AND UPDATE PARAMETERS
            self.scale_dml = False
            if self.scale_dml:
                sl_signal_dml = []
                with torch.no_grad():
                    for i in range(self.model_num):
                        sl_signal = self.loss_ce(outputs[i], true_labels[i])
                        if self.unlabelled and not self.discard_unlabelled:
                            sl_signal *= unlabelled_mask[i]
                            sl_signal = sl_signal[sl_signal != 0].mean()
                        else:
                            sl_signal = sl_signal.mean()
                        sl_signal_dml.append(sl_signal)
                cum_sl_signals = sum(sl_signal_dml) * len(sl_signal_dml)

            for i in range(self.model_num):
                # supervised learning signal
                sl_signal = self.loss_ce(outputs[i], true_labels[i])
                if self.unlabelled and not self.discard_unlabelled:
                    sl_signal *= unlabelled_mask[i]
                    sl_signal = sl_signal[sl_signal != 0].mean()
                else:
                    sl_signal = sl_signal.mean()

                # update the signal for logging
                if self.sl_condition:
                    sl_signals[i].update(sl_signal)

                # initialize the overall loss
                sl_part = self.sl_fraction * sl_signal
                loss = sl_part

                # knowledge distillation signal
                if self.kd_condition:
                    p_i = F.log_softmax(outputs[i] / self.temperature, dim=1)
                    p_j = F.softmax(psuedo_labels[i] / self.temperature, dim=1)
                    kd_signal = self.loss_kl(p_i, p_j)
                    kd_signal *= self.temperature ** 2

                    # update the signal for logging
                    kd_signals[i].update(kd_signal)

                    # add kd signal to the overall loss
                    kd_part = self.kd_fraction * kd_signal
                    loss += kd_part

                # deep mutual learning signal
                if self.dml_condition:
                    p_i = F.log_softmax(outputs[i], dim=1)
                    dml_signal = 0
                    for j in range(self.model_num):
                        if i != j:
                            p_j = F.softmax(Variable(outputs[j]), dim=1)
                            if len(self.devices) > 1:
                                p_j = p_j.clone().to(self.devices[i])
                            dml_signal += self.loss_kl(p_i, p_j)
                            if self.scale_dml:
                                dml_signal /= sl_signal_dml[i] * cum_sl_signals

                    dml_signal /= max(1, self.model_num - 1)

                    # update the signal for logging
                    dml_signals[i].update(dml_signal)

                    # add dml signal to the overall loss
                    dml_part = self.dml_fraction * dml_signal
                    loss += dml_part

                if self.discard_unlabelled:
                    loss /= torch.mean(_unlabelled.float())

                # COMPUTE GRADIENTS AND UPDATE SGD
                if train_mode:
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                # MEASURE ACCURACY AND RECORD LOSS
                preds = outputs[i].clone().to('cpu').data
                prec_at_1 = accuracy(
                    preds,
                    _true_labels.data,
                    topk=(1,))[0]
                prec_at_5 = accuracy(
                    preds,
                    _true_labels.data,
                    topk=(5,))[0]

                accs_at_1[i].update(prec_at_1.item())
                accs_at_5[i].update(prec_at_5.item())
                losses[i].update(loss.item())

            # update progressbar
            if train_mode and self.progress_bar:
                pbar.set_description(
                    "Average loss: {:.3f}, average acc: {:.3f}".format(
                        sum([loss for loss in losses]) / len(losses),
                        sum([acc for acc in accs_at_1]) / len(accs_at_1)
                    )
                )
                pbar.update(len(_unlabelled))

        mode = 'train' if train_mode else 'valid'
        metrics = {
            f'{mode} loss': losses,
            f'{mode} acc @ 1': accs_at_1,
            f'{mode} acc @ 5': accs_at_5
        }
        if self.sl_condition:
            metrics[f'{mode} sl signal'] = sl_signals

        if self.kd_condition:
            metrics[f'{mode} kd signal'] = kd_signals

        if self.dml_condition:
            metrics[f'{mode} dml signal'] = dml_signals

        return metrics

    def test(self, config, best=False, return_results=True):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        losses = RunningAverageMeter()
        top1 = RunningAverageMeter()
        top5 = RunningAverageMeter()

        keep_track_of_results = return_results or self.use_wandb

        if best:
            self.load_checkpoints(best=True, inplace=True, verbose=False)

        if not hasattr(self, 'test_loader'):
            kwargs = {}
            if not config.disable_cuda and torch.cuda.is_available():
                kwargs = {'num_workers': 4,
                          'pin_memory': True}
            data_dict = get_dataset(config.dataset, config.data_dir, 'test')
            kwargs.update(data_dict)
            self.test_loader = get_test_loader(
                batch_size=config.batch_size,
                **kwargs)

        if keep_track_of_results:
            results = {}
            all_accs = []

        for net, model_name in zip(self.nets, self.model_names):
            net.eval()

            if self.progress_bar:
                pbar = tqdm(
                    total=len(self.test_loader.dataset),
                    leave=False,
                    desc=f'Testing {model_name}'
                )

                for i, (images, labels, _, _) in enumerate(self.test_loader):
                    if self.use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)

                    # forward pass
                    with torch.no_grad():
                        outputs = net(images)
                    loss = self.loss_ce(outputs, labels).mean()

                    # measure accuracy and record loss
                    prec_at_1, prec_at_5 = accuracy(
                        outputs.data,
                        labels.data,
                        topk=(1, 5)
                    )
                    losses.update(loss.item(), images.size()[0])
                    top1.update(prec_at_1.item(), images.size()[0])
                    top5.update(prec_at_5.item(), images.size()[0])

                    if self.progress_bar:
                        pbar.update(self.test_loader.batch_size)
                if self.progress_bar:
                    pbar.write(
                        '[*] {:5}: Test loss: {:.3f}, top1_acc: {:.3f}%, top5_acc: {:.3f}%'
                        .format(model_name, losses.avg, top1.avg, top5.avg)
                    )
                    pbar.close()

            fold = 'best' if best else 'last'

            if self.use_wandb:
                wandb.run.summary[f"{fold} test acc {model_name}"] = top1.avg

            if keep_track_of_results:
                results[f'{model_name} test loss'] = losses.avg
                results[f'{model_name} test acc @ 1'] = top1.avg
                results[f'{model_name} test acc @ 5'] = top5.avg
                all_accs.append(top1.avg)

        if keep_track_of_results:
            results['average test acc'] = sum(all_accs) / len(all_accs)
            results['min test acc'] = min(all_accs)
            results['max test acc'] = max(all_accs)

        if best:
            self.load_checkpoints(best=False, inplace=True, verbose=False)

        if self.use_wandb:
            wandb.log(results)

        if return_results:
            return results

    def save_checkpoint(self, model_name, state, is_best, use_wandb):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(
            self.experiment_dir,
            'ckpt',
            filename
        )
        torch.save(state, ckpt_path)
        if use_wandb:  # currently getting symlink errors (on Colab)
            wandb.run.summary[f"last valid acc {model_name}"] = state["current_valid_acc"]
            try:
                wandb.save(ckpt_path)
            except OSError:
                pass

        if is_best:
            path = os.path.join(
                self.experiment_dir,
                'ckpt',
                model_name + '_best.pth.tar'
            )
            shutil.copyfile(ckpt_path, path)
            if use_wandb:
                wandb.run.summary[f"best valid acc {model_name}"] = state["best_valid_acc"]
                try:
                    wandb.save(path)
                except OSError:
                    pass

    def load_checkpoints(self, best=False, inplace=False, verbose=False):
        """
        Load the best copy of a model. This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.
        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        ckpt_dir_path = os.path.join(self.experiment_dir, 'ckpt')
        if verbose:
            print(f"[*] Loading models from ./{ckpt_dir_path}")
        checkpoint_extension = '_best.pth.tar' if best else '_ckpt.pth.tar'
        ckpt_paths = [os.path.join(ckpt_dir_path, model_name + checkpoint_extension)
                      for model_name in self.indexed_model_names]

        if inplace:
            nets = self.nets
            optimizers = self.optimizers
        else:
            nets = [deepcopy(net) for net in self.nets]
            optimizers = [deepcopy(optimizer) for optimizer in self.optimizers]

        ckpts_dict = {}
        for ckpt_path, model, optimizer, model_name, device in zip(ckpt_paths, nets, optimizers, self.indexed_model_names, self.devices):
            ckpt = torch.load(ckpt_path, map_location=device)

            # load variables from checkpoint
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optim_state'])
            ckpt['model'] = model
            ckpt['optimizer'] = optimizer
            ckpts_dict[model_name] = ckpt

            if verbose:
                if best:
                    print(
                        "[*] Loaded {} checkpoint @ epoch {} "
                        "with best valid acc of {:.3f}".format(
                            ckpt_path, ckpt['epoch'], ckpt['best_valid_acc'])
                    )
                else:
                    print(
                        "[*] Loaded {} checkpoint @ epoch {}".format(
                            ckpt_path, ckpt['epoch'])
                    )

        if inplace:
            for i, model_name in enumerate(self.indexed_model_names):
                self.nets[i] = ckpts_dict[model_name]['model']
                self.optimizers[i] = ckpts_dict[model_name]['optimizer']
                self.best_valid_accs[i] = ckpts_dict[model_name]['best_valid_acc']
                self.start_epoch = ckpts_dict[model_name]['epoch']
