import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import shutil
import warnings
import wandb
import utils
from datetime import datetime
from utils import accuracy, RunningAverageMeter, MovingAverageMeter

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

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # DATA PARAMS
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
            batch = next(iter(self.train_loader))
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
            batch = next(iter(self.test_loader))
        self.input_size = utils.infer_input_size(batch)
        self.num_classes = config.num_classes

        # TRAINING PARAMS
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.nesterov = config.nesterov
        self.gamma = config.gamma
        self.lambda_a = config.lambda_a
        self.lambda_b = config.lambda_b
        self.temperature = config.temperature

        # MISCELLANEOUS PARAMS
        self.model_num = len(config.model_names)
        if not config.disable_cuda and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.devices = utils.get_devices(self.model_num, self.use_gpu)
        self.best = config.best
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.use_wandb = config.use_wandb
        self.experiment_name = config.experiment_name.lower().replace(' ', '_')
        self.experiment_level = config.experiment_level
        self.resume = config.resume
        self.progress_bar = config.progress_bar

        # MODEL SPECIFIC PARAMS
        self.model_names = config.model_names
        self.nets = utils.model_init_and_placement(
            self.model_names,
            self.devices,
            self.input_size,
            self.num_classes)
        self.indexed_model_names = [
            f'({i})_{model_name}' for i, model_name in enumerate(self.model_names, 1)]

        # LIST AND FUNC INITIALIZATIONS
        self.optimizers = []
        self.schedulers = []
        self.best_valid_accs = [0.] * self.model_num
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()

        # LEARNING SIGNAL CONDITIONNS
        # if lambda b = 1. (which it always should be for the first level)
        # the kd part is disabled and should therefore not be logged
        self.kd_condition = bool(1 - self.lambda_b)

        # deep mutual learning can only be done on a set of models
        # the dml is therefore disabled for model_num <= 1 (for a kd experiment for instance)
        self.dml_condition = any(
            [self.model_num >= 2, bool(self.lambda_b), bool(self.lambda_a)])

        # if both previous conditions are False, the sl_signal is equal to the overall loss,
        # therefore we do not need the sl_signal explicitely
        self.sl_condition = any([self.dml_condition, self.kd_condition])

        # CONFIGURE WEIGHTS & BIASES LOGGING AND SAVE DIR
        self.experiment_dir = utils.prepare_dirs(config)
        if self.use_wandb:
            wandb.init(name=f'Level {self.experiment_level}',
                       project=self.experiment_name,
                       dir=self.experiment_dir,
                       config=config,
                       id=str(self.experiment_level),
                       tags=list(set(config.model_names))
                       )

        # INITIALIZE OPTIMIZER & SCHEDULER AND LOG THE MODEL DESCRIPTIONS
        model_stats = []
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
                step_size=60,
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
            "[*] Trainloop started at {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print("[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
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

            # log all metrics to Weights and Biases
            if self.use_wandb:
                log_dict = {}
                for i, model_name in enumerate(self.indexed_model_names):
                    for metric_dict in [train_metrics, valid_metrics]:
                        for metric_name, avg_meter in metric_dict.items():
                            log_dict[f'{model_name} {metric_name}'] = avg_meter[i].avg
                wandb.log(log_dict)

            # save epoch state
            for i, net in enumerate(self.nets):
                is_best = False
                if self.best_valid_accs[i] < valid_metrics['valid acc @ 1'][i].avg:
                    is_best = True
                    self.best_valid_accs[i] = valid_metrics['valid acc @ 1'][i].avg

                self.save_checkpoint(
                    self.indexed_model_names[i],
                    {'epoch': epoch + 1,
                     'model_state': net.state_dict(),
                     'optim_state': self.optimizers[i].state_dict(),
                     'current_valid_acc': valid_metrics['valid acc @ 1'],
                     'best_valid_acc': self.best_valid_accs[i]},
                    is_best,
                    self.use_wandb
                )

            for scheduler in self.schedulers:
                scheduler.step()

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
            _images, _true_labels, _psuedo_labels = batch

            # place the data to the possibly multiple devices
            images, true_labels, psuedo_labels = [], [], []
            for i, device in zip(range(self.model_num), self.devices):
                images.append(
                    Variable(_images.clone().to(device)))
                true_labels.append(
                    Variable(_true_labels.clone().to(device)))
                psuedo_labels.append(
                    _psuedo_labels[:, :, i].to(device))

            # forward pass
            outputs = []
            for i, net in enumerate(self.nets):
                # if not train_mode:
                #     with torch.no_grad():
                #         outputs.append(net(images[i]))
                # else:
                #     outputs.append(net(images[i]))
                outputs.append(net(images[i]))

            # CALCULATE AGGREGATED LOSSES AND UPDATE PARAMETERS
            for i in range(self.model_num):
                # supervised learning signal
                sl_signal = self.loss_ce(outputs[i], true_labels[i])

                # update the signal for logging
                if self.sl_condition:
                    sl_signals[i].update(sl_signal)

                # initialize the overall loss
                sl_part = (1 - self.lambda_a) * sl_signal
                loss = sl_part

                # knowledge distillation signal
                if self.kd_condition:
                    kd_signal = nn.KLDivLoss()(
                        F.log_softmax(
                            outputs[i] / self.temperature,
                            dim=1),
                        F.softmax(
                            psuedo_labels[i] / self.temperature,
                            dim=1)
                    )

                    # update the signal for logging
                    kd_signals[i].update(kd_signal)

                    # add kd signal to the overall loss
                    kd_part = (1 - self.lambda_b) * self.lambda_a * \
                        self.temperature * self.temperature * kd_signal
                    loss += kd_part

                # deep mutual learning signal
                if self.dml_condition:
                    p_i = outputs[i]
                    dml_signal = 0
                    for j in range(self.model_num):
                        if i != j:
                            p_j = outputs[j]
                            if len(self.devices) > 1:
                                p_j = p_j.clone().to(self.devices[i])
                            dml_signal += self.loss_kl(
                                F.log_softmax(p_i, dim=1),
                                F.softmax(Variable(p_j), dim=1)
                            )

                    # update the signal for logging
                    dml_signals[i].update(dml_signal)

                    # add dml signal to the overall loss
                    dml_part = self.lambda_b * \
                        (dml_signal / max(1, self.model_num - 1))
                    loss += dml_part

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
                        sum([loss.avg for loss in losses]) / len(losses),
                        sum([acc.avg for acc in accs_at_1]) / len(accs_at_1)
                    )
                )
                batch_size = _images.shape[0]
                pbar.update(batch_size)

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

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        losses = RunningAverageMeter()
        top1 = RunningAverageMeter()
        top5 = RunningAverageMeter()

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        self.model.eval()
        for i, (images, labels) in enumerate(self.test_loader):
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # forward pass
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            # measure accuracy and record loss
            prec_at_1, prec_at_5 = accuracy(
                outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.item(), images.size()[0])
            top1.update(prec_at_1.item(), images.size()[0])
            top5.update(prec_at_5.item(), images.size()[0])

        print(
            '[*] Test loss: {:.3f}, top1_acc: {:.3f}%, top5_acc: {:.3f}%'
            .format(
                losses.avg, top1.avg, top5.avg)
        )

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
            # wandb.run.summary["current valid acc"] = state["current_valid_acc"]
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
                # wandb.run.summary["best valid acc"] = state["best_valid_acc"]
                try:
                    wandb.save(path)

                except OSError:
                    pass
