import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import os
import time
import shutil
import warnings

from utils import accuracy, RunningAverageMeter, MovingAverageMeter, model_init, infer_input_size, isnotebook, print_epoch_stats
import wandb
# from tensorboard_logger import configure, log_value

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# get rid of the torch.nn.KLDivLoss(reduction='batchmean') warning
warnings.filterwarnings(
    "ignore", message="reduction: 'mean' divides the total loss by both the batch size and the support size.")


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

        # data params
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
        self.input_size = infer_input_size(batch)
        self.num_classes = config.num_classes

        # training params
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

        # misc params
        if not config.disable_cuda and torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_wandb = config.use_wandb
        self.resume = config.resume
        self.model_name = config.save_name

        # model specific params
        self.model_names = config.model_names
        self.nets = [model_init(model_name,
                                self.use_gpu,
                                self.input_size,
                                self.num_classes)
                     for model_name in self.model_names]
        self.model_num = len(self.model_names)

        # list and func initializations
        self.optimizers = []
        self.schedulers = []

        self.best_valid_accs = [0.] * self.model_num

        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()

        # configure tensorboard logging
        if self.use_wandb:
            wandb.init(name='test experiment',
                       project='mutual-knowledge-distillation')

        print(self.gamma)
        for i, net in enumerate(self.nets):
            # initialize optimizer and scheduler
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum,
                                  weight_decay=self.weight_decay, nesterov=self.nesterov)

            self.optimizers.append(optimizer)

            # set learning rate decay
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizers[i], step_size=60, gamma=self.gamma, last_epoch=-1)

            self.schedulers.append(scheduler)

            print('[*] Number of parameters of {} model: {:,}'.format(
                self.model_names[i],
                sum([p.data.nelement() for p in net.parameters()])))

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
            )

            # train for 1 epoch
            train_losses, train_accs_at_1, train_accs_at_5 = self.train_one_epoch(
                epoch)

            # evaluate on validation set
            valid_losses, valid_accs_at_1, valid_accs_at_5 = self.validate(
                epoch)

            print_epoch_stats(
                model_names=self.model_names,
                train_losses=train_losses,
                train_accs=train_accs_at_1,
                valid_losses=valid_losses,
                valid_accs=valid_accs_at_1
            )

            if self.use_wandb:
                log_dict = {}
                for i, model_name in enumerate(self.model_names):
                    log_dict[f'{model_name} train loss'] = train_losses[i].avg
                    log_dict[f'{model_name} train acc @ 1'] = train_accs_at_1[i].avg
                    log_dict[f'{model_name} train acc @ 5'] = train_accs_at_5[i].avg
                    log_dict[f'{model_name} valid loss'] = valid_losses[i].avg
                    log_dict[f'{model_name} valid acc @ 1'] = valid_accs_at_1[i].avg
                    log_dict[f'{model_name} valid acc @ 5'] = valid_accs_at_5[i].avg
                wandb.log(log_dict)

            # save epoch state
            for i, net in enumerate(self.nets):
                is_best = False
                if self.best_valid_accs[i] > valid_accs_at_1[i].avg:
                    is_best = True
                    self.best_valid_accs[i] = valid_accs_at_1[i].avg

                self.save_checkpoint(self.model_names[i],
                                     {'epoch': epoch + 1,
                                      'model_state': net.state_dict(),
                                      'optim_state': self.optimizers[i].state_dict(),
                                      'best_valid_acc': self.best_valid_accs[i],
                                      }, is_best
                                     )

            for scheduler in self.schedulers:
                scheduler.step()

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = RunningAverageMeter()
        losses = []
        accs_at_1 = []
        accs_at_5 = []

        for net in self.nets:
            net.train()
            losses.append(MovingAverageMeter())
            accs_at_1.append(MovingAverageMeter())
            accs_at_5.append(MovingAverageMeter())

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for batch_i, batch in enumerate(self.train_loader):
                images, true_labels, psuedo_labels = batch
                if self.use_gpu:
                    images = images.cuda()
                    true_labels = true_labels.cuda()
                    psuedo_labels = psuedo_labels.cuda()
                images, true_labels = Variable(images), Variable(true_labels)

                # forward pass
                outputs = []
                for net in self.nets:
                    outputs.append(net(images))
                for i in range(self.model_num):
                    sl_signal = self.loss_ce(outputs[i], true_labels)
                    kd_signal = nn.KLDivLoss()(F.log_softmax(outputs[i] / self.temperature, dim=1),
                                               F.softmax(psuedo_labels[:, :, i] / self.temperature, dim=1))
                    dml_signal = 0
                    for j in range(self.model_num):
                        if i != j:
                            dml_signal += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                       F.softmax(Variable(outputs[j]), dim=1))

                    sl_part = (1 - self.lambda_a) * sl_signal
                    kd_part = (1 - self.lambda_b) * self.lambda_a * \
                        self.temperature * self.temperature * kd_signal
                    dml_part = self.lambda_b * \
                        (dml_signal / max(1, self.model_num - 1))
                    loss = sl_part + kd_part + dml_part

                    # measure accuracy and record loss
                    prec_at_1 = accuracy(outputs[i].data,
                                         true_labels.data, topk=(1,))[0]
                    prec_at_5 = accuracy(outputs[i].data,
                                         true_labels.data, topk=(5,))[0]
                    losses[i].update(loss.item())
                    accs_at_1[i].update(prec_at_1.item())
                    accs_at_5[i].update(prec_at_5.item())

                    # compute gradients and update SGD
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - {} loss: {:.3f}, acc: {:.3f}".format(
                            (toc -
                             tic), self.model_names[0], losses[0].avg, accs_at_1[0].avg
                        )
                    )
                )

                self.batch_size = images.shape[0]
                pbar.update(self.batch_size)

            return losses, accs_at_1, accs_at_5

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = []
        accs_at_1 = []
        accs_at_5 = []
        for i, net in enumerate(self.nets):
            net.eval()
            losses.append(RunningAverageMeter())
            accs_at_1.append(MovingAverageMeter())
            accs_at_5.append(MovingAverageMeter())

        for i, batch in enumerate(self.valid_loader):
            images, true_labels, psuedo_labels = batch
            if self.use_gpu:
                images = images.cuda()
                true_labels = true_labels.cuda()
                psuedo_labels = psuedo_labels.cuda()
            images, true_labels = Variable(images), Variable(true_labels)

            # forward pass
            outputs = []
            for net in self.nets:
                outputs.append(net(images))
            for i in range(self.model_num):
                sl_signal = self.loss_ce(outputs[i], true_labels)
                kd_signal = nn.KLDivLoss()(F.log_softmax(outputs[i] / self.temperature, dim=1),
                                           F.softmax(psuedo_labels[:, :, i] / self.temperature, dim=1))
                dml_signal = 0
                for j in range(self.model_num):
                    if i != j:
                        dml_signal += self.loss_kl(F.log_softmax(outputs[i], dim=1),
                                                   F.softmax(Variable(outputs[j]), dim=1))

                sl_part = (1 - self.lambda_a) * sl_signal
                kd_part = (1 - self.lambda_b) * self.lambda_a * \
                    self.temperature * self.temperature * kd_signal
                dml_part = self.lambda_b * \
                    (dml_signal / max(1, self.model_num - 1))
                loss = sl_part + kd_part + dml_part

                # measure accuracy and record loss
                prec_at_1 = accuracy(outputs[i].data,
                                     true_labels.data, topk=(1,))[0]
                prec_at_5 = accuracy(outputs[i].data,
                                     true_labels.data, topk=(5,))[0]
                losses[i].update(loss.item())
                accs_at_1[i].update(prec_at_1.item())
                accs_at_5[i].update(prec_at_5.item())

        return losses, accs_at_1, accs_at_5

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
            '[*] Test loss: {:.3f}, top1_acc: {:.3f}%, top5_acc: {:.3f}%'.format(
                losses.avg, top1.avg, top5.avg)
        )

    def save_checkpoint(self, model_name, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = model_name + '_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )
