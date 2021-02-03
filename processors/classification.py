#! /usr/bin/env python
import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb

from utils import mean_ap, cmc, pairwise_distance, AverageMeter, accuracy
from utils import import_class, LR
# mgpu
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Classification():
    """ Processor for Skeleton-based Action Recognition """
    def __init__(self, arg):
        self.arg = arg
        self.work_dir = self.arg.work_dir
        self.writer = SummaryWriter(arg.work_dir)
        if arg.mgpu:
            dist_backend = 'nccl'
            torch.cuda.set_device(arg.local_rank)
            dist.init_process_group(backend=dist_backend)

        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.global_step = 0
        self.start_time = time.time()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        train_feeder = Feeder(**self.arg.train_feeder_args)
        test_feeder = Feeder(**self.arg.test_feeder_args)
        if self.arg.mgpu:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

            train_sampler = DistributedSampler(
                train_feeder,
                num_replicas=num_replicas,
                rank=rank)

            self.data_loader['train'] = torch.utils.data.DataLoader(
                train_feeder,
                batch_size=int(self.arg.batch_size/num_replicas),
                sampler=train_sampler,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)
        else:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_feeder,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=test_feeder,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)


    def load_model(self):
        Model = import_class(self.arg.model)
        model = Model(**self.arg.model_args)
        if self.arg.mgpu:
            self.model = DistributedDataParallel(model.cuda(),
                                                 device_ids=[self.arg.local_rank],
                                                 find_unused_parameters=True)
        else:
            self.model = nn.DataParallel(model).cuda()
        self.loss = nn.CrossEntropyLoss().cuda()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            raise ValueError()
        self.lr_scheduler = LR(self.optimizer, base_lr=self.arg.base_lr,
                               **self.arg.lr_args)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        if (not self.arg.mgpu) or self.arg.local_rank == 0:
            print(str)
            if self.arg.print_log:
                if not os.path.exists(self.arg.work_dir):
                    os.makedirs(self.arg.work_dir)
                with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                    print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch):
        self.global_step += 1
        self.model.train()

        losses = AverageMeter()
        accs = AverageMeter()
        process = tqdm(self.data_loader['train'])
        for batch_idx, (data, label, filename) in enumerate(process):
            data, label = data.float().cuda(), label.long().cuda()
            outputs = self.model(data)
            loss = self.loss(outputs, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            prec = accuracy(outputs, label, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            accs.update(prec.item(), data.size(0))
        self.writer.add_scalar('train/acc', accs.avg, self.global_step)
        self.writer.add_scalar('train/loss', losses.avg, self.global_step)
        self.print_log('train' +
                       ' - loss: {:.2f}'.format(losses.avg) +
                       ' - acc: {:.2%}'.format(accs.avg))

    def eval(self, epoch):
        if self.arg.top5:
            return self.eval_top5(epoch)
        else:
            return self.eval_top1(epoch)

    def eval_top5(self, epoch):
        self.model.eval()

        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets, _) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()
                outputs = self.model(inputs)

                loss = self.loss(outputs, targets)
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1_acc.update(prec1.item(), inputs.size(0))
                top5_acc.update(prec5.item(), inputs.size(0))
        self.writer.add_scalar('test/top1_acc', top1_acc.avg, self.global_step)
        self.writer.add_scalar('test/top5_acc', top5_acc.avg, self.global_step)
        self.writer.add_scalar('test/loss', losses.avg, self.global_step)

        str = ('test ' +
               ' - loss: {:.2f}'.format(losses.avg) +
               ' - top1-acc: {:.2%}'.format(top1_acc.avg) +
               ' - top5-acc: {:.2%}'.format(top5_acc.avg))
        is_best = top1_acc.avg > self.best_acc
        if is_best:
            self.best_acc = top1_acc.avg
            self.best_epoch = epoch + 1
            self.print_log(str + ' (*)')
        else:
            self.print_log(str)
        return is_best

    def eval_top1(self, epoch):
        self.model.eval()

        losses = AverageMeter()
        accs = AverageMeter()
        process = tqdm(self.data_loader['test'])
        for batch_idx, (inputs, targets, _) in enumerate(process):
            with torch.no_grad():
                inputs = inputs.float().cuda()
                targets = targets.long().cuda()
                outputs = self.model(inputs)

                loss = self.loss(outputs, targets)
                prec = accuracy(outputs, targets, topk=(1,))[0]
                losses.update(loss.item(), inputs.size(0))
                accs.update(prec.item(), inputs.size(0))
        self.writer.add_scalar('test/acc', accs.avg, self.global_step)
        self.writer.add_scalar('test/loss', losses.avg, self.global_step)

        str = ('test ' +
               ' - loss: {:.2f}'.format(losses.avg) +
               ' - acc: {:.2%}'.format(accs.avg))
        is_best = accs.avg > self.best_acc
        if is_best:
            self.best_acc = accs.avg
            self.best_epoch = epoch + 1
            self.print_log(str + ' (*)')
        else:
            self.print_log(str)
        return is_best


    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        self.model.module.load_state_dict(state)
        return self.arg.start_epoch

    def save_checkpoint(self, i, state, is_best, save_name='model'):
        if (not self.arg.mgpu) or (self.arg.local_rank == 0):
            ckpt_dir = os.path.join(self.work_dir, 'ckpt')
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            filename = str(i + 1)
            filename = save_name + '-' + filename if save_name else filename
            # ckpt_path = os.path.join(self.work_dir, 'ckpt.pth.tar')
            ckpt_path = os.path.join(ckpt_dir, filename + '-ckpt.pth.tar')
            torch.save(state, ckpt_path)
            if is_best:
                best_path = os.path.join(self.work_dir, save_name+'.pth.tar')
                torch.save(state, best_path)

    def start(self):
        if self.arg.phase == 'train':
            ''' Print out information log '''
            self.print_log('{} samples for training'.format(
                len(self.data_loader['train'].dataset)))
            self.print_log('{} samples for testing'.format(
                len(self.data_loader['test'].dataset)))
            if self.arg.print_model:
                self.print_log('Architecture:\n{}'.format(self.model))
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log('Parameters: {}'.format(num_params))
            if self.arg.weights:
                self.load_checkpoint(self.arg.weights, optim=False)
                self.arg.start_epoch = 0
            self.print_log('Configurations:\n{}\n'.format(str(vars(self.arg))))

            self.best_acc = 0
            self.best_epoch = -1
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                lr = self.lr_scheduler.step(epoch)
                self.print_log('\nEpoch: {}/{} - LR: {:6f}'.format(
                    epoch+1, self.arg.num_epoch, lr), print_time=False)
                self.train(epoch)
                is_best = self.eval(epoch)
                if ((epoch + 1) % self.arg.save_interval == 0) or  (epoch + 1 == self.arg.num_epoch) or is_best:
                    self.save_checkpoint(epoch,
                                         self.model.module.state_dict(),
                                         is_best)

            self.print_log('\nBest accuracy: {:.2%}, epoch: {}, dir: {}, time: {:.0f}'.format(
                self.best_acc, self.best_epoch, self.work_dir, time.time()-self.start_time))

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.load_checkpoint(self.arg.weights, optim=False)
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.best_acc = 0
            self.best_epoch = -1
            self.eval(epoch=0)
            self.print_log('Done.\n')

        elif self.arg.phase == 'debug':
            self.arg.print_log = False
            self.train(0)
            self.eval(0)

        else:
            raise ValueError("Unknown phase: {}".format(self.arg.phase))

