#!/usr/bin/env python
import argparse
import random

import numpy as np
import yaml
import pdb
import processors


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/debug',
        help='the work folder for storing results')

    parser.add_argument(
        '--config',
        default=None,
        help='path to the configuration file')
    parser.add_argument(
        '--processor',
        default='retrieval',
        type=str,
        help='Type of Processor')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=5,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--model-save-name',
        type=str,
        default='model',
        help='Checkpoint name')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--print-model',
        type=str2bool,
        default=False,
        help='print model or not')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--joint-model',
        default=None,
        help='the joint model will be used')
    parser.add_argument(
        '--joint-model-args',
        type=dict,
        default=dict())
    parser.add_argument(
        '--bone-model',
        default=None,
        help='the bone model will be used')
    parser.add_argument(
        '--bone-model-args',
        type=dict,
        default=dict())
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--joint-model-weight',
        type=str,
        default=None,
        help='the weights for joint model')
    parser.add_argument(
        '--bone-model-weight',
        type=str,
        default=None,
        help='the weights for bone model')
    parser.add_argument(
        '--strict-load',
        type=str2bool,
        default=False,
        help="whether load model strictly")

    # loss
    parser.add_argument(
        '--triplet-margin',
        type=float,
        default=0,
        help='Margin for triplet loss')
    parser.add_argument(
        '--softmax-coef',
        type=float,
        default=1,
        help='Coeffficient of softmax loss')
    parser.add_argument(
        '--triplet-coef',
        type=float,
        default=0.1,
        help='Coeffficient of triplet loss')
    parser.add_argument(
        '--aux-loss-coef',
        type=float,
        default=0,
        help='Coeffficient of aux loss')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--lr-args',
        type=dict,
        default={'policy': 'MultiStep',
                 'milestones': [30, 40],
                 'gammas': 0.1},
        help='Args for lr scheduler')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--part-train-epoch',
        type=int,
        default=0,
        help="# of epochs when detaching PA's gradient")
    parser.add_argument(
        '--freeze-params',
        type=list,
        default=[],
        help="Parameters which will be freezed during training")

    # testing
    parser.add_argument('--strict-test', default=False)
    parser.add_argument(
        '--top5',
        type=str2bool,
        default=False)

    # multi-gpu
    parser.add_argument(
        '--local_rank',
        type=int)
    parser.add_argument(
        '--mgpu',
        type=str2bool,
        default=False)
    return parser

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    Processor = getattr(processors, arg.processor)
    p = Processor(arg)
    p.start()
