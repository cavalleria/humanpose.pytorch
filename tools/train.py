# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import dataset
import models
from tqdm import tqdm

from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate

from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger
from utils.utils import get_model_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--seed',
                        help='random seed',
                        default=1337,
                        type=int)

    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(int(args.seed))
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, final_output_dir, tb_log_dir))
   


def main_worker(gpu, ngpus_per_node, args, final_output_dir, tb_log_dir):

    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print('Init process group: dist_url: {}, world_size: {}, rank: {}'.format(cfg.DIST_URL, args.world_size, args.rank))
    dist.init_process_group(backend=cfg.DIST_BACKEND, init_method=cfg.DIST_URL, world_size=args.world_size, rank=args.rank)

    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True)

    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (cfg.MULTIPROCESSING_DISTRIBUTED and args.rank % ngpus_per_node == 0):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'), final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if not cfg.MULTIPROCESSING_DISTRIBUTED or (cfg.MULTIPROCESSING_DISTRIBUTED and args.rank % ngpus_per_node == 0):
        dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
        writer_dict['writer'].add_graph(model, (dump_input, ))
        # logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.MODEL.SYNC_BN:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda(args.gpu)

    # Data loading code
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=(train_sampler is None),
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    logger.info(train_loader.dataset)

    best_perf = -1
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
        lr_scheduler.step()

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(gpu)
    )

    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()