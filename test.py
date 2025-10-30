from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import os.path as osp
import shutil
import random
import time

import torch
import torch.nn as nn
from torch.optim.adamax import Adamax
from torch.optim.adadelta import Adadelta

import src
from src import utils
from src.data import build_val_loader
from src.matlab import matlab_speedy

logger = logging.getLogger(__name__)

#device
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-cover-dir',
        default="cover test path",
    )
    parser.add_argument(
        '--test-stego-dir',
        default="stego test path",
    )


    parser.add_argument('--wd', dest='wd', type=float, default=0)#1e-5
    parser.add_argument('--eps', dest='eps', type=float, default=1e-8)
    
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=10)


    parser.add_argument('--finetune', dest='finetune', type=str,                                             
                         default='Model save path')#model_best.pth.tar #checkpoint.pth.tar

    parser.add_argument('--gpu-id', dest='gpu_id', type=int, default=0)
    parser.add_argument('--seed', dest='seed', type=int, default=-1)
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=375)
    parser.add_argument('--log-dir',default='log path')


    args = parser.parse_args()
    return args


def setup(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)

    log_file = osp.join(args.log_dir, 'test.txt')
    utils.configure_logging(file=log_file, root_handler_type=0)

    utils.set_random_seed(None if args.seed < 0 else args.seed)

    logger.info('Command Line Arguments: {}'.format(str(args)))



args = parse_args()
setup(args)

logger.info('Building data loader')



test_loader = build_val_loader(
    args.test_cover_dir, args.test_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)

logger.info('Building model')

net = src.models.HSMNet()


if args.finetune is not None:
    net.load_state_dict(torch.load(args.finetune,map_location='cuda:0')['state_dict'], strict=False)
    print("Successfully loaded")

criterion_1 = nn.CrossEntropyLoss()



net.to(device)
criterion_1.to(device)



def test():
    net.eval()
    test_loss = 0.
    test_accuracy = 0.
    with torch.no_grad():
        for data in test_loader:
            image=data['image']
            image=image.to(device)
            labels=data['label']
            labels=labels.to(device)
            outputs= net(image)
            
            test_loss += criterion_1(outputs, labels).item() 
            test_accuracy += src.models.accuracy(outputs, labels).item()
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    logger.info('Test set: Loss: {:.4f}, Accuracy: {:.2f}%)'.format(
        test_loss, 100 * test_accuracy))

logger.info('start test')
test()