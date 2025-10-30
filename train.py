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
import concurrent.futures
import torch
import torch.nn as nn
from torch.optim.adamax import Adamax
from torch.optim.adadelta import Adadelta

import src
from src import utils
from src.data import build_train_loader
from src.data import build_val_loader
from src.data import build_otf_train_loader


logger = logging.getLogger(__name__)

#device
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-cover-dir',
        default="cover test path"",
    )
    parser.add_argument(
        '--train-stego-dir',
        default="cover test path"",
    )
    parser.add_argument(
        '--val-cover-dir',
        default="cover test path"",
       
    )
    parser.add_argument(
        '--val-stego-dir',
        default="cover test path"",

    )
    parser.add_argument(
        '--test-cover-dir',
        default="cover test path"",
    )
    parser.add_argument(
        '--test-stego-dir',
        default="cover test path"",
       
    )

    parser.add_argument('--epoch', dest='epoch', type=int, default=500)  
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--wd', dest='wd', type=float, default=0)
    parser.add_argument('--eps', dest='eps', type=float, default=1e-8)

    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)#32
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=10)

    parser.add_argument('--finetune', dest='finetune', type=str, default=None)
    parser.add_argument('--seed', dest='seed', type=int, default=-1)
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=250)#250
    parser.add_argument('--ckpt-dir',default='Model save path')
    parser.add_argument('--log-dir',default='log save path')
    
    args = parser.parse_args()
    return args


def setup(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)

    log_file = osp.join(args.log_dir, 'log.txt')
    utils.configure_logging(file=log_file, root_handler_type=0)

    utils.set_random_seed(None if args.seed < 0 else args.seed)

    logger.info('Command Line Arguments: {}'.format(str(args)))



args = parse_args()
setup(args)
with open(os.path.join(args.ckpt_dir, 'log_train.csv'), 'w') as f:
    f.write('epoch,train_loss,train_acc,valid_loss,valid_acc,test_acc\n')


logger.info('Building data loader')



train_loader, epoch_length = build_train_loader(
        args.train_cover_dir, args.train_stego_dir, batch_size=args.batch_size,
        num_workers=args.num_workers
)
val_loader = build_val_loader(
    args.val_cover_dir, args.val_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)
test_loader = build_val_loader(
    args.test_cover_dir, args.test_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)

train_loader_iter = iter(train_loader)


net = src.models.HSMNet()


if args.finetune is not None:
    checkpoint = torch.load(
    args.finetune,
    map_location=torch.device('cuda:0')
    )
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    # net.load_state_dict(torch.load(args.finetune)['state_dict'], strict=False)

criterion_1 = nn.CrossEntropyLoss()



net.to(device)
criterion_1.to(device)


optimizer = Adamax(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)


scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,400],  #18,25   30,45  #300,400
                                                     gamma=0.1)  # milestones=[900,975]




def train(epoch):
    net.train()
    running_loss, running_accuracy = 0., 0.

    for batch_idx in range(epoch_length):
        data = next(train_loader_iter)
        image=data['image']
        image=image.to(device)
        labels=data['label']
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs= net(image)
        loss = criterion_1(outputs, labels) 

        accuracy = src.models.accuracy(outputs, labels).item()
        running_accuracy += accuracy
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            running_accuracy /= args.log_interval
            running_loss /= args.log_interval
            
            logger.info(
                'Train epoch: {} [{}/{}]\tAccuracy: {:.2f}%\tLoss: {:.6f}'.format(
                    epoch, batch_idx + 1, epoch_length, 100 * running_accuracy,
                    running_loss))
                    
            ###############################log per log_interval start
            is_best=False
            save_checkpoint(
                {
                    'iteration': batch_idx + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': running_accuracy,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
                best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))
            ###############################
            running_loss = 0.
            running_accuracy = 0.
            net.train()
    return running_loss, running_accuracy

def valid():
    net.eval()
    valid_loss = 0.
    valid_accuracy = 0.
    with torch.no_grad():
        for data in val_loader:
            image=data['image']
            image=image.to(device)
            labels=data['label']
            labels=labels.to(device)

            outputs= net(image)
            
            valid_loss += criterion_1(outputs, labels).item() 
            
            valid_accuracy += src.models.accuracy(outputs, labels).item()
    valid_loss /= len(val_loader)
    valid_accuracy /= len(val_loader)
    logger.info('Test set: Loss: {:.4f}, Accuracy: {:.2f}%)'.format(
        valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy

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
    test_loss /= len(val_loader)
    test_accuracy /= len(val_loader)
    logger.info('Test set: Loss: {:.4f}, Accuracy: {:.2f}%)'.format(
        test_loss, 100 * test_accuracy))
    return test_loss, test_accuracy

def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


_time = time.time()
best_accuracy = 0.
for e in range(1, args.epoch + 1):
    logger.info('Epoch: {}'.format(e))
    logger.info('Train')
    train_loss,train_acc=train(e)
    logger.info('Time: {}'.format(time.time() - _time))
    logger.info('Test')
    val_loss, val_accuracy = valid()
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_accuracy)
    else:
        scheduler.step()
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        is_best = True
    else:
        is_best = False
    with open(os.path.join(args.ckpt_dir, 'results_xiaorong.csv'), 'a') as f:
        f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
            e,
            train_loss,
            train_acc,
            val_loss,
            val_accuracy,
        ))
    logger.info('Best accuracy: {}'.format(best_accuracy))
    logger.info('Time: {}'.format(time.time() - _time))
    save_checkpoint(
        {
            'epoch': e,
            'state_dict': net.state_dict(),
            'best_prec1': val_accuracy,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
        best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))

test_module_path=os.path.join(args.ckpt_dir, 'model_best.pth.tar')
if test_module_path is not None:
     net.load_state_dict(torch.load(test_module_path)['state_dict'], strict=False)

test_loss, test_accuracy = test()
print("test acc:  "+str(test_accuracy))