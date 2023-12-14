import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.models.resnet import ResNet, BasicBlock

import pandas as pd
from PIL import Image
import glob
import matplotlib.pyplot as plt
from clearml import Task

task = Task.init(project_name="test01", task_name="task01")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--pretrained-model', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
best_acc1 = 0

# #################################################
class DogCatDataset(data.Dataset):
    def __init__(self, root, csvfile, split='train', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
        # Load all file paths
        self.data = []
        self.labels = []
        
        # Assuming two classes: 'cat' and 'dog'
        classes = {'cat': 0, 'dog': 1}

        if split == 'test':
            label_data = pd.read_csv(csvfile, names=['id', 'label'])
            label_dict = dict(zip(label_data['id'], label_data['label']))
            # Read image files and match with labels
            for img_id, label in label_dict.items():
                img_path = os.path.join(root, f'{img_id}.jpg')
                if os.path.exists(img_path):
                    self.data.append(img_path)
                    self.labels.append(classes[label.strip()])
        else:
            # Read image files
            for class_name, class_label in classes.items():
                # Glob gets a list of all files matching the pattern
                file_pattern = os.path.join(root, f'{class_name}.*.jpg')
                file_list = glob.glob(file_pattern)
                
                # Add to the dataset list
                for file in file_list:
                    self.data.append(file)
                    self.labels.append(class_label)
            
            # Shuffle the data and split
            combined = list(zip(self.data, self.labels))
            random.shuffle(combined)
            self.data[:], self.labels[:] = zip(*combined)
            
            # Split the dataset
            split_boundary = int(len(self.data) * 0.8)
            if split == 'train':
                self.data = self.data[:split_boundary]
                self.labels = self.labels[:split_boundary]
            elif split == 'valid':
                self.data = self.data[split_boundary:]
                self.labels = self.labels[split_boundary:]
            else:
                raise ValueError("split must be 'train' or 'valid'")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
# #################################################

def main():
    global best_acc1

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model")
    #model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)  # DO NOT CHANGE num_classes=10 (or num_classes=2) kwargs. (especially no do not set num_classes=1000)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # load pretrained model
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=> loading pretrained model '{}'".format(args.pretrained_model))
            #state_dict = torch.load(args.pretrained_model)

            # #################################################
            state_dict = torch.load(args.pretrained_model, map_location=lambda storage, loc: storage.cuda(args.gpu))
            # Modify the fc layer to match the number of classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2).cuda(args.gpu)
            # Load the state dict into the model
            # Since the architecture is different in the fc layer, we need to remove the pre-trained weights for this layer.
            state_dict = {k: v for k, v in state_dict.items() if k not in ['fc.weight', 'fc.bias']}
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            # Check for any missing or unexpected keys
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            # #################################################

            #model.load_state_dict(state_dict)
            print("=> loaded pretrained model '{}'".format(args.pretrained_model))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained_model))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume)
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # #################################################
    # # DogCat
    csv_file = "C:\\Users\\yuncs\\Desktop\\workspace\\dev\\my_lab\\CNN\\Lab_04_src\\FilesForLab4\\test.csv"
    test_path = "C:\\Users\\yuncs\\Desktop\\workspace\\dev\\my_lab\\CNN\\Lab_04_src\\FilesForLab4\\dogs-vs-cats\\test1"

    train_dataset = DogCatDataset(
        root=args.data,
        csvfile=csv_file,
        split='train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = DogCatDataset(
        root=args.data,
        csvfile=csv_file,
        split='valid',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    test_dataset = DogCatDataset(
        root=test_path,
        csvfile=csv_file,
        split='test',
        transform=transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor()
        ]))
    
    # #################################################

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    """
    for x_batch, y_batch in test_loader:
        print(x_batch.shape)
        print(y_batch.shape)
        print(y_batch[0])
        image = x_batch[0].permute(1, 2, 0)
        if image.shape[2] == 3:
            plt.imshow(image)
        else:
            plt.imshow(image.squeeze(), cmap='gray')
        
        plt.show()
        #break
    """

    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    # #################################################
    train_losses = []
    val_losses = []
    epochs = []
    # Plotting the training and validation losses
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Losses')
    train_line, = ax.plot(train_losses, label='Training Loss')
    val_line, = ax.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    # #################################################

    for epoch in range(args.start_epoch, args.epochs):
        epochs.append(epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        train_losses.append(train_loss)

        # evaluate on validation set
        val_loss, acc1 = validate(val_loader, model, criterion, args)
        val_losses.append(val_loss)

        acc_test = testdate(test_loader, model, criterion, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print("test acc: ", acc_test)

        save_checkpoint(model.state_dict(), is_best)

        # #################################################
        # Update the plot
        train_line.set_data(epochs, train_losses)
        val_line.set_data(epochs, val_losses)
        ax.relim()  # Recalculate limits
        ax.autoscale_view(True,True,True)  # Autoscale
        fig.canvas.draw()
        fig.canvas.flush_events()
        # #################################################
    
    plt.ioff()
    plt.show()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    
    return losses.avg


def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Valid: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return losses.avg, top1.avg


def testdate(test_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(test_loader)

    progress.display_summary()

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
