""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import preproc


def get_data(dataset, data_path, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    shape = trn_data.data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation: # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)

def load_checkpoint(model, load_dir, epoch=None, is_best=True):
    if is_best:
        ckpt = os.path.join(load_dir, "best.pth.tar")
    elif epoch is not None:
        ckpt = os.path.join(load_dir, "checkpoint_{}.pth.tar".format(epoch))
    else:
        ckpt = os.path.join(load_dir, "checkpoint.pth.tar")
    model.load_state_dict(torch.load(ckpt))
    return model

def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform

def compute_group_std(feature_list, indices, anchor):
    groups = []
    # Divide the list into groups based on the indices
    start = 0
    for index in indices:
        group = feature_list[start:index]
        groups.append(group)
        start = index

    # Include the remaining elements after the last specified index
    if start < len(feature_list):
        groups.append(feature_list[start:])
    
    if anchor == True:
        # Distance for in group elements
        group_distance = 0
        anchor_distance = 0
        anchor_list = []
                
        for group in groups:
            anchor_list.append(group[0])
            for elem in range(1, len(group)):
                group_distance += (group[elem] - group[0]).pow(2).sum().sqrt()
            group_distance /= len(group) - 1
        group_distance /= len(groups)
        
        anchor_center = torch.mean(torch.stack(anchor_list))
        for anchor in anchor_list:
            anchor_distance += (anchor - anchor_center).pow(2).sum().sqrt()
        anchor_distance /= len(anchor_list)
        
        return group_distance, anchor_distance
            
    else:
        # Calculate the mean and std value in each group
        # To make same group's feature closer
        group_means = []
        group_stds = []
        for group in groups:
            # Initialize values for each group
            mean_list = []
            group_mean = 0
            var = 0
            std = 0
            
            for elem in range(len(group)): # elem is index in the group
                mean = torch.mean(group[elem], dim=1) # ex) elem=0 -> mean value of sep_conv_3x3_16bit's feature in channel dim
                mean_list.append(mean)
                group_mean += mean
            group_mean /= len(group)
            group_means.append(group_mean)
            
            for elem in range(len(group)):
                var += (mean_list[elem]-group_mean).pow(2)
            var /= len(group)
            std = torch.sqrt(var)
            std_mean = torch.mean(std).item()
            group_stds.append(std_mean)
        

        # Calculate std of every group's mean value
        # To make different group's feature farther
        gvar = 0
        gstd = 0
        group_mean_mean = 0
        for gm in group_means:
            group_mean_mean = torch.add(group_mean_mean, gm)
        group_mean_mean /= len(group_means)
        
        for gm in group_means:
            gvar += (gm - group_mean_mean)**2
        gvar /= len(group_means)
        gstd = torch.sqrt(gvar)
        gstd_mean = torch.mean(gstd).item()
        
        return sum(group_stds)/len(group_stds), gstd_mean