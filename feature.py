# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import genotypes
from config import SearchConfig
from utils import get_data
from models.search_cnn import SearchCNNController
import feature_map as fmp
import utils


config = SearchConfig()
config.feature_dir = os.path.join(config.path, "features", config.feature_epoch)
os.system("mkdir -p {}".format(config.feature_dir))
config.stage = "feature"
config.total_samples = 10000
config.log_dir = os.path.join("searchs", config.name)
device = torch.device("cuda")

logger = utils.get_logger(
    os.path.join(config.log_dir, "{}.log".format(
        config.name)))


def compute_online(data_loader, model, feature_dir):
    logger.info("computing online...")
    save_dir = os.path.join(feature_dir, "online")
    os.system("mkdir -p {}".format(save_dir))

    op_names = genotypes.PRIMITIVES_SECOND
    num_features = len(op_names)

    lambda_zero_means = lambda: [0.0 for _ in range(num_features)]
    lambda_zero_covariances = lambda: [
        [0.0 for j in range(i, num_features)] for i in range(num_features)]
    feature_str = "cell{}_node{}_edge{}"
    # concern_edges = [
    #     [i, j] 
    #     for i in range(config.num_nodes) for j in range(i + 2)]
    concern_edges = [
        [0, 0], 
        [config.num_nodes - 1, config.num_nodes]]
    
    mean_list = {}
    covariance_list = {}
    for cell_id in range(config.layers):
        for edge in concern_edges:
            key = feature_str.format(cell_id, *edge)
            mean_list.update({key: lambda_zero_means()})
            covariance_list.update({key: lambda_zero_covariances()})

    cnt_samples = 0
    for step, (images, labels) in enumerate(data_loader):
        images = images.cuda() #images.to(device, non_blocking=True)
        labels = labels.cuda() #labels.to(device, non_blocking=True)
        num_samples = images.size(0)

        _ = model(images) #save=True, feature_dir=feature_dir, mode="wb")

        for key in mean_list.keys():
            feature_file = os.path.join(feature_dir, "{}.pk".format(key))
            feature_list = fmp.load_features(feature_file)
            online_feature_list = [
                [feas[i] for feas in feature_list] 
                for i in range(feature_list[0].shape[0])]

            for i, feas_list in enumerate(online_feature_list):
                _ = fmp.compute_covariance_online(
                    feature_list=feas_list,
                    num_samples=cnt_samples + i,
                    mean_list=mean_list[key],
                    covariance_list=covariance_list[key])
        cnt_samples += num_samples
        if cnt_samples == 5:
            break
    for key in mean_list.keys():
        covariance = np.zeros((num_features, num_features), np.float32)
        for i in range(num_features):
            covariance[i, i:] = covariance[i:, i] = covariance_list[key][i]

        correlation = fmp.compute_correlation(covariance)
        
        corr_file = os.path.join(save_dir, "corr_{}.txt".format(key))
        with open(corr_file, "wb") as f:
            np.savetxt(f, correlation, fmt="%8.4f", delimiter=",")

        for k in range(2, min(num_features, 4) + 1):
            dist_file = os.path.join(
                save_dir, "dist_{}_group{}.png".format(key, k))
            fmp.plot_clusters(mean_list[key], k, dist_file, annotate=op_names)

    
    logger.info("num_samples: {}".format(cnt_samples))


def main():
    if not torch.cuda.is_available():
        logger.info("no gpu device available")
        sys.exit(1)

    logger.info("*** Begin {} ***".format(config.stage))

    # set default gpu device
    torch.cuda.set_device(config.gpus[0])

    # set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    logger.info("preparing data...")
    
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
    
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:n_train]),
        pin_memory=True, num_workers=4)
    
    
    logger.info("loading model...")
    net_crit = nn.CrossEntropyLoss().cuda() #to(device)
    CLASSES = 10
    model = SearchCNNController(input_size, input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = utils.load_checkpoint(model, config.path)
    model = model.cuda() #to(device)
    model.eval()

    config.num_cells = len(model.net.cells)
    config.num_nodes = len(model.net.cells[0].dag)

    logger.info("start computing...")
    #compute_offline(train_loader, model, config.feature_dir)
    compute_online(train_loader, model, config.feature_dir)

    logger.info("*** Finish {} ***".format(config.stage))


if __name__ == "__main__":
    main()