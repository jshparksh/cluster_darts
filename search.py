""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as dset
import utils

from tensorboardX import SummaryWriter
from config import SearchConfig
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

n_classes = 10

def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
    
    train_transform, valid_transform = utils._data_transforms_cifar10(config)
    train_data = dset.CIFAR10(root=config.data_path, train=True, download=True, transform=train_transform)

    criterion = nn.CrossEntropyLoss().cuda()
    #model = SearchCNN(config.init_channels, n_classes, config.layers,
    #                            criterion) #, device_ids=config.gpus)
    model = SearchCNNController(input_size, input_channels, config.init_channels, n_classes, config.layers,
                                criterion, device_ids=config.gpus)
    model = model.cuda()
    #model = model.to(device)
    """if len(config.gpus) > 1:
        model = nn.DataParallel(model, device_ids = config.gpus)
        model = model.cuda()
    else:
        model = model.cuda()"""
    # weights optimizer
    w_optim = torch.optim.SGD(model.parameters(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=0.)
    # alphas optimizer
    #alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
    #                               weight_decay=config.alpha_weight_decay)

    # split data to train/validation
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
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    
    architect = Architect(model, criterion, config)
    
    fixed_info_normal, fixed_info_reduce = None, None
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        if epoch == config.switching_epoch:
            logger.info("Switching to cluster training mode")
            config.cluster = True
            fixed_info_normal, fixed_info_reduce = model.transfer_mode()
            
            # new model and new optimizers
            model = model.cuda()
            w_optim = torch.optim.SGD(model.parameters(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=0.)
            architect.new_arch_optimizer()
            # ops -> swap_ops
            # get genotype, select top 2 skip, pooling layers and fix with search_cell -> swap_dag
        train(train_loader, valid_loader, model, architect, w_optim, lr, epoch, fixed_info_normal, fixed_info_reduce) #alpha_optim,
        """
        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)
        """
        # log
        # genotype
        if epoch >= config.switching_epoch:
            genotype = model.genotype_fixed(fixed_info_normal, fixed_info_reduce)
        else:
            genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))
        with open(os.path.join(config.path, 'genotype.txt'), 'w') as f:
            f.write(str(genotype))
        
        if config.cluster == True:
            utils.save_checkpoint(model.state_dict(), os.path.join(config.path, 'features', str(epoch)), True)
        else:
            if not os.path.exists(os.path.join(config.path, str(epoch))):
                os.mkdir(os.path.join(config.path, str(epoch)))
            utils.save_checkpoint(model.state_dict(), os.path.join(config.path, str(epoch)), False)
        print()
    """
        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)
        
        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    """

def train(train_loader, valid_loader, model, architect, w_optim, lr, epoch, fixed_info_normal, fixed_info_reduce): #alpha_optim,
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    arc_losses = utils.AverageMeter()
    cluster_losses = utils.AverageMeter()
    
    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
        val_X, val_y = val_X.cuda(), val_y.cuda()
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        architect.step(trn_X, trn_y, val_X, val_y, lr, w_optim, epoch, cluster=config.cluster, unrolled=config.unrolled)
        # phase 1. child network step (w)
        w_optim.zero_grad()
        if config.cluster == True:
            logits = model(trn_X, fixed=True)
        else:
            logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        
        w_optim.step()
        
        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
        if config.cluster == True:
            arc_loss = architect.arc_loss
            cluster_loss = architect.cl_loss
            arc_losses.update(arc_loss.item(), N)
            cluster_losses.update(cluster_loss.item(), N)
            
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            if config.cluster == True:
                logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} Arc_Loss {arc_losses.avg:.3f} Cluster_Loss {cluster_losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses, arc_losses=arc_losses, cluster_losses=cluster_losses, 
                        top1=top1, top5=top5))
            else:
                logger.info(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1
        
        """if cur_step == epoch*len(train_loader) + 3:
            break"""
        
    if config.cluster == True:
        model.net._save_features(config.path, epoch, fixed_info_normal, fixed_info_reduce)

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.cuda(), y.cuda()
            #X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
