""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
import genotypes as gt
import logging
import utils
import os

from torch.autograd import Variable
from models.search_cells import SearchCell
from feature_map import save_features


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C, n_classes, n_layers, criterion, n_nodes=4, multiplier=4, stem_multiplier=3): #, device_ids=None):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.criterion = criterion
        self.n_nodes = n_nodes
        self.multiplier = multiplier
        """
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        """
        self._mixed_cell_feature = [0] * n_layers
        
        self._initialize_alphas()
        
        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, multiplier, C_pp, C_p, C_cur, reduction_p, reduction) #, i)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
        self._mixed_cell_feature = self.mixed_cell_feature()
        
    def forward(self, x):
        s0 = s1 = self.stem(x)
        """
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alpha_reduce, dim=-1)
            else:
                weights = F.softmax(self.alpha_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        """
        
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits
    
    def mixed_cell_feature(self):
        for i in range(len(self.cells)):
            self._mixed_cell_feature[i] = self.cells[i].mixed_op_feature()
        return self._mixed_cell_feature
    """
    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)
    """
    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.parameters()

    def named_weights(self):
        return self.named_parameters()

    def _initialize_alphas(self):
        # initialize architect parameters: alphas
        """
        k = sum(1 for i in range(self.n_nodes) for n in range(2+i))
        
        self.alpha_normal = nn.Parameter(1e-3*torch.randn(k, n_ops).cuda())
        self.alpha_reduce = nn.Parameter(1e-3*torch.randn(k, n_ops).cuda())
        """
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        
        for i in range(self.n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
        
        self._arch_parameters = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._arch_parameters.append((n, p))
    
    def arch_parameters(self):
        for n, p in self._arch_parameters:
            yield p
    
    def _save_features(self, path, curr_epoch):
        mixed_cell_feature = self.mixed_cell_feature()
        dir_epoch = os.path.join(path, "features", str(curr_epoch))
        os.system("mkdir -p {}".format(dir_epoch))
        for cell in range(self.n_layers):
            for node in range(self.n_nodes):
                for edge in range(2+node):
                    feature = mixed_cell_feature[cell]["node{}_edge{}".format(node, edge)]
                    feature_str = "cell{}_node{}_edge{}.pk".format(cell, node, edge)
                    save_features(feature[:-1], os.path.join(dir_epoch, feature_str))