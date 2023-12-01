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
from torch.nn.parallel._functions import Broadcast
from feature_map import save_features

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies

class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, multiplier=4, stem_multiplier=3): #, device_ids=None):
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
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.multiplier = multiplier
        """
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        """
        self._mixed_cell_feature = [0] * n_layers
                
        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
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
        
    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
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
    """
    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)
    """
    
class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, input_size, C_in, C, n_classes, n_layers,
                 criterion, n_nodes=4, multiplier=4, stem_multiplier=3, device_ids=None):
        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        """# initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))"""

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, multiplier, stem_multiplier)
        
        self._initialize_alphas()

    def forward(self, x):        
        weights_normal = self.generate_weights(self.alpha_normal)
        weights_reduce = self.generate_weights(self.alpha_reduce)

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
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
        #### fix me
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        #### fix me, should copy weights into new operations
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
        n_ops = len(gt.PRIMITIVES_FIRST)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        
        for i in range(self.n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
        
        self._alphas = [self.alpha_normal, self.alpha_reduce]
        self._arch_parameters = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._arch_parameters.append((n, p))
    
    def _transfer_alphas(self, fixed_idx):
        # fixed_idx = [(node_idx, edge_idx), (node_idx, edge_idx)]
        self.new_alpha_normal = nn.ParameterList()
        self.new_alpha_reduce = nn.ParameterList()
        
        n_ops = len(gt.PRIMITIVES_SECOND)
        
        # To except one edge of fixed node's alpha
        for i in range(self.n_nodes):
            if i == fixed_idx[0] or i == fixed_idx[1]: # will not fix two edges in same node
                self.new_alpha_normal.append(nn.Parameter(torch.zeros(i+1, n_ops)))
                self.new_alpha_reduce.append(nn.Parameter(torch.zeros(i+1, n_ops)))
            else:
                self.new_alpha_normal.append(nn.Parameter(torch.zeros(i+2, n_ops)))
                self.new_alpha_reduce.append(nn.Parameter(torch.zeros(i+2, n_ops)))
                
        for i in range(self.n_nodes):
            for j in range(i+2):
                if i == fixed_idx[0] or i == fixed_idx[1]:
                    if j == i+1:
                        break
                for first_idx in range(len(gt.PRIMITIVES_FIRST)):
                    layer_type = gt.PRIMITIVES_FIRST[first_idx].split('_')[0]
                    for second_idx in range(len(gt.PRIMITIVES_SECOND)):
                        if layer_type == gt.PRIMITIVES_SECOND[second_idx].split('_')[0]:
                            self.new_alpha_normal[i][j][second_idx].data += self.alpha_normal[i][j][first_idx]
                            self.new_alpha_reduce[i][j][second_idx].data += self.alpha_reduce[i][j][first_idx]
    
        self.alpha_normal = self.new_alpha_normal
        self.alpha_reduce = self.new_alpha_reduce
    
    def arch_parameters(self):
        for n, p in self._arch_parameters:
            yield p
                    
    def generate_weights(self, alphas):
        weights = []
        
        for alpha in alphas:
            weight = torch.empty_like(alpha)
            
            for i in range(alpha.size(0)):
                denominator = torch.sum(torch.exp(alpha[i]))
                weight[i] = torch.exp(alpha[i]) / denominator
                
            weights.append(weight)
            
        return weights