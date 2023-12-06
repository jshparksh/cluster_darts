import numpy as np
import torch
from torch.autograd import Variable, grad
import sys
import math
import utils
import torch.nn.functional as F
import genotypes as gt

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, criterion, config):
        self.network_momentum = config.w_momentum
        self.network_weight_decay = config.w_weight_decay
        self.first_gpu = 'cuda:'+str(config.gpus[0])
        self.model = model
        self.criterion = criterion
        self.max_lmd = 0 #torch.nn.Parameter(torch.tensor(0.0, dtype= torch.float32, requires_grad=False))
        self.min_lmd = 100000 #torch.nn.Parameter(torch.tensor(10000, dtype= torch.float32, requires_grad=False))
        self.anchor = config.anchor
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=config.alpha_lr, betas=(0.5, 0.999),
                                          weight_decay=0.)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self._train_loss(model=self.model, input=input, target=target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch, cluster, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid, epoch, cluster)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, epoch, cluster):
        if cluster == True:
            loss = self._compute_loss(self.model(input_valid), target_valid) #, epoch)
        else:
            loss = self.criterion(self.model(input_valid), target_valid)
        loss.backward() #retain_graph=True)

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = self._compute_loss(model=unrolled_model, input=input_valid, target=target_valid)

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # Compute expression (8) from paper
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    
    def _mlc_loss(self, arch_param):
        y_pred_neg = arch_param
        norm_loss = 0
        red_loss = 0
        for i in range(self.model.n_nodes):
            norm_loss += torch.mean(torch.logsumexp(y_pred_neg[0][i], dim=-1))
            red_loss += torch.mean(torch.logsumexp(y_pred_neg[1][i], dim=-1))
        norm_loss /= self.model.n_nodes
        red_loss /= self.model.n_nodes
        aux_loss = (norm_loss + red_loss) / 2
        return aux_loss

    def _cluster_loss(self):
        # get output data from MixedOP, flatten and mean value
        loss = 0
        op_groups = gt.PRIMITIVES_GROUPS
        
        # Group indices for seperation
        indices = []
        index = 0
        for i in range(len(op_groups)):
            indices.append(index + len(op_groups[i]))
            index += len(op_groups[i])
        
        # Calculate mean of std values for loss
        iteration = 0
        mixed_cell_feature = self.model.net.mixed_cell_feature()
        
        # max value for normalization
        for cell in range(self.model.n_layers):
            for node in range(self.model.n_nodes):
                for edge in range(2+node):
                    feature = mixed_cell_feature[cell]["node{}_edge{}".format(node, edge)]
                    if self.anchor == 'True':
                        group_dist, anchor_dist = utils.compute_group_std(feature, indices, self.anchor)
                        group_dist, anchor_dist = group_dist.to(torch.device(self.first_gpu)), anchor_dist.to(torch.device(self.first_gpu))
                        if group_dist > self.max_lmd:
                            self.max_lmd = group_dist
                        if anchor_dist < self.min_lmd:
                            self.min_lmd = anchor_dist
                        loss += group_dist/self.max_lmd.detach()+ self.min_lmd.detach()/anchor_dist
                        iteration += 1
                    else:
                        std, gstd = utils.compute_group_std(feature, indices, self.anchor)
                        std, gstd = std.to(torch.device(self.first_gpu)), gstd.to(torch.device(self.first_gpu))
                        if std > self.max_lmd:
                            self.max_lmd = std
                        if gstd < self.min_lmd:
                            self.min_lmd = gstd
                        loss += std/self.max_lmd.detach() + self.min_lmd.detach()/gstd
                        iteration += 1
        loss /= iteration
        
        return loss
    
    def _compute_loss(self, input_valid, target_valid): #, epoch):
        loss = self.criterion(input_valid, target_valid)
        #weights = 0 + 50*epoch/100
        #ssr_normal = self._mlc_loss(self.model._alphas)
        cl_loss = self._cluster_loss()
        
        self.loss = loss
        #self.normal_term = weights*ssr_normal
        self.cl_loss = cl_loss
        
        if self.anchor == 'True':
            lmd1 = 1/2
            lmd2 = 1/2
            new_loss = lmd1 * loss + lmd2 * cl_loss #+ weights*ssr_normal
            
        else:
            lmd1 = 1/2
            lmd2 = 1/2
            new_loss = lmd1 * loss + lmd2 * cl_loss #+ weights*ssr_normal
            
        self.arc_loss = new_loss
        return new_loss