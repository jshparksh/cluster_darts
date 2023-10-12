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

    def __init__(self, model, config):
        self.network_momentum = config.w_momentum
        self.network_weight_decay = config.w_weight_decay
        self.model = model
        self.anchor = config.anchor
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=config.alpha_lr, betas=(0.5, 0.999),
                                          weight_decay=config.alpha_weight_decay)
        
    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)

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

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, epoch, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid, epoch)
        self.optimizer.step()

    def zero_hot(self, norm_weights):
        # pos = (norm_weights == norm_weights.max(axis=1, keepdims=1))
        valid_loss = torch.log(norm_weights)
        base_entropy = torch.log(torch.tensor(2).float())
        aux_loss = torch.mean(valid_loss) + base_entropy
        return aux_loss

    def mlc_loss(self, arch_param):
        y_pred_neg = arch_param
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        aux_loss = torch.mean(neg_loss)
        return aux_loss

    def _backward_step(self, input_valid, target_valid, epoch):
        weights = 0 + 50*epoch/100
        ssr_normal = self.mlc_loss(self.model._arch_parameters)
        loss = self._val_loss(self.model, input_valid, target_valid) + weights*ssr_normal
        # loss = self._val_loss(self.model, input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = self._val_loss(model=unrolled_model, input=input_valid, target=target_valid)

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
    
    def cluster_loss(self):
        # get output data from MixedOP, flatten and mean value
        loss = 0
        
        op_names = gt.PRIMITIVES
        op_groups = gt.PRIMITIVES_GROUPS
        num_ops = len(op_names)
        
        # Group indices for seperation
        indices = []
        index = 0
        for i in range(len(op_groups)):
            indices.append(index + len(op_groups[i]))
            index += len(op_groups[i])
        
        # Calculate mean of std values for loss
        iteration = 0
        mixed_cell_feature = self.net.mixed_cell_feature()
        for cell in range(self.n_layers):
            for node in range(self.n_nodes):
                for edge in range(2+node):
                    feature = mixed_cell_feature[cell]["node{}_edge{}".format(node, edge)]
                    std_mean, gstd_mean = utils.compute_group_std(feature, indices, self.anchor)
                    loss += std_mean + 1/gstd_mean
                    iteration += 1
        loss /= iteration
        
        return loss
    
    def _compute_loss(self, logits, target, epoch):
        loss = self.criterion(logits, target)
        self.loss = loss
        
        ssr_normal = self.mlc_loss(self.model._arch_parameters)
        
        
        latency = self.model.lp
        lmd1 = 15
        lmd2 = 100
        th = 0.0005
        
        C_curr = self.config.init_channels
        img_size = 32
        for i in range(self.model._layers):
            if i in [self.model._layers//3, 2*self.model._layers//3]:
                C_curr *= 2
                reduction = True
                img_size /= 2
            else:
                reduction = False

            latency += self.model.latency_point(PRIMITIVES, self.model.genotype(), C_curr, img_size, reduction, self.est)
        
        self.lp = latency

        latency *= 2 * (10 ** (-9))

        if epoch <= 8:
            lmd1 = 0
        
        cost = loss + lmd1 * (math.log(1 + lmd2 * (_step_func(latency - th))))
        self.final_loss = cost
        return cost