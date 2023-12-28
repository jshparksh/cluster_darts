""" CNN cell for architecture search """
import torch
import torch.nn as nn
import genotypes as gt
from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, multiplier, C_pp, C_p, C, reduction_p, reduction): #, cell_id):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.multiplier = multiplier
        self._mixed_op_feature = {}
        self.C = C
        self.cluster = False
        self.fixed_info = None
        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp(C, stride) #, cell_id, i, j) #i=node_id, j=edge_id maybe for visualization
                self.dag[i].append(op) # self.dag.append(op)
        self._mixed_op_feature = self.mixed_op_feature()

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        """
        offset = 0
        for i in range(self.n_nodes):
            s = sum(self.dag[offset+j](h, w_dag[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        """
        if self.cluster==True:
            node_idx = 0
            for edges, w_list in zip(self.dag, w_dag):
                s_cur = 0
                if node_idx == self.fixed_info[0][0]:
                    w_list = torch.cat((w_list, torch.ones(len(gt.PRIMITIVES_SECOND)).unsqueeze(0).cuda()), dim=0) # to match the size of w_list
                    w_list[[node_idx, -1]] = w_list[[-1, node_idx]] # swap the position of the fixed op and the last op
                    for i, (s,w) in enumerate(zip(states, w_list)):
                        if i == self.fixed_info[0][1]:
                            s_out = edges[i](s) # edges[i] is the fixed op, no need for w
                        else:
                            s_out = edges[i](s, w)
                        s_cur += s_out
                elif node_idx == self.fixed_info[1][0]:
                    w_list = torch.cat((w_list, torch.ones(len(gt.PRIMITIVES_SECOND)).unsqueeze(0).cuda()), dim=0)
                    w_list[[node_idx, -1]] = w_list[[-1, node_idx]]
                    for i, (s,w) in enumerate(zip(states, w_list)):
                        if i == self.fixed_info[1][1]:
                            s_out = edges[i](s)
                        else:
                            s_out = edges[i](s, w)
                        s_cur += s_out
                else:
                    s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
                states.append(s_cur)
                node_idx += 1
        else:
            for edges, w_list in zip(self.dag, w_dag):
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
                states.append(s_cur)
        
        for i in range(self.n_nodes):
            for j in range(2+i):
                feature_str = "node{}_edge{}".format(i, j)
                if self.cluster == True:
                    if (i == self.fixed_info[0][0] and j == self.fixed_info[0][1]) or (i == self.fixed_info[1][0] and j == self.fixed_info[1][1]):
                        self._mixed_op_feature[feature_str] = None
                    else:
                        self._mixed_op_feature[feature_str] = self.dag[i][j].feature()
                else:
                    self._mixed_op_feature[feature_str] = self.dag[i][j].feature() #ops.MixedOp().feature()
        return torch.cat(states[-self.multiplier:], dim=1)

    def mixed_op_feature(self):
        return self._mixed_op_feature

    def _swap_dag(self, fixed_info):
        # fix_info = [(node_idx, edge_idx, op_type), ...]
        # should call swap_ops() at ops.py first to swap MixedOp
        self.new_dag = nn.ModuleList()
        self.cluster = True
        self.fixed_info = fixed_info
        for i in range(self.n_nodes):
            self.new_dag.append(nn.ModuleList())
            for j in range(2+i):
                stride = 2 if self.reduction and j < 2 else 1
                if i == fixed_info[0][0] and j == fixed_info[0][1]:
                    op = ops.OPS[fixed_info[0][2]](self.C, stride, affine=False)
                elif i == fixed_info[1][0] and j == fixed_info[1][1]:
                    op = ops.OPS[fixed_info[1][2]](self.C, stride, affine=False)
                else:
                    op = ops.MixedOp_Fixed(self.C, stride)
                self.new_dag[i].append(op)
        self.dag = self.new_dag