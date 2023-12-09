""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
import torch
import torch.nn as nn

from models import ops
from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'conv_3x3_16',
    'conv_3x3_8',
    'conv_3x3_4',
    'sep_conv_3x3_16',
    'sep_conv_3x3_8',
    'sep_conv_3x3_4',
    'dil_conv_3x3_16',
    'dil_conv_3x3_8',
    'dil_conv_3x3_4',
    'max_pool_3x3',
    'avg_pool_3x3', 
    'skip_connect',
    'none'
]
PRIMITIVES_FIRST = [
    'conv_3x3_16',
    'sep_conv_3x3_16',
    'dil_conv_3x3_16',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'none'
]
PRIMITIVES_SECOND = [
    'conv_3x3_16',
    'conv_3x3_8',
    'conv_3x3_4',
    'sep_conv_3x3_16',
    'sep_conv_3x3_8',
    'sep_conv_3x3_4',
    'dil_conv_3x3_16',
    'dil_conv_3x3_8',
    'dil_conv_3x3_4',
    'none'
]
PRIMITIVES_GROUPS = [
    [
    'conv_3x3_16',
    'conv_3x3_8',
    'conv_3x3_4'],
    [
    'sep_conv_3x3_16',
    'sep_conv_3x3_8',
    'sep_conv_3x3_4'],
    [
    'dil_conv_3x3_16',
    'dil_conv_3x3_8',
    'dil_conv_3x3_4']
]


def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype

def fix_np(alpha, k):
    # fix two top value np layer and return info
    # k: number of fixed np edges
    np_info = []
    np_edge = []
    node_idx = 0

    for edges in alpha:
        np_max, primitive_indices = torch.topk(edges[:,3:-1], 1) # 3 is the index of first np edge
        topk_edge_values, topk_edge_indices = torch.topk(np_max.view(-1), k)
        primitive_indices += 3
        for i in range(len(topk_edge_values)):
            edge_idx = topk_edge_indices[i]
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES_FIRST[prim_idx]
            if len(np_edge) != 2:
                np_edge.append((node_idx, edge_idx.item(), prim, topk_edge_values[i].item()))
            else:
                for j in range(len(np_edge)):
                    if np_edge[j][3] < topk_edge_values[i].item():
                        if j == 0:
                            np_edge[j+1] = np_edge[j]
                        np_edge[j] = (node_idx, edge_idx.item(), prim, topk_edge_values[i].item())
                        break
        node_idx += 1
    
    for data in np_edge:
        # to return w/o alpha values
        np_info.append(data[:-1])
        
    return np_info
    
def parse(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """
    gene = []
    assert PRIMITIVES_FIRST[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES_FIRST[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene

def parse_fixed(alpha, k, fixed_info):
    # edges: Tensor(n_edges, n_ops)
    # fixed_info: [(fixed_node_idx, fixed_edge_idx, fixed_op_type (should be same with PRIMITIVE)), ...]
    # to do: how to get edge's type (mixedOp or not)?
    # if I get type -> fix the non-mixedOp edge into that edge
    # change k value for that node 
    gene = []
    assert PRIMITIVES_SECOND[-1] == 'none' # assume last PRIMITIVE is 'none'
    node_idx = 0
    
    for edges in alpha:
        if node_idx == fixed_info[0][0]:
            edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none' 
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k-1)
            node_gene = []
            node_gene.append((fixed_info[0][2], fixed_info[0][1]))
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                prim = PRIMITIVES_SECOND[prim_idx]
                if edge_idx >= fixed_info[0][1]:
                    node_gene.append((prim, edge_idx.item()+1))
                else:
                    node_gene.append((prim, edge_idx.item()))
        elif node_idx == fixed_info[1][0]:
            edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none' 
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k-1)
            node_gene = []
            node_gene.append((fixed_info[1][2], fixed_info[1][1]))
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                prim = PRIMITIVES_SECOND[prim_idx]
                if edge_idx >= fixed_info[1][1]:
                    node_gene.append((prim, edge_idx.item()+1))
                else:
                    node_gene.append((prim, edge_idx.item()))
        else:
            edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none' 
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
            node_gene = []
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                prim = PRIMITIVES_SECOND[prim_idx]
                node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)
        node_idx += 1

    return gene
"""
def save_alphas(alphas, primitives, save_dir, epoch=None, is_best=False):
    if epoch is not None:
        alpha_file = os.path.join(save_dir, "alphas_{}.pk".format(epoch))
    else:
        alpha_file = os.path.join(save_dir, "alphas.pk")
    with open(alpha_file, "wb") as f:
        pickle.dump(alphas, f)

    genotypes = [
        build_genotype_from_alpha(alpha, primitive, 2) 
        for alpha, primitive in zip(alphas, primitives)]
    if epoch is not None:
        genotype_file = os.path.join(save_dir, "genotypes_{}.pk".format(epoch))
    else:
        genotype_file = os.path.join(save_dir, "genotypes.pk")
    with open(genotype_file, "wb") as f:
        pickle.dump(genotypes, f)

    if is_best:
        shutil.copyfile(
            alpha_file, os.path.join(save_dir, "alphas_best.pk"))
        shutil.copyfile(
            genotype_file, os.path.join(save_dir, "genotypes_best.pk"))

        for i, genotype in enumerate(genotypes):
            plot_cell(genotype=genotype,
                      name="cell_{}".format(i),
                      save_dir=os.path.join(save_dir, "dags"))

    return genotypes
"""