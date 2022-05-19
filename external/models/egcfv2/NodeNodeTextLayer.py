from abc import ABC

from torch_geometric.nn import MessagePassing
import torch
from torch_sparse import matmul, mul_nnz, mul, fill_diag, sum


def apply_norm(original_edge_index, current_edge_index, add_self_loops=True):
    adj_t = current_edge_index
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.)
    deg = sum(original_edge_index, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


class NodeNodeTextLayer(MessagePassing, ABC):
    def __init__(self, normalize=True):
        super(NodeNodeTextLayer, self).__init__(aggr='add')
        self.normalize = normalize
        self.activation = torch.nn.Sigmoid()

    def forward(self, x, edge_index, node_attr_rows, node_attr_cols, edge_attr):
        original_edge_index = edge_index
        # original
        weights = torch.nn.functional.cosine_similarity(torch.mul(node_attr_rows, edge_attr), torch.mul(node_attr_cols, edge_attr), dim=1)
        # new_fake
        '''
        v2
        sim_edge_items = torch.nn.functional.cosine_similarity(edge_attr, node_attr_cols, dim=1)
        sim_edge_user = torch.nn.functional.cosine_similarity(edge_attr, node_attr_rows, dim=1)
        sim_user_items = torch.nn.functional.cosine_similarity(node_attr_cols, node_attr_rows, dim=1)

        sim_edges = torch.mul(sim_edge_items, sim_edge_user)

        sim_edges = sim_edges.view(-1, 1)
        sim_user_items = sim_user_items.view(-1, 1)
        weights = torch.nn.functional.cosine_similarity(sim_edges, sim_user_items)
        
        # v3
        sim_edge_items = torch.nn.functional.cosine_similarity(edge_attr, node_attr_cols, dim=1)
        sim_edge_user = torch.nn.functional.cosine_similarity(edge_attr, node_attr_rows, dim=1)
        sim_edge_user = sim_edge_user.view(-1, 1)
        sim_edge_items = sim_edge_items.view(-1, 1)
        weights = torch.nn.functional.cosine_similarity(sim_edge_items, sim_edge_user)
        '''
        weights = self.activation(weights)  # cos(edge_attr, cos(node_attr_rows, node_attr_cols) ?
        edge_index = mul_nnz(edge_index, weights, layout='coo')

        if self.normalize:
            edge_index = apply_norm(original_edge_index, edge_index, add_self_loops=True)

        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
