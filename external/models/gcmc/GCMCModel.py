"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC
from torch_geometric.nn import GCNConv
from .NodeDropout import NodeDropout
from collections import OrderedDict

import torch
import torch_geometric
import numpy as np
import random

from torch_sparse import SparseTensor


class GCMCModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 convolutional_layer_size,
                 dense_layer_size,
                 n_convolutional_layers,
                 n_dense_layers,
                 node_dropout,
                 dense_layer_dropout,
                 edge_index,
                 random_seed,
                 name="GCMC",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.convolutional_layer_size = [self.embed_k] + convolutional_layer_size
        self.dense_layer_size = [self.convolutional_layer_size[-1]] + dense_layer_size
        self.n_convolutional_layers = n_convolutional_layers
        self.n_dense_layers = n_dense_layers
        self.node_dropout = torch.tensor(node_dropout, dtype=torch.float32)
        self.dense_layer_dropout = dense_layer_dropout
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        self.adj = SparseTensor(row=torch.cat([self.edge_index[0], self.edge_index[1]], dim=0),
                                col=torch.cat([self.edge_index[1], self.edge_index[0]], dim=0),
                                sparse_sizes=(self.num_users + self.num_items,
                                              self.num_users + self.num_items))

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)
        self.Q = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.dense_layer_size[-1], self.dense_layer_size[-1]))))
        self.Q.to(self.device)

        # Convolutional part
        convolutional_network_list = []
        for layer in range(self.n_convolutional_layers):
            convolutional_network_list.append((NodeDropout(self.node_dropout, self.num_users, self.num_items),
                                               'edge_index -> edge_index'))
            convolutional_network_list.append((GCNConv(in_channels=self.convolutional_layer_size[layer],
                                                       out_channels=self.convolutional_layer_size[layer + 1],
                                                       add_self_loops=True,
                                                       bias=True), 'x, edge_index -> x'))
        self.convolutional_network = torch_geometric.nn.Sequential('x, edge_index', convolutional_network_list)
        self.convolutional_network.to(self.device)

        # Dense part
        dense_network_list = []
        for layer in range(self.n_dense_layers):
            dense_network_list.append(('dense_' + str(layer), torch.nn.Linear(in_features=self.dense_layer_size[layer],
                                                                              out_features=self.dense_layer_size[
                                                                                  layer + 1],
                                                                              bias=False)))
            dense_network_list.append(('dropout_' + str(layer), torch.nn.Dropout(self.dense_layer_dropout)))
        self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
        self.dense_network.to(self.device)

        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.995)

    def propagate_embeddings(self, evaluate=False):
        current_embeddings = torch.cat((self.Gu, self.Gi), 0)

        for layer in range(0, self.n_convolutional_layers * 2, 2):
            if not evaluate:
                dropout_edge_index = list(
                    self.convolutional_network.children()
                )[layer](self.edge_index.to(self.device))
                adj = SparseTensor(row=torch.cat([dropout_edge_index[0], dropout_edge_index[1]], dim=0),
                                   col=torch.cat([dropout_edge_index[1], dropout_edge_index[0]], dim=0),
                                   sparse_sizes=(self.num_users + self.num_items,
                                                 self.num_users + self.num_items))
                current_embeddings = torch.nn.functional.relu(list(
                    self.convolutional_network.children()
                )[layer + 1](current_embeddings.to(self.device), adj.to(self.device)))
            else:
                self.convolutional_network.eval()
                with torch.no_grad():
                    current_embeddings = torch.nn.functional.relu(list(
                    self.convolutional_network.children()
                )[layer + 1](current_embeddings.to(self.device), self.adj.to(self.device)))

        if evaluate:
            self.dense_network.eval()
            with torch.no_grad():
                current_embeddings = self.dense_network(current_embeddings.to(self.device))
        else:
            current_embeddings = self.dense_network(current_embeddings.to(self.device))

        if evaluate:
            self.convolutional_network.train()
            self.dense_network.train()

        zu, zi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
        return zu, zi

    def forward(self, inputs, **kwargs):
        zu, zi = inputs
        zeta_u = torch.squeeze(zu).to(self.device)
        zeta_i = torch.squeeze(zi).to(self.device)

        xui = torch.sum(zeta_u.to(self.device) * torch.matmul(zeta_i.to(self.device), self.Q.to(self.device)), 1)

        return xui

    def predict(self, zu, zi, **kwargs):
        return torch.sigmoid(torch.matmul(zu.to(self.device), torch.matmul(self.Q.to(self.device), torch.transpose(zi.to(self.device), 0, 1))))

    def train_step(self, batch):
        zu, zi = self.propagate_embeddings()
        user, item, _ = batch
        xui = self.forward(inputs=(zu[user], zi[item]))

        log_likelihood = torch.log(self.sigmoid(xui))
        loss = - torch.sum(log_likelihood)
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2) +
                               torch.norm(self.Q, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.convolutional_network.parameters()],
                                           dim=0).sum(dim=0) +
                               torch.stack([torch.norm(value, 2) for value in self.dense_network.parameters()],
                                           dim=0).sum(dim=0))
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
