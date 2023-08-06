import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import GATConv, GINConv, GCNConv, GINEConv, Linear, global_mean_pool, BatchNorm, GraphConv

class GINModel(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, embedding_dim, num_layers, dropout_prob):
        super(GINModel, self).__init__()
        self.embedding_node = nn.ModuleList()
        self.embedding_edge = nn.ModuleList()
        # print('--------->')
        # print(num_node_features)
        if num_node_features == 1:
            # print('hh')
            self.embedding_node.append(nn.Embedding(120, embedding_dim))
        elif num_node_features == 9:
            n_f = [120, 5, 12, 12, 10, 6, 7, 2, 2]
            
            for i in range(num_node_features):
                self.embedding_node.append(nn.Embedding(n_f[i], hidden_dim))
        if num_edge_features == 1:
            self.embedding_edge.append(nn.Embedding(5, embedding_dim))
        elif num_edge_features == 3:
            e_f = [15, 6, 2]
            
            for i in range(num_edge_features):
                self.embedding_edge.append(nn.Embedding(e_f[i], embedding_dim))

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            ))
            self.convs.append(conv)

            batch_norm = nn.BatchNorm1d(embedding_dim)
            self.batch_norms.append(batch_norm)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, edge_index, edge_attr, batch):
        if len(x.shape) == 1:
            new_x = self.embedding_node[0](x)
        else:
            new_x = self.embedding_node[0](x[:,0].long().squeeze())
            for i in range(1, len(self.embedding_node)):
                new_x += self.embedding_node[i](x[:,i].long().squeeze())
        if len(edge_attr.shape) == 1:
            new_edge_attr = self.embedding_edge[0](edge_attr)
        else:
            new_edge_attr = self.embedding_edge[0](edge_attr[:,0].long().squeeze())  # Apply edge attribute embedding
            for i in range(1, len(self.embedding_edge)):
                new_edge_attr += self.embedding_edge[i](edge_attr[:,i].long().squeeze())

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            new_x = conv(new_x, edge_index, new_edge_attr)
            new_x = batch_norm(new_x)
            new_x = F.relu(new_x)
            if i != len(self.convs)-1:
                new_x = self.dropout(new_x)
        new_x = global_mean_pool(new_x, batch)
        return new_x

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2, num_layers=5):
        super(GIN, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        self.classifier = Linear(hidden_dim, num_classes)
        self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))))
        self.linear = Linear(num_features, hidden_dim)

    def forward(self, x, data):
        edge_index = data.edge_index
        for i in range(len(self.convs)-1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            if i != len(self.convs) - 2:
                x = F.dropout(x, p=0.0, training=self.training)
            # x = self.convs[-1](x, edge_index)
            x = self.classifier(x)
            return x