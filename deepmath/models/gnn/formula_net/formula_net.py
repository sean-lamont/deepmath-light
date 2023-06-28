import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn.functional import dropout
import torch.nn.functional as F


class CombinedAggregation(nn.Module):
    def __init__(self, embedding_dim, batch_norm=True):
        super(CombinedAggregation, self).__init__()
        # self.fc = nn.Linear(embedding_dim, embedding_dim)
        # self.bn = nn.BatchNorm1d(embedding_dim)

        if batch_norm:
            self.mlp = torch.nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU()
            )
        else:
            self.mlp = torch.nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.mlp(x)
        # x = torch.relu(self.bn(self.fc(x)))
        return x


class BinaryClassifier(nn.Module):
    def __init__(self, input_shape):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_shape, input_shape)
        self.fc2 = nn.Linear(input_shape, 1)
        self.bn = nn.BatchNorm1d(input_shape)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))


#####################################################################################################
# FormulaNet with no edge attributes
#####################################################################################################

# F_o summed over children
class ChildAggregation(MessagePassing):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__(aggr='sum', flow='target_to_source')
        if batch_norm:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU())
        else:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           ReLU())

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

    def forward(self, x, edge_index):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x)


# F_i summed over parents
class ParentAggregation(MessagePassing):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__(aggr='sum', flow='source_to_target')
        if batch_norm:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           nn.BatchNorm1d(out_channels),
                           ReLU())
        else:
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels),
                           ReLU())

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)

        return self.mlp(tmp)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # edge index 1 for degree wrt parents
        deg = degree(edge_index[1], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0

        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x)


class FormulaNet(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_iterations, batch_norm=True):
        super(FormulaNet, self).__init__()

        self.num_iterations = num_iterations

        self.initial_encoder = nn.Embedding(input_shape, embedding_dim)
        self.parent_agg = ParentAggregation(embedding_dim, embedding_dim, batch_norm=batch_norm)
        self.child_agg = ChildAggregation(embedding_dim, embedding_dim, batch_norm=batch_norm)
        self.final_agg = CombinedAggregation(embedding_dim, batch_norm=batch_norm)

    def forward(self, data):
        nodes = data.x
        edges = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        # nodes = self.initial_encoder(nodes)

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges)
            fo_sum = self.child_agg(nodes, edges)
            node_update = self.final_agg(nodes + fi_sum + fo_sum)
            nodes = nodes + node_update

        return gmp(nodes, batch)


#####################################################################################################
# FormulaNet with edge attributes
#####################################################################################################

class CombinedAggregation(nn.Module):
    def __init__(self, embedding_dim, dropout =0.5):
        super().__init__()
        self.mlp = Seq(nn.Dropout(dropout),
                       Linear(3 * embedding_dim, 2 * embedding_dim),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(2 * embedding_dim, embedding_dim),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(embedding_dim, embedding_dim),
                       ReLU())

    def forward(self, x):
        x = self.mlp(x)
        return x


class ChildAggregationEdges(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__(aggr='sum', flow='target_to_source')

        self.mlp = Seq(nn.Dropout(dropout),
                       Linear(3 * in_channels, 2 * out_channels),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(2 * out_channels, out_channels),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(out_channels, out_channels),
                       ReLU())

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)

class ParentAggregationEdges(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__(aggr='sum', flow='source_to_target')

        self.mlp = Seq(nn.Dropout(dropout),
                       Linear(3 * in_channels, 2 * out_channels),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(2 * out_channels, out_channels),
                       ReLU(),
                       nn.Dropout(dropout),
                       Linear(out_channels, out_channels),
                       ReLU())

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv.view(-1, 1) * self.propagate(edge_index, x=x, edge_attr=edge_attr)


class FormulaNetEdges(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_iterations, max_edges=3, global_pool=True, dropout=0.5):
        super().__init__()
        self.num_iterations = num_iterations
        self.global_pool = global_pool

        # from paper, does 2 hidden layers mean 4 in total, or 3??
        self.initial_encoder = nn.Sequential(nn.Embedding(input_shape, embedding_dim * 2),
                                             nn.Dropout(dropout),
                                             nn.Linear(embedding_dim * 2, embedding_dim),
                                             nn.ReLU(),
                                             nn.Dropout(dropout),
                                             nn.Linear(embedding_dim, embedding_dim),
                                             nn.ReLU())

        self.edge_encoder = nn.Sequential(nn.Embedding(max_edges, embedding_dim * 2),
                                             nn.Dropout(dropout),
                                             nn.Linear(embedding_dim * 2, embedding_dim),
                                             nn.ReLU(),
                                             nn.Dropout(dropout),
                                             nn.Linear(embedding_dim, embedding_dim),
                                             nn.ReLU())

        self.parent_agg = ParentAggregationEdges(embedding_dim, embedding_dim)
        self.child_agg = ChildAggregationEdges(embedding_dim, embedding_dim)
        self.final_agg = CombinedAggregation(embedding_dim)

        # 1x1 conv equivalent to linear projection in output channel
        self.out_proj = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(embedding_dim, embedding_dim * 4),
                                      nn.ReLU(),
                                      nn.Linear(embedding_dim * 4, embedding_dim * 8),
                                      nn.ReLU())

    def forward(self, batch):
        nodes = batch.x
        edges = batch.edge_index
        edge_attr = batch.edge_attr

        nodes = self.initial_encoder(nodes)
        edge_attr = self.edge_encoder(edge_attr)

        for t in range(self.num_iterations):
            fi_sum = self.parent_agg(nodes, edges, edge_attr)
            fo_sum = self.child_agg(nodes, edges, edge_attr)
            node_update = self.final_agg(torch.cat([nodes,fi_sum,fo_sum], dim=-1))
            nodes = nodes + node_update

        nodes = self.out_proj(nodes)

        if self.global_pool:
            return gmp(nodes, batch.batch)
        else:
            return nodes

