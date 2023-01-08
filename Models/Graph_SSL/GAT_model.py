import dgl
import torch
from dgl.nn.pytorch.conv import GATConv
from torch import nn


class GAT_Classifier(nn.Module):
    def __init__(self, cfg, input_dim = 89):
        super(GAT_Classifier, self).__init__()
        hidden_dim = cfg.GNN.hidden_dim
        n_classes = cfg.model.number_classes

        self.layers = nn.ModuleList([
            GATConv(input_dim, hidden_dim, num_heads = 6, allow_zero_in_degree = True),
            GATConv(hidden_dim * 6, hidden_dim, num_heads = 6, allow_zero_in_degree = True),
            GATConv(hidden_dim * 6, hidden_dim, num_heads = 6, allow_zero_in_degree = True),
        ]
        )
        # self.classify = nn.Linear(hidden_dim, n_classes)
        self.classify = nn.Sequential(
                nn.Linear(hidden_dim * 6, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes)
        )
        # self.loss_func = nn.CrossEntropyLoss()
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, graphs):

        # h = graphs.in_degrees().view(-1, 1).float()
        h = graphs.ndata['nf']
        for layer_index, conv in enumerate(self.layers):
            # print('before, index:{},h shape:{}'.format(layer_index,h.shape))
            h = conv(graphs, h)
            h = torch.flatten(h, start_dim = 1)
            # print('after, index:{},h shape:{}'.format(layer_index, h.shape))
        graphs.ndata['h'] = h
        hg = dgl.mean_nodes(graphs, 'h')
        out = self.classify(hg)
        return out
