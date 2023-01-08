import dgl
import torch
import torch.nn.functional as F
from torch import nn

# def msg(edge):
#     msg = edge.src['h']
#     return {'m': msg}

msg = dgl.function.copy_u('h', 'm')


# msg = dgl.function.src_mul_edge('h','ef','m')


def reduce(nodes):

    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):


    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        # print('NodeApplyModule h:{}'.format(h.shape))
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):

        g.ndata['h'] = feature
        g.update_all(msg, reduce, self.apply_mod)

        return g.ndata.pop('h')


class GCN_Classifier(nn.Module):
    def __init__(self, cfg, input_dim = 89):
        super(GCN_Classifier, self).__init__()
        hidden_dim = cfg.GNN.hidden_dim
        n_classes = cfg.model.number_classes

        self.layers = nn.ModuleList([
            GCN(input_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        # self.classify = nn.Linear(hidden_dim, n_classes)
        self.classify = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
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
            # print('after, index:{},h shape:{}'.format(layer_index, h.shape))
        graphs.ndata['h'] = h
        hg = dgl.mean_nodes(graphs, 'h')
        out = self.classify(hg)
        return out
