
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling


class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels

        self.encoder_decoder = nn.Sequential(
                nn.Linear(in_channels, se_channels),
                nn.ELU(),
                nn.Linear(se_channels, in_channels),
                nn.Sigmoid(),
        )

    def forward(self, x):
        """"""
        # Aggregate input representation
        x_global = torch.mean(x, dim = 0)
        # Compute reweighting vector s
        s = self.encoder_decoder(x_global)

        return x * s


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp, use_selayer):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = (
            SELayer(self.mlp.output_dim, int(np.sqrt(self.mlp.output_dim)))
            if use_selayer
            else nn.BatchNorm1d(self.mlp.output_dim)
        )

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_selayer):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(
                        SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                        if use_selayer
                        else nn.BatchNorm1d(hidden_dim)
                )

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class UnsupervisedGIN(nn.Module):
    """GIN model"""

    def __init__(
            self,
            num_layers = 6,
            num_mlp_layers = 2,
            input_dim = 89,
            hidden_dim = 256,
            output_dim = 20,
            final_dropout = 0.5,
            learn_eps = False,
            graph_pooling_type = "sum",
            neighbor_pooling_type = "sum",
            use_selayer = False,
    ):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(UnsupervisedGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(
                        num_mlp_layers, input_dim, hidden_dim, hidden_dim, use_selayer
                )
            else:
                mlp = MLP(
                        num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, use_selayer
                )

            self.ginlayers.append(
                    GINConv(
                            ApplyNodeFunc(mlp, use_selayer),
                            neighbor_pooling_type,
                            0,
                            self.learn_eps,
                    )
            )
            self.batch_norms.append(
                    SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                    if use_selayer
                    else nn.BatchNorm1d(hidden_dim)
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = SumPooling()
        elif graph_pooling_type == "mean":
            self.pool = AvgPooling()
        elif graph_pooling_type == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        # self.loss_func = nn.CrossEntropyLoss()
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, graphs, labels = None, return_loss = True):
        # list of hidden representation at each layer (including input)

        h = graphs.ndata['nf']
        batch_size = graphs.batch_size

        hidden_rep = [h]
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](graphs, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = torch.zeros(batch_size,20).float().cuda()

        # perform pooling over all nodes in each graph in every layer
        all_outputs = []
        for i, h in list(enumerate(hidden_rep)):
            pooled_h = self.pool(graphs, h)
            all_outputs.append(pooled_h)
            score_cur = self.drop(self.linears_prediction[i](pooled_h))
            # print('shape cur:{}, score_over_layer:{}'.format(score_cur.shape,score_over_layer.shape))
            score_over_layer += score_cur

        return score_over_layer


        # if (self.training):
        #     if (return_loss):
        #         loss = self.loss_func(score_over_layer, labels)
        #         return loss
        #     else:
        #         return score_over_layer
        # else:
        #     return self.softmax(score_over_layer)
