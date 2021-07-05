import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from layers import GraphConvolution


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.x_bar_layer = Linear(n_dec_2, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        # self.dropout = dropout
        self.act = act

    def forward(self, z):
        # z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, n_z):
        super(GAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1)
        self.gz = GraphConvolution(hidden_dim1, n_z)

        self.dc = InnerProductDecoder(act=lambda x: x)

    def forward(self, x, adj):
        hidden1 = F.relu(self.gc1(x, adj))
        z = F.relu(self.gz(hidden1, adj))

        a_bar = self.dc(z)

        return a_bar, z
