import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from layers import GraphConvolution


class AE(nn.Module):
    '''
    n_enc_1: 第一层编码器的输出维度。
    n_dec_1: 第一层解码器的输出维度。
    n_input: 输入数据的维度。
    n_z: 编码器输出的潜在表示（也称为瓶颈层或编码层）的维度。
    '''
    def __init__(self, n_enc_1, n_dec_1,
                 n_input, n_z):
        super(AE, self).__init__()
        #创建第一个编码器层，是一个线性层，输入维度为n_input，输出维度为n_enc_1
        self.enc_1 = Linear(n_input, n_enc_1)
        # self.enc_2 = Linear(n_enc_1, n_enc_2)
        #创建一个线性层，将编码器的输出映射到潜在表示空间，输入维度为n_enc_1，输出维度为n_z。
        self.z_layer = Linear(n_enc_1, n_z)

        #创建第一个解码器层，是一个线性层，输入维度为n_z，输出维度为n_dec_1。
        self.dec_1 = Linear(n_z, n_dec_1)
        # self.dec_2 = Linear(n_dec_1, n_dec_2)
        #创建最后一个解码器层，将解码器的输出映射回原始输入空间，输入维度为n_dec_1，输出维度为n_input。
        self.x_bar_layer = Linear(n_dec_1, n_input)

    def forward(self, x):
        #输入数据x通过self.enc_1线性层进行变换，然后使用F.relu函数（PyTorch中的ReLU激活函数）进行非线性激活。
        enc_h1 = F.relu(self.enc_1(x))
        # enc_h2 = F.relu(self.enc_2(enc_h1))
        #编码器的输出enc_h1通过self.z_layer线性层变换，得到潜在表示z。
        z = self.z_layer(enc_h1)

        #解码器的第一层操作，将潜在表示z通过self.dec_1线性层变换，并使用ReLU激活函数。
        dec_h1 = F.relu(self.dec_1(z))
        # dec_h2 = F.relu(self.dec_2(dec_h1))
        #解码器的输出dec_h1通过self.x_bar_layer线性层变换，得到重构的输入x_bar。
        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, z


'''
为了保持原始网络结构，并使学习到的表示更具区分性，引入了一个解码器。该解码器通过节点潜在表示的内积来重建节点间的关系
'''
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        # self.dropout = dropout
        self.act = act

    def forward(self, z):
        # z = F.dropout(z, self.dropout, training=self.training)
        '''
        计算输入 z 和其转置 z.t() 之间的矩阵乘法，得到一个内积矩阵。这个内积矩阵的每个元素 [i][j] 表示 z[i] 和 z[j] 之间的点积。
        然后，使用类的属性 self.act（默认为 Sigmoid 函数）对这个内积矩阵进行激活，得到一个介于 0 和 1 之间的邻接矩阵 adj。
        '''
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, n_z):
        super(GAE, self).__init__()
        #创建第一个图卷积层 gc1，用作自编码器的编码器部分的第一层。它将输入特征从 input_feat_dim 维度映射到 hidden_dim1 维度。
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1)
        #创建第二个图卷积层 gz，用作编码器的第二层，将 hidden_dim1 维度的特征进一步编码到 n_z 维度的潜在表示空间。
        self.gz = GraphConvolution(hidden_dim1, n_z)
        '''
        self.dc 使用恒等激活函数（即 lambda x: x），这表明在给定的 GAE 类构造函数中，解码器的输出直接使用内积的结果。 
        本来InnerProductDecoder就有sigmoid函数。
        '''
        self.dc = InnerProductDecoder(act=lambda x: x)

    def forward(self, x, adj):
        hidden1 = F.relu(self.gc1(x, adj))
        # z = F.relu(self.gz(hidden1, adj))
        z = self.gz(hidden1, adj)

        a_bar = self.dc(z)

        return a_bar, z
