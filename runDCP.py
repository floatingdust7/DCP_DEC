
from __future__ import print_function, division

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
from torch.optim import Adam
from evaluation import eva
from utils import load_graph
from models import AE, GAE
import numpy as np


class DCP_DEC(nn.Module):
    '''
    n_enc_1: 编码器第一层的神经元数量。
    n_dec_1: 解码器第一层的神经元数量。
    n_input: 输入特征的维度。
    n_z: 潜在空间（编码后的特征表示）的维度。
    n_clusters: 聚类任务中的簇数量。
    v: 一个可选参数，默认值为1，可能与网络的某些操作相关。
    '''
    def __init__(self, n_enc_1, n_dec_1, n_input, n_z,
                 n_clusters, v=1):
        super(DCP_DEC, self).__init__()

        # autoencoder for intra information
        '''创建一个自编码器实例 ae'''
        self.ae = AE(n_enc_1=n_enc_1, n_dec_1=n_dec_1, n_input=n_input, n_z=n_z)
        '''加载预训练的自编码器模型权重'''
        self.ae.load_state_dict(torch.load(args.dnn_pretrain_path, map_location='cpu'))
        # cluster layer
        '''定义了一个可学习的参数 dnn_cluster_layer'''
        self.dnn_cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.dnn_cluster_layer.data)

        # GCN for inter information
        self.gae = GAE(input_feat_dim=n_input, hidden_dim1=512, n_z=n_z)
        self.gae.load_state_dict(torch.load(args.gcn_pretrain_path, map_location='cpu'))
        # cluster layer
        self.gcn_cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.gcn_cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, dz = self.ae(x)

        # GCN Module
        a_bar, gz = self.gae(x, adj)

        # Dual Self-supervised Module for DNN
        q_dnn = 1.0 / (1.0 + (torch.sum(torch.pow(dz.unsqueeze(1) - self.dnn_cluster_layer, 2), 2) / self.v))
        q_dnn = q_dnn.pow((self.v + 1.0) / 2.0)
        q_dnn = (q_dnn.t() / torch.sum(q_dnn, 1)).t()

        # Dual Self-supervised Module for GAE
        q_gcn = 1.0 / (1.0 + (torch.sum(torch.pow(gz.unsqueeze(1) - self.gcn_cluster_layer, 2), 2) / self.v))
        q_gcn = q_gcn.pow((self.v + 1.0) / 2.0)
        q_gcn = (q_gcn.t() / torch.sum(q_gcn, 1)).t()

        return x_bar, q_dnn, a_bar, q_gcn, dz, gz


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_dcp(model, adj, data, y):

    optimizer = Adam(model.parameters(), lr=args.lr)

    # k-menas initial for cluster centers of AE-based module
    with torch.no_grad():
        _, dz = model.ae(data)
    dnn_kmeans = KMeans(n_clusters=y.max()+1, n_init=20)
    # print(dnn_kmeans)
    y_dnnpred = dnn_kmeans.fit_predict(dz.data.cpu().numpy())
    model.dnn_cluster_layer.data = torch.tensor(dnn_kmeans.cluster_centers_).to(device)
    eva(y, y_dnnpred, 'dnn-pre')

    # k-menas initial for cluster centers of GAE-based module
    with torch.no_grad():
        _, gz = model.gae(data, adj)
    gcn_kmeans = KMeans(n_clusters=y.max()+1, n_init=20)
    # print(gcn_kmeans)
    y_gcnpred = gcn_kmeans.fit_predict(gz.data.cpu().numpy())
    model.gcn_cluster_layer.data = torch.tensor(gcn_kmeans.cluster_centers_).to(device)
    eva(y, y_gcnpred, 'gae-pre')

    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        if epoch % 1 == 0:
            # update_interval
            _, tmp_qdnn, _, tmp_qgcn, tmp_dz, tmp_gz = model(data, adj)

            p_dnn = target_distribution(tmp_qdnn.data)
            p_gcn = target_distribution(tmp_qgcn.data)

            res1 = tmp_qdnn.data.cpu().numpy().argmax(1)  # Q_dnn
            res2 = p_dnn.data.cpu().numpy().argmax(1)  # P_dnn
            res3 = tmp_qgcn.data.cpu().numpy().argmax(1)  # Q_gcn
            res4 = p_gcn.data.cpu().numpy().argmax(1)  # P_gcn

            qdnn = eva(y, res1, str(epoch) + ' Q_DNN')
            eva(y, res2, str(epoch) + ' P_DNN')
            qgcn = eva(y, res3, str(epoch) + ' Q_GCN')
            eva(y, res4, str(epoch) + ' P_GCN')

        x_bar, q_dnn, a_bar, q_gcn, dz, gz = model(data, adj)

        dnn_cluloss = F.kl_div(q_dnn.log(), p_dnn, reduction='batchmean')  # dnn_cluster
        dnn_reloss = F.mse_loss(x_bar, data)  # dnn_reconstruction
        gcn_cluloss = F.kl_div(q_gcn.log(), p_gcn, reduction='batchmean')  # gcn_cluster
        gcn_reloss = F.mse_loss(a_bar, adj.to_dense())  # gcn_reconstruction

        # clustering distribution consistency
        con_loss = F.kl_div(q_dnn.log(), q_gcn, reduction='batchmean')  # GCN guide

        ae_loss = args.alpha * dnn_cluloss + args.rae * dnn_reloss
        gae_loss = args.beta * gcn_cluloss + 1.0 * gcn_reloss
        loss = ae_loss + gae_loss + args.gamma * con_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return qgcn, qdnn


if __name__ == "__main__":
    #使用了 Python 的 argparse 模块来定义命令行接口，用于解析运行脚本时提供的命令行参数。
    #创建一个 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    #潜在空间的维度
    parser.add_argument('--n_z', default=64, type=int)
    #预训练模型
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--rae', type=int, default=1)
    #解析命令行输入的参数，并将解析后的参数赋值给 args 变量。parse_args 方法会从 sys.argv 中获取命令行参数，并根据前面定义的规则进行解析。
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("use cuda: {}".format(device))

    args.dnn_pretrain_path = './pretrain/{}_ae.pkl'.format(args.name)
    args.gcn_pretrain_path = './pretrain/{}_gae.pkl'.format(args.name)

    np.random.seed(0)
    torch.manual_seed(0)

    # A / KNN Graph, feature and label
    adj, feature, label = load_graph(args.name, args.k)
    adj = adj.to(device)
    feature = torch.FloatTensor(feature).to(device)
    #设置输入特征的维度 n_input，它是特征矩阵的第二维度（列数），代表每个节点的特征数量。
    n_input = feature.shape[1]
    #如果所有的类别标签都是连续的，并且从0开始，那么通过计算 label 向量中的最大值并加1是一种快速简便的方法来确定类别的总数
    n_clusters = label.max() + 1

    model = DCP_DEC(512, 512, n_input=n_input, n_z=args.n_z, n_clusters=n_clusters, v=1).to(device)

    print("Start training...............")
    result_qgcn, result_qdnn = train_dcp(model, adj, feature, label)
    print(".........................")
    print("The result of Q-GAE:")
    print(result_qgcn)
    print("The result of Q-AE:")
    print(result_qdnn)
    print(".........................")
