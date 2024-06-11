
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
        '''定义了一个可学习的参数 dnn_cluster_layer
        self.dnn_cluster_layer 包含了一组随机的向量，这些向量作为初始的簇心。'''
        self.dnn_cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        '''使用 Xavier 正态分布初始化方法来初始化 dnn_cluster_layer 的权重。'''
        torch.nn.init.xavier_normal_(self.dnn_cluster_layer.data)

        # GCN for inter information
        '''创建一个图卷积自编码器实例 gae，用于学习图数据的表示。参数包括输入特征维度、一个隐藏层的维度和潜在空间维度。'''
        self.gae = GAE(input_feat_dim=n_input, hidden_dim1=512, n_z=n_z)
        self.gae.load_state_dict(torch.load(args.gcn_pretrain_path, map_location='cpu'))
        # cluster layer
        self.gcn_cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.gcn_cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        '''这行代码调用了自编码器模块 ae，输入原始特征 x，得到重构的特征 x_bar 和潜在空间的表示 dz。'''
        x_bar, dz = self.ae(x)

        # GCN Module
        '''这行代码调用了图卷积自编码器模块 gae，输入原始特征 x 和邻接矩阵 adj，得到重构的邻接矩阵 a_bar 和潜在空间的表示 gz。'''
        a_bar, gz = self.gae(x, adj)

        '''公式4'''
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
    '''公式5'''
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_dcp(model, adj, data, y):
    '''使用 Adam 优化器来优化模型参数，学习率由 args.lr 指定。'''
    optimizer = Adam(model.parameters(), lr=args.lr)

    # k-menas initial for cluster centers of AE-based module
    '''
    使用 torch.no_grad() 来禁用梯度计算，以减少内存消耗并加快计算速度，因为在这个步骤中不需要计算梯度。
    调用模型的自编码器部分 model.ae 并传入数据 data，得到潜在表示 dz。
    '''
    with torch.no_grad():
        _, dz = model.ae(data)
    '''
    使用 KMeans 算法从 dz 计算得到的潜在表示中初始化聚类中心。n_clusters=y.max()+1 确定聚类的数量，这里使用 y（真实标签）中的最大值加1作为聚类数目。
    n_init=20 表示 KMeans 算法将随机初始化质心20次，以找到最佳的聚类结果。
    '''
    dnn_kmeans = KMeans(n_clusters=y.max()+1, n_init=20)
    # print(dnn_kmeans)
    '''
    使用 KMeans 对潜在表示 dz 进行拟合（fit），并预测每个样本最有可能属于的聚类中心的标签 y_dnnpred。
    '''
    y_dnnpred = dnn_kmeans.fit_predict(dz.data.cpu().numpy())
    '''将 KMeans 算法找到的聚类中心（cluster_centers_）赋值给模型的 dnn_cluster_layer 参数。'''
    model.dnn_cluster_layer.data = torch.tensor(dnn_kmeans.cluster_centers_).to(device)
    '''调用 eva 函数来评估 KMeans 聚类的预测结果 y_dnnpred 与真实标签 y 之间的性能。'''
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
        '''这个条件判断确保每隔一个epoch执行一次下面的代码块。由于 epoch % 1 永远为0，这意味着它将在每个epoch执行一次。'''
        if epoch % 1 == 0:
            # update_interval
            '''获取dnn和gcn对应的公式4的分布和潜在表示'''
            _, tmp_qdnn, _, tmp_qgcn, tmp_dz, tmp_gz = model(data, adj)

            '''获取公式5的分布（目标分布）'''
            p_dnn = target_distribution(tmp_qdnn.data)
            p_gcn = target_distribution(tmp_qgcn.data)

            '''将临时和目标分布转换为 NumPy 数组，使用 argmax 函数沿着第二个维度（通常是特征维度）找到每个数据点最可能的聚类标签。'''
            res1 = tmp_qdnn.data.cpu().numpy().argmax(1)  # Q_dnn
            res2 = p_dnn.data.cpu().numpy().argmax(1)  # P_dnn
            res3 = tmp_qgcn.data.cpu().numpy().argmax(1)  # Q_gcn
            res4 = p_gcn.data.cpu().numpy().argmax(1)  # P_gcn

            '''
            使用 eva 函数评估不同分布的聚类性能。y 是真实标签，res1、res2、res3 和 res4 是不同阶段预测的聚类标签。
            字符串标识（如 'Q_DNN'、'P_DNN' 等）用于区分不同阶段或类型的聚类结果。
            '''
            qdnn = eva(y, res1, str(epoch) + ' Q_DNN')
            eva(y, res2, str(epoch) + ' P_DNN')
            qgcn = eva(y, res3, str(epoch) + ' Q_GCN')
            eva(y, res4, str(epoch) + ' P_GCN')

        x_bar, q_dnn, a_bar, q_gcn, dz, gz = model(data, adj)
        '''使用 KL 散度（Kullback-Leibler divergence）衡量 q_dnn（DNN 模块的聚类分布）与目标分布 p_dnn 之间的差异。reduction='batchmean' 表示将损失在整个批次上取平均。'''
        dnn_cluloss = F.kl_div(q_dnn.log(), p_dnn, reduction='batchmean')  # dnn_cluster
        '''使用均方误差（MSE）计算重构的数据 x_bar 与原始数据 data 之间的差异。'''
        dnn_reloss = F.mse_loss(x_bar, data)  # dnn_reconstruction
        gcn_cluloss = F.kl_div(q_gcn.log(), p_gcn, reduction='batchmean')  # gcn_cluster
        gcn_reloss = F.mse_loss(a_bar, adj.to_dense())  # gcn_reconstruction

        # clustering distribution consistency
        '''使用 KL 散度衡量 DNN 和 GCN 模块的聚类分布之间的一致性，促使两者的聚类结果趋于一致。'''
        con_loss = F.kl_div(q_dnn.log(), q_gcn, reduction='batchmean')  # GCN guide

        '''将各个损失组合起来，形成最终的损失函数。这里使用了不同的权重（args.alpha、args.rae、args.beta、args.gamma）来平衡不同损失项的贡献。'''
        ae_loss = args.alpha * dnn_cluloss + args.rae * dnn_reloss
        gae_loss = args.beta * gcn_cluloss + 1.0 * gcn_reloss
        loss = ae_loss + gae_loss + args.gamma * con_loss

        '''
        清除优化器中的旧梯度
        反向传播
        更新模型参数
        '''
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

    mode1l = DCP_DEC(512, 512, n_input=n_input, n_z=args.n_z, n_clusters=n_clusters, v=1).to(device)

    print("Start training...............")
    result_qgcn, result_qdnn = train_dcp(mode1l, adj, feature, label)
    print(".........................")
    print("The result of Q-GAE:")
    print(result_qgcn)
    print("The result of Q-AE:")
    print(result_qdnn)
    print(".........................")
