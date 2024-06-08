import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

'''
这个 GraphConvolution 类实现了图卷积网络的基本构件，它可以捕捉图中节点的局部连接模式，并用于节点分类等任务。
权重矩阵 self.weight 在训练过程中会被优化，以最小化模型的损失函数。
通过 forward 方法，每个节点的输出特征是通过其自身特征和邻居特征的聚合来计算的，这是图卷积网络的核心思想。
'''

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #Parameter 是 PyTorch 中的一个类，它将一个张量包装成一个模型的参数，这样 PyTorch 就能自动地将其纳入到梯度计算和优化过程中。
        # 这意味着在反向传播时，这些权重会根据损失函数计算得到的梯度来更新。
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()


    def reset_parameters(self):
        '''
          这里计算了一个用于权重初始化的标准差 stdv。它基于权重矩阵的第二维度（self.weight.size(1)），即输出特征的数量。
          这是为了确保权重的方差在每一层保持一致，这是基于Xavier初始化（也称为Glorot初始化）的思想。
          '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        '''
        使用计算出的标准差，权重数据被初始化为一个均匀分布，范围在 -stdv 到 stdv 之间。
        这行代码直接对 self.weight 属性的 data 部分进行操作，以避免修改 Parameter 对象本身。
        '''
        self.weight.data.uniform_(-stdv, stdv)
        # torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        #这行代码计算节点特征与权重矩阵 self.weight 的矩阵乘法。
        support = torch.mm(x, self.weight)
        # 这行代码使用稀疏矩阵乘法 torch.spmm 来计算邻接矩阵 adj 和变换后的特征矩阵 support 的乘积。
        output = torch.spmm(adj, support)
        # if active:
        #     output = F.relu(output)
        return output

    #当 __repr__ 方法被调用时，它会返回一个格式化的字符串，显示类的名称以及输入和输出特征的数量。
    # 这在调试神经网络模型时特别有用，因为它可以清晰地显示层的输入和输出维度。
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
