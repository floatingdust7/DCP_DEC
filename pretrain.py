import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
from models import AE, GAE
from utils import load_graph
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

'''
这里的预训练相比正式训练只是用了自编码器，只使用均方误差计算损失函数，学习率不一样
'''
class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    '''返回数据及其索引'''
    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


# pretrain AE
def pretrain_ae(model, dataset, y):
    savepath = "./pretrain/"
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-3)

    acc_list = []
    total_loss = []
    for epoch in range(50):
        # adjust_learning_rate(optimizer, epoch)
        '''使用 enumerate 函数遍历 train_loader。enumerate 会返回每个元素的索引和元素本身。'''
        for batch_idx, (x, _) in enumerate(train_loader):
            '''前向传播获得重构数据，并且计算重构损失'''
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # x = torch.Tensor(dataset.x).cuda().float()
            x = torch.Tensor(dataset.x).float()
            x_bar, z = model(x)

            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=int(y.max().item() + 1), n_init=20).fit(z.data.cpu().numpy())
            result = eva(y, kmeans.labels_, epoch)

        '''将评估函数 eva 的结果添加到 acc_list 列表中。'''
        acc_list.append(result)
        '''将损失的数值添加到 total_loss 列表中。'''
        total_loss.append(loss.data.cpu().tolist())
        torch.save(model.state_dict(), "{}_ae.pkl".format(savepath))

    '''
    预训练结束后，使用 matplotlib 绘制了损失和评估指标随 epoch 变化的图表。
    '''
    print(max(acc_list))
    xrange = range(0, 50)
    plt.plot(xrange, total_loss, 'o-', label='pretrain_loss')
    plt.plot(xrange, np.array(acc_list)[:, 0], 'o-', label='ACC_ae')
    plt.plot(xrange, np.array(acc_list)[:, 1], '+-', label='NMI_ae')
    plt.plot(xrange, np.array(acc_list)[:, 2], '*-', label='ARI_ae')
    plt.plot(xrange, np.array(acc_list)[:, -1], 'x-', label='PWF_ae')
    plt.legend(loc="best", fontsize=8)
    plt.title('loss vs epoch')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.show()


'''和ae的pretrain函数类似'''
# pretrain GAE
def pretrain_gae(model, adj, x, y):
    savepath = "./pretrain/"
    optimizer = Adam(model.parameters(), lr=1e-3)
    acc_list = []
    total_loss = []
    for epoch in range(50):
        a_bar, gz = model(x, adj)
        loss = F.mse_loss(a_bar, adj.to_dense())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            a_bar, gz = model(x, adj)
            loss = F.mse_loss(a_bar, adj.to_dense())
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=int(y.max().item() + 1), n_init=20).fit(gz.data.cpu().numpy())
            result = eva(y, kmeans.labels_, epoch)

        acc_list.append(result)
        total_loss.append(loss.data.cpu().tolist())
        torch.save(model.state_dict(), "{}_gae.pkl".format(savepath))

    print(max(acc_list))
    xrange = range(0, 50)
    plt.plot(xrange, total_loss, 'o-', label='pretrain_loss')
    plt.plot(xrange, np.array(acc_list)[:, 0], 'o-', label='ACC_gae')
    plt.plot(xrange, np.array(acc_list)[:, 1], '+-', label='NMI_gae')
    plt.plot(xrange, np.array(acc_list)[:, 2], '*-', label='ARI_gae')
    plt.plot(xrange, np.array(acc_list)[:, -1], 'x-', label='PWF_gae')
    plt.legend(loc="best", fontsize=8)
    plt.xlabel('loss vs. epoches')
    plt.ylabel('loss')
    plt.show()


dataname = 'acm'
k = None

x = np.loadtxt('data/{}.txt'.format(dataname), dtype=float)
y = np.loadtxt('data/{}_label.txt'.format(dataname), dtype=int)
# y = y[:, 1] - 1     # for lfr
# y = y - 1           # for cite, pubmed, hhar
dataset = LoadDataset(x)

# pretrain AE
model = AE(n_enc_1=512, n_dec_1=512, n_input=x.shape[1], n_z=64)
print(model)
pretrain_ae(model, dataset, y)

# pretrain GAE
adj, data, _ = load_graph(dataname, k)
data = torch.Tensor(data)

model = GAE(hidden_dim1=512, input_feat_dim=x.shape[1], n_z=64)
print(model)
pretrain_gae(model, adj, data, y)

