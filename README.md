# DCP-DEC
Implementation for DCP-DEC model.

Paper: Deep Embedded Clustering with Distribution Consistency Preservation for Attributed Networks

The first round of reviews of this paper has been completed, and the revised version is being processed.

# Data
Due to the limitation of file size, we give some examples of the real-word datasets (ACM and USPS) and the generated artificial attributed networks (μ=0.6 and μ=0.8 with noise ratio 20%).

# Code & Usage
Here we provide an implementation in PyTorch, along with execution examples on real and artificial (with μ=0.6 & noise ratio is 20%) attributed networks. The repository is organised as follows:

runDCP.py: main process to train the model for real-world networks.

runDCP_lfr.py: train the model for LFR artificial attributed network.

utils.py: process the dataset before passing to the network.

utils_lfr.py: process the LFR artificial attributed network.

model.py: define the autoencoder and graph autoencoder.

evaluation.py: evaluation indicators to verify the performance of the model.

pretrain.py: pretrain the autoencoder and graph autoencoder to get node representations for initializing.

Note that for different datasets, we should change configures of the model to get the best performance.

Example:

python runDCP.py --name acm --alpha 0.1 --beta 0.8 --gamma 0.4 --rae 40 --epochs 500

python runDCP_lfr.py --name lfr100060 --alpha 0.1 --beta 0.01 --gamma 0.5

# Parameters setting

|  Dataset |     AE-enc          | GAE-enc       | Epoch |   Lr  | alpha | beta | gamma | rae |
|:--------:|:-------------------:|:-------------:|:-----:|:-----:|:-----:|:----:|:-----:|:-------:|
| Citeseer |        256-64       |     256-64    |  200  | 0.001 |  0.75 |  0.1 |   1   |    1    |
|   Cora   |        256-64       |     256-64    |  200  | 0.001 |  0.15 | 0.25 |  0.7  |    40   |
|  PubMed  |   1024-512-256-64   |     256-64    |  200  | 0.001 |  0.1  | 0.05 |  0.8  |    10   |
|    ACM   |        512-64       |     512-64    |  500  | 0.001 |  0.1  |  0.8 |  0.4  |    40   |
|   DBLP   |        512-64       |     512-64    |  500  | 0.001 |  0.05 | 0.85 |   1   |    40   |
|   USPS   | 1024-512-512-256-64 |     256-64    |  500  | 0.001 |   1   | 0.55 |  0.8  |    40   |
|   HHAR   | 1024-256-256-256-64 |     256-64    |  500  | 0.001 |  0.9  | 0.45 |  0.7  |    40   |
|  Reuters |   1024-512-256-64   |     256-64    |  500  | 0.001 |  0.9  | 0.95 |  0.1  |    10   |

Note that we adjust the weight of reconstruction loss in the AE-based module in our experiments.

# Acknowledgement
The baseline models and the reference codes are as below:

D. Bo, X. Wang, C. Shi, et al. Structural Deep Clustering Network. In WWW, 2020.

--https://github.com/bdy9527/SDCN

W. Tu, S. Zhou, X. Liu, X. Guo, Z. Cai, E. zhu, and J. Cheng. In AAAI 2021.

--https://github.com/WxTu/DFCN
