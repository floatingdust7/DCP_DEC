# DCP
Implementation for DCP-DEC model.
Paper: Deep Embedded Clustering with Distribution Consistency Preservation for Attributed Networks

# Data
Due to the limitation of file size, we give some examples of the real-word datasets and the generated artificial attributed networks.

# Code & Usage
Here we provide an implementation in PyTorch, along with an execution example on an artificial attributed network (with μ=0.6 & noise ratio is 20%)(due to file size limit). The repository is organised as follows:

runDCP.py: main process to train the model.
utils.py: processes the dataset before passing to the network.
model.py: defines the autoencoder and graph autoencoder.
evaluation.py: evaluation indicators to verify the performance of the model.
pretrain.py: pretrain the autoencoder and graph autoeencoder to get node representations for initializing.

Note that for different datasets, we should change configures of the model to get the best performance.

Example:
python runDCP.py --name lfr100060 --alpha 0.1 --beta 0.01 --gamma 0.5

# Acknowledgement
The baseline models and the reference codes are as below:
D. Bo, X. Wang, C. Shi, et al. Structural Deep Clustering Network. In WWW, 2020.

--https://github.com/bdy9527/SDCN

W. Tu, S. Zhou, X. Liu, X. Guo, Z. Cai, E. zhu, and J. Cheng. In AAAi 2021.

--https://github.com/WxTu/DFCN
