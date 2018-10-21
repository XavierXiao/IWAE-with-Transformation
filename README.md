# IWAE-with-Transformation
Pytorch implementaion of IWAE (Importance Weighted Auto Encoder) with Affine/TPS Transformation.

In this work, I try to combine IWAE (Burda et al.) with Spatial Transformer Networks (Jaderberg et al.), and the new model is called Tranformer Varaitional Auto Encoder (TVAE). The objective of this model is trying to disentangle some components of the latent variables to make them have specific meaning (here the meaning is spatial transformatiion). Disentangle latent space is a hot research area with many recent works (Lelarge et al; Dupont et al; Chen et al.).

Our model decomposes the laten space in two parts: z and u. z is the usual laten variables whose meaning is not clear, while u is laten variables that controls spatial transformation. Like STN, I incorporate affine transformation and Thin Plate Spline transformation.

I further combined the TVAE with IWAE to see if importance weight can help. 

Model architecture:
Our model takes MNIST image as input (input size [batch_size,784]), followed by two fully connected layers with 256 neurons and ReLU activation. the resulting 256-d vector will be mapped to the mean and log variance of z and u by 4 fully connected layers with no activation respectively. For affine transformation, u has dim 6, and for 3x3 thin plate spline, u has dim 18. Then z and u are sampled by reparametrization, and z is sent to the decoder, which also have two fully connected layers with ReLU activation, and finally a fully connected layer with 784 neurons and Sigmoid activation. The resulting output layer is reshaped in to 28x28, and applied the transformation represented by u with a grid sampler.

In doing this, the original 784-d output represent an "upright" version of the input image, and the spatial variation is captured by the tranformation parameter. Therefore part of the latent space is disentangled. This method improve the reconstruction loss (which is a bianry cross entropy loss). Remember that the objective function, which is ELBO, is reconstruction + KL.

Below is the result of experiment ran on MNIST. batch_size = 128, num_epoches = 50, optimizer = AdaDelta. For plain VAE, z_dim=60; For TVAE with affine transformation, z_dim = 54, u_dim=6; For TVAE with TPS, z_dim = 42, u_dim = 18. The result is an average on MNIST test set.

Method  | Reconstruction loss(lower is better) |
| ------------- | ------------- |
| Plain VAE  |  88.2669   |
| TVAE-affine  | 83.6312   |
| TVAE-TPS  | 82.8589   |

Another evaluation metric is NLL, which is being used in IWAE paper. NLL is an (negative) estimate of w = p(x|z)p(z)/q(z|x), and the estimate is obtained by sampling latent variables 5000 times and then take importance weighted average for each data on test set. Although IWAE is not used so far, the NLL can still be computed.

Method  | NLL|
| ------------- | ------------- |
| Plain VAE  |  101.1427   |
| TVAE-affine  | 99.1765   |
| TVAE-TPS  | 97.9127   |

Finally importance weighted sample is used, which is called "IW-TVAE". Experiment with k=5 (5 samples from posterior) is done and results are shown below. IW-TVAE is compared with usual TVAE that uses multiple monte carlo samples from posterior to estimated log probability and KL (although KL can be computed in closed form, we still estimate by MC).

Method  | NLL of IWAE| NLL of Multi-sample VAE
| ------------- | ------------- |------------- |
| Plain VAE  |  98.2965  |93.9943|
| TVAE-affine  | 96.0623   |92.7514|
| TVAE-TPS  | 93.9956  |92.1647|


Codes are partly based on Xinqiang Ding's implementation of IWAE: https://github.com/xqding/Importance_Weighted_Autoencoders

Reference:
[1]Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in neural information processing systems. 2015.

[2]Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "Importance weighted autoencoders." arXiv preprint arXiv:1509.00519 (2015).

[3]Dupont, Emilien. "Joint-VAE: Learning Disentangled Joint Continuous and Discrete Representations." arXiv preprint arXiv:1804.00104 (2018).

[4]Chen, Tian Qi, et al. "Isolating Sources of Disentanglement in Variational Autoencoders." arXiv preprint arXiv:1802.04942 (2018).
