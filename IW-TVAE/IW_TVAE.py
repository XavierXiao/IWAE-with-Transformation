import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch
from tps import TPSGridGen
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MNIST_Dataset(Dataset):    
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')
    '''
    ramdomly binarized MNIST
    '''
    
class Encoder(nn.Module):
    '''
    encoder
    '''
    def __init__(self, input_dim, hidden_dim, output_dim,dim_u):
        '''
        input_sim = 784
        hidden_dim = 200
        output_dim = 50
        '''
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dim_u = dim_u
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU())
        
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim) #log_sd
        
        #for transformer variable
        self.u_mu = nn.Linear(hidden_dim, dim_u)
        self.u_logsigma = nn.Linear(hidden_dim, dim_u)
        
    
    def forward(self, x):
        out = self.transform(x)
        mu = self.fc_mu(out)
        logsigma = F.threshold(self.fc_logsigma(out), -6,6)
        sigma = torch.exp(logsigma)
        
        mu_u = torch.tanh(self.u_mu(out)) 
        logsigma_u = F.threshold(self.u_logsigma(out),-6,6)
        sigma_u = torch.exp(logsigma_u)
        return mu, sigma, mu_u, sigma_u
 
''' 
######################################################################################   
'''  
class IWAE_1(nn.Module):
    def __init__(self, dim_h1, dim_u, dim_image_vars,t='tps'): 
        '''
        remember we add dim_u here to represent the dimension of u
        '''
        super(IWAE_1, self).__init__()
        self.dim_h1 = dim_h1 #dim of latent variable z
        self.dim_image_vars = dim_image_vars #784
        self.dim_u = dim_u   #dimension of u
        self.tf = t

        ## encoder
        self.encoder_h1 = Encoder(dim_image_vars, 256, dim_h1,dim_u)
        
        ## decoder
        self.decoder_x =  nn.Sequential(nn.Linear(dim_h1, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, dim_image_vars),
                                        nn.Sigmoid())
        '''dim_u2 is the dim of the actual tranformation parameter'''
        if self.tf == 'aff':
            self.u2_dim = 6 # u = Wu' + b
            #the identity is a single element, need to be repeated to fit data dimension
            self.idty = torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1)
            '''This is identity transform, tensor([[1., 0., 0.],
                                                  [0., 1., 0.]])'''   
        
        elif self.tf == 'tps':
            self.u2_dim = 18 # 2 * 3^2=18
            self.gridGen = TPSGridGen(out_h=28,out_w=28,device='cuda')
            px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            self.idty = torch.cat((px,py))
            '''This is identity tps transform'''
        
        self.u2u = nn.Linear(self.dim_u,self.u2_dim)
        
    def encoder(self, x):
        #here the sigma are already "exponentialed", and mu_u is processed by a tanh
        mu_h1, sigma_h1, mu_u, sigma_u= self.encoder_h1(x)
        eps  = Variable(sigma_h1.data.new(sigma_h1.size()).normal_()).to('cuda').double()
        #eps for u
        eps2 = Variable(sigma_u.data.new(sigma_u.size()).normal_()).to('cuda').double()
        h1 = mu_h1 + sigma_h1 * eps      
        #compute u by repara
        u = mu_u + eps2*sigma_u
                 
        return h1, mu_h1, sigma_h1, u, mu_u, sigma_u
    
    def decoder(self, h1):
        p = self.decoder_x(h1)
        return p
    
    
    def forward(self, x):
        
        h1, mu_h1, sigma_h1, u, mu_u, sigma_u = self.encoder(x)
        
        p = self.decoder(h1)
        #p is of size[num_sample, batch_size, 784]
        
        '''Below: transform p by transformation defined, result p still a berboulli dist'''
       
        
        self.id = self.idty.expand((x.size()[0]*x.size()[1],)+self.idty.size())
        self.id = self.id.to('cuda')
        self.id = self.id.double()
        
        
        if self.tf == 'aff':
            
            '''test(change it to identity), remember to get it back'''
            self.theta = self.u2u(u).view(x.size()[0]*x.size()[1], 2, 3) + self.id
            
            #self.theta = self.id
            grid = F.affine_grid(self.theta, p.view(x.size()[0]*x.size()[1],28,28).unsqueeze(1).size())
            '''
            grid is of size [num_samples*batch_size,28,28,2]
            '''

        elif self.tf == 'tps':
            self.theta = self.u2u(u).view(x.size()[0]*x.size()[1], 18) + self.id
            #self.theta = self.u2u(u) + self.id
            
            '''make sure the input size to gridGen'''
            grid = self.gridGen(self.theta.view(x.size()[0]*x.size()[1], 18))
        
        p = F.grid_sample(p.view(x.size()[0]*x.size()[1],28,28).unsqueeze(1), grid, padding_mode='border')
        #p now is of size [k*batch_size, 1, 28,28]
        p = p.squeeze(1)
        p = p.view(x.size()[0],x.size()[1],self.dim_image_vars)
        #p.clamp_(1e-6, 1-1e-6)
        
        return (h1, mu_h1, sigma_h1, u, mu_u, sigma_u ), (p)

    def train_loss(self, inputs):
        h1, mu_h1, sigma_h1, u, mu_u, sigma_u = self.encoder(inputs)
        
        log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        log_QuGx =  torch.sum(-0.5*((u-mu_u)/sigma_u)**2 - torch.log(sigma_u), -1)
        '''
        log_posterior for q(h1|x) and q(u|x)
        both have size [k,batch_size]
        '''
        p = self.decoder(h1)
        ''' transform p here'''
        
        '''############################################################################'''
        self.id = self.idty.expand((inputs.size()[0]*inputs.size()[1],)+self.idty.size())
        self.id = self.id.to('cuda')
        self.id = self.id.double()
        if self.tf == 'aff':
            self.theta = self.u2u(u).view(inputs.size()[0]*inputs.size()[1], 2, 3) + self.id
            #self.theta = self.id
            grid = F.affine_grid(self.theta, p.view(inputs.size()[0]*inputs.size()[1],28,28).unsqueeze(1).size())
            #grid of size [k*batch_size,28,28,2]
       
        elif self.tf == 'tps':
            self.theta = self.u2u(u).view(inputs.size()[0]*inputs.size()[1], 18) + self.id
            
            '''make sure the input size to gridGen'''
            grid = self.gridGen(self.theta.view(inputs.size()[0]*inputs.size()[1], 18))
        
        p = F.grid_sample(p.view(inputs.size()[0]*inputs.size()[1],28,28).unsqueeze(1), grid, padding_mode='border')
        '''
        p now is of size [k*batch_size, 1, 28,28]
        '''
        p = p.squeeze(1)
        p = p.view(inputs.size()[0],inputs.size()[1],self.dim_image_vars)
        #p is now [k,batch_size, 784]
        
        '''########################################################################'''
        
        
        
        log_Ph1 = torch.sum(-0.5*h1**2, -1)
        log_Pu = torch.sum(-0.5*u**2, -1)
        '''
        log_prior for p(h1) and p(u)
        this has size [k,batch_size]
        '''
        
        #log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
        log_PxGh1 = -torch.sum(F.binary_cross_entropy(p, inputs, reduction='none'), -1)
        '''
        log likelihod for the decoder, which is a log likelihood over bernoulli
        this has size [k,batch_size]
        '''

        log_weight = log_Ph1 + log_Pu + log_PxGh1 - log_Qh1Gx - log_QuGx
        '''
        matrix of log(w_i)
        this has size [k,batch_size]
        '''
        
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        '''
        normalize to prevent overflow
        maximum w's for each batch element, where maximum is taking over k samples from posterior
        
        Note: For plian version of VAE, this is identically 0
        '''
        
        weight = torch.exp(log_weight)
        '''
        exponential the log back to get w_i's
        Note: For plian version of VAE, this is identically 1
        '''
        
        weight = weight / torch.sum(weight, 0)
        '''
        \tilda(w_i)
        Note: For plian version of VAE, this is identically 1
        '''
       
        weight = Variable(weight.data, requires_grad = False)
        '''
        stop gradient on \tilda(w)
        '''
        
        loss = -torch.mean(torch.sum(weight * (log_Ph1 + log_Pu + log_PxGh1 - log_Qh1Gx - log_QuGx), 0))
        return loss

    def test_loss(self, inputs):
        #h1, mu_h1, sigma_h1 = self.encoder(inputs)
        h1, mu_h1, sigma_h1, u, mu_u, sigma_u = self.encoder(inputs)
        log_QuGx =  torch.sum(-0.5*((u-mu_u)/sigma_u)**2 - torch.log(sigma_u), -1)
        log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)        
        p = self.decoder(h1)
        '''#################################################################################'''
        self.id = self.idty.expand((inputs.size()[0]*inputs.size()[1],)+self.idty.size())
        self.id = self.id.to('cuda')
        self.id = self.id.double()
        if self.tf == 'aff':
            self.theta = self.u2u(u).view(inputs.size()[0]*inputs.size()[1], 2, 3) + self.id
            #self.theta = self.u2u(u).view(inputs.size()[0]*inputs.size()[1], 2, 3)
            #self.theta = self.id
            grid = F.affine_grid(self.theta, p.view(inputs.size()[0]*inputs.size()[1],28,28).unsqueeze(1).size())
            #grid of size [k*batch_size,28,28,2]
        else:
            self.theta = self.u2u(u).view(inputs.size()[0]*inputs.size()[1], 18) + self.id
            
            '''make sure the input size to gridGen'''
            grid = self.gridGen(self.theta.view(inputs.size()[0]*inputs.size()[1], 18))
        
        pp = F.grid_sample(p.view(inputs.size()[0]*inputs.size()[1],28,28).unsqueeze(1), grid, padding_mode='border')
        '''
        p now is of size [k*batch_size, 1, 28,28]
        '''
        pp = pp.squeeze(1)
        p = pp.view(inputs.size()[0],inputs.size()[1],self.dim_image_vars)
        #p = p.clamp_(1e-6, 1-1e-6)

        '''#################################################################################'''
        log_Ph1 = torch.sum(-0.5*h1**2, -1)
        log_Pu = torch.sum(-0.5*u**2, -1)
        
        #log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
        log_PxGh1 = -torch.sum(F.binary_cross_entropy(p, inputs, reduction='none'), -1)
        log_weight = log_Ph1 + log_Pu + log_PxGh1 - log_Qh1Gx - log_QuGx
        weight = torch.exp(log_weight)
        loss = -torch.mean(torch.log(torch.mean(weight, 0)))        
        return loss

    
