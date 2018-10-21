import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tps import TPSGridGen
from torch.autograd import Variable

class CTVAE(nn.Module):

    def __init__(self, x_h, x_w, h_dim, z_dim, u_dim, mb_size, device, t='aff'):
        super(CTVAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = h_dim # hidden layer
        self.z_dim = z_dim # generic latent variable
        self.u1_dim = u_dim # dimension of u'
        self.bsz = mb_size

        self.dv = device

        """
        encoder: two fc layers
        """
        self.x2h = nn.Sequential(
            nn.Linear(self.x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
            )

        self.h2zmu = nn.Linear(h_dim, z_dim)
        self.h2zvar = nn.Linear(h_dim, z_dim)
        self.tf = t
        if t == 'aff':
            self.u2_dim = 6 # u = Wu' + b
            self.idty = torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1)
        elif t == 'tps':
            self.u2_dim = 18 # 2 * 3^2=18
            self.gridGen = TPSGridGen(out_h=self.h,out_w=self.w,device=self.dv)
            px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            self.idty = torch.cat((px,py))
        else:
            raise ValueError( """An invalid option for transformation type was supplied, options are ['aff' or 'tps']""")
        
        
            
        self.h2umu = nn.Linear(h_dim, self.u1_dim)
        self.h2uvar = nn.Linear(h_dim, self.u1_dim)

        self.u2u = nn.Linear(self.u1_dim,self.u2_dim)
        self.z2h = nn.Linear(z_dim, h_dim)

        """
        decoder: two fc layers
        """

        self.z2x = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.x_dim),
            nn.Sigmoid()
        )


    def sample_z(self, mu, var):
        #eps = torch.randn(self.bsz, self.z_dim).to(self.dv)
        eps = Variable(mu.data.new(mu.size()).normal_()).to(self.dv).double()
        return mu + var * eps

    def sample_u(self, mu, var):
        #eps = torch.randn(self.bsz, self.u1_dim).to(self.dv)
        eps = Variable(mu.data.new(mu.size()).normal_()).to(self.dv).double()
        return mu + var * eps


    def forward(self, inputs):
        h = self.x2h(inputs.view(-1, self.x_dim))
        z_mu = self.h2zmu(h)
        z_logvar = F.threshold(self.h2zvar(h), -6, -6)
        z_var = torch.exp(z_logvar)
        u_mu = torch.tanh(self.h2umu(h))
        u_logvar = F.threshold(self.h2uvar(h), -6, -6)
        u_var = torch.exp(u_logvar)
        
        z = self.sample_z(z_mu, z_var)
        x = self.z2x(z)
        
        u = self.sample_u(u_mu, u_var)
        
        self.id = self.idty.expand((inputs.size()[0],)+self.idty.size()).to(self.dv).double()
        
        if self.tf == 'aff':
            self.theta = self.u2u(u).view(-1, 2, 3) + self.id
            #self.theta = self.id
            grid = F.affine_grid(self.theta, x.view(-1,self.h,self.w).unsqueeze(1).size())
        else:
            self.theta = self.u2u(u).view(-1,18) + self.id
            #self.theta = self.id
            grid = self.gridGen(self.theta)
            
        x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='border')
        x.clamp_(1e-6, 1-1e-6)

        return x, z_mu, z_var, u_mu, u_var,z,u

