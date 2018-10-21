import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append("../model/")
from IW_TVAE import *

from sys import exit

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


## read data
with open('data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']


batch_size = 100
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data, batch_size = batch_size,
                               shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)


#arguments

model = 'IWAE'
num_samples = 5  #which is k, number of samples from posterior
#############################################

z_dim = 42   #dim of z
u_dim = 18   #dim of u


vae = IWAE_1(z_dim, u_dim, 784,'tps')

    
vae.double()

vae.to(device)

    
#optimizer = optim.Adam(vae.parameters(), weight_decay = 0, amsgrad = True)

num_epoches = 50
optimizer = optim.Adadelta(vae.parameters())
l2 = lambda epoch: pow((1.-1.*epoch/num_epoches),0.9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l2)


train_loss_epoch = []
for epoch in range(num_epoches):
    scheduler.step()
    #optimizer = optim.Adam(vae.parameters(), lr = lr, weight_decay = 0, amsgrad = False, eps=1e-04)
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        data = data.double()
        inputs = Variable(data).to(device)
        if model == "IWAE":
            inputs = inputs.unsqueeze_(0)
            inputs = inputs.expand(num_samples, batch_size, 784)
        elif model == "VAE":
            inputs = inputs.repeat(num_samples, 1)
            inputs = inputs.unsqueeze_(0)
            inputs = inputs.expand(1, batch_size*num_samples, 784)
            
        optimizer.zero_grad()
        loss = vae.train_loss(inputs)
        loss.backward()
        optimizer.step()
        if (idx+1) % 100 == 0:
            print(("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}").format(epoch, idx, loss.item()), flush = True)
        running_loss.append(loss.item())        
    train_loss_epoch.append(np.mean(running_loss))

    if (epoch + 1) % 50 == 0:
        torch.save(vae.state_dict(),
                   ("{}_k_{}_epoch_{}_Transform.model")
                   .format(model, 
                           num_samples, epoch))
