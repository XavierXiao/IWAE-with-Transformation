import torch
from torch import nn, optim
from torchvision import transforms, datasets
from model import CTVAE
import torch.nn.functional as F
from torchvision.utils import save_image
from math import pow

import argparse

parser = argparse.ArgumentParser(
    description='Variational Autoencoder with Spatial Transformation'
)

parser.add_argument('--transformation', default='aff',
                    help='type of transformation: aff or tps')
parser.add_argument('--zdim', type=int, help='dimension of z')
parser.add_argument('--udim', type=int, help='dimension of u')
parser.add_argument('--nepoch', type=int, default=50, help='number of training epochs')
#parser.add_argument('--lamda',type=float,default=1,help='balancing parameter in front of classification loss')
parser.add_argument('--save', default='output/model.pt')
parser.add_argument('--gpu', default=True, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

args = parser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_gpu else "cpu")

'''Parameters, adjust here'''
mb_size = 128 # batch size

h = 28
w = 28
x_dim = h * w # image size

z_dim = 54
u_dim = 6


epochs = args.nepoch
h_dim = 256 # hidden dimension
log_interval = 100 # for reporting

'''########################################################################'''

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
# add 'download=True' when use it for the first time
mnist_tr = datasets.MNIST(root='../../MNIST/', download=True, transform=transforms.ToTensor())
mnist_te = datasets.MNIST(root='../../MNIST/', download=True, train=False, transform=transforms.ToTensor())
tr = torch.utils.data.DataLoader(dataset=mnist_tr,
                                batch_size=mb_size,
                                shuffle=True,
                                drop_last=True, **kwargs)
te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=1,
                                shuffle=True,
                                drop_last=True, **kwargs)


model = CTVAE(h, w, h_dim, z_dim, u_dim, mb_size, device, args.transformation).to(device).double()
parameters = filter(lambda p: p.requires_grad, model.parameters())

#optimizer and lr scheduling
optimizer = optim.Adadelta(model.parameters())
l2 = lambda epoch: pow((1.-1.*epoch/epochs),0.9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l2)



def loss_V(recon_x, x, mu1, var1, mu2, var2):
    '''loss = reconstruction loss + KL_z + KL_u'''
    BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1, x_dim), size_average=False)
    KLD1 = -0.5 * torch.sum(1 + 2*torch.log(var1) - mu1**2 - var1**2) # z
    KLD2 = -0.5 * torch.sum(1 + 2*torch.log(var2) - mu2**2 - var2**2) # u'

    return BCE, KLD1+KLD2


def NLL_test_loss(recon_x, x, mu1, var1,mu2,var2,z,u):
    #NLL just for testing 
    log_PxGh1 = -torch.sum(F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1,x_dim), reduction='none'), -1)
    log_Qh1Gx = torch.sum(-0.5*((z-mu1)/var1)**2 - torch.log(var1), -1)  
    log_QuGx =  torch.sum(-0.5*((u-mu2)/var2)**2 - torch.log(var2), -1)
    log_Ph1 = torch.sum(-0.5*z**2, -1)
    log_Pu = torch.sum(-0.5*u**2, -1)
    log_weight = log_Ph1 + log_Pu + log_PxGh1 - log_Qh1Gx - log_QuGx   
    weight = torch.exp(log_weight)
    NLL = -torch.log(torch.mean(weight,0))    
    return NLL


def test(epoch):
    model.eval()
    test_recon_loss = 0
    
    with torch.no_grad():
        for _, (data, target) in enumerate(te):
            data = data.to(device).double()
            #sample 1000 times from posterior to compute NLL (IWAE paper uses 5000)
            data = data.expand(1000,1,28,28)
            recon_batch, zmu, zvar, umu, uvar,z,u = model(data)
            NLL = NLL_test_loss(recon_batch, data, zmu, zvar, umu, uvar,z,u)
            test_recon_loss += NLL.item()
            
    test_recon_loss /= (len(te))
    print('====> Epoch:{} NLL: {:.4f}'.format(epoch, test_recon_loss))

def train(epoch):
    model.train()
    tr_recon_loss = 0
    c_loss = 0
    
    for batch_idx, (data, target) in enumerate(tr):
        data = data.to(device).double()
        optimizer.zero_grad()
    
        recon_batch, zmu, zlogvar, umu, ulogvar,z,u = model(data)
        recon_loss, kl = loss_V(recon_batch, data, zmu, zlogvar, umu, ulogvar)
        
        loss = recon_loss + kl 
        loss.backward()
        tr_recon_loss += recon_loss.item()
        
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tReconstruction-Loss: {:.4f} Classification Loss'.format(
                epoch, batch_idx * len(data), len(mnist_tr),
                100. * batch_idx / len(tr),
                recon_loss / len(data),
                c_loss / len(data)))

    print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(
          epoch, tr_recon_loss / (len(tr)*mb_size)))

for epoch in range(epochs):
    scheduler.step()
    train(epoch)
    if epoch==49:
        test(epoch)
