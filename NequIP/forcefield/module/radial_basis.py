from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F 
from torch import Tensor 
import torch 

class distance_embedding(torch.nn.Module):
    r"""
    r embedding : map r to a set of radial basis functions
    """
    __constants__ = ['bmax', 'p', 'rc']
    bmax: int
    p: int
    rc: float

    def __init__(self, rc: float, bmax: int, p : int, device=None, dtype=None) -> None:
         factory_kwargs = {'device': device, 'dtype': dtype}
         super().__init__()
         #self.device = device
         self.rc = rc 
         self.bmax = bmax
         self.p = p
         #self.eb = torch.zeros([nbatch, self.bmax]) 
         #self.to(device)  # Move the entire module to the specified device

    def forward(self, r: Tensor) -> Tensor:
          #input is of size [num_edges, 1]
          #output is of size [num_edges, bmax]
          p = self.p 
          rc = self.rc 

          bs = torch.arange(1, self.bmax + 1, device=r.device, dtype=torch.float32, requires_grad=False)
          r_mesh, b_mesh = torch.meshgrid(r[:,0], bs, indexing='ij')
          w = 1.0  - (p+1)*(p+2)/2.0 * torch.pow(r_mesh/rc, p) \
                    + p*(p+2) * torch.pow(r_mesh/rc, p+1) \
                    - p*(p+1)/2.0 * torch.pow(r_mesh/rc, p+2)
                    
          eb = 2.0/rc * torch.sin( b_mesh*torch.pi * r_mesh/rc ) * w 
          return eb 


class radial_nn(torch.nn.Module):
    r"""
    """
    __constants__ = ['bmax', 'p', 'rc', 'nlayer']
    bmax: int
    p: int
    r: float
    nlayer: int 
    rc: float 

    def __init__(self, rc: float, bmax: int, p: int, nlayer : int, device=None, dtype=None) -> None:
         factory_kwargs = {'device': device, 'dtype': dtype}
         super().__init__()
         self.rc =rc 
         self.bmax = bmax
         self.p = p
         self.nlayer = nlayer 
         self.r_embedding = distance_embedding(rc, bmax, p, device, dtype)
         self.weight = []
         self.bais = [] 
         self.silu = torch.nn.SiLU()
         self.layers = torch.nn.ModuleList() 
         for i in range(nlayer):
            if i==nlayer-1:
               self.layers.append(torch.nn.Linear(in_features=bmax, out_features=1, bias=False))
            else:
               self.layers.append(torch.nn.Linear(in_features=bmax, out_features=bmax, bias=False))
         self.to(device)  # Move the entire model to the specified device
   
    def forward(self, r: Tensor) -> Tensor:
          #input is of size [*, 1]
          #print(r.device, self.r_embedding(r).device)
          fr = self.layers[0](self.r_embedding(r))
          if self.nlayer>=2 :
               for i in range(1, self.nlayer):
                    fr =  self.layers[i](self.silu(fr))
          return fr



if __name__ == '__main__':
   import numpy as np 
   import matplotlib.pyplot as plt 
   import time 

   t1 = time.time()
   # hyper parameters 
   rc = 4.0       # cutoff radius 
   b = 10         # number of radial basis function 
   p = 6          # window function power 
   nlayer = 2     # number of layers 
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   device = 'cpu'
   model = radial_nn(rc, b, p, nlayer, device=device)
   
   x = torch.linspace(1e-4, rc, 2000).reshape(2000,1).to(device)  # Move x to the same device as the model
   
   y = model(x)  # Use model(x) instead of model.forward(x)
   #print(y.size())

   #using this raidal nn to fit an x**2 * exp(-x)
   y_target = x**2 * torch.exp(-x*4) * 25 * torch.heaviside( rc - x, torch.zeros_like(x))
   
   plt.plot(x.cpu().detach(), y_target.cpu().detach(),lw=2, label='ground truth' )
   plt.plot(x.cpu().detach(), y.cpu().detach(),lw=2,label='before training')

   #SGD 
   loss_fn = torch.nn.MSELoss(reduction='sum')
   optimier = torch.optim.SGD(model.parameters(), lr=1e-3)

   dps = 10 
   loss_old = 1e10
   t = 0 
   for t in range(1000):
        y_pred = model.forward(x) 
        loss = loss_fn(y_pred, y_target)
        t += 1
        if t%100 == 99:
             loss_new = loss.item() + 0.0
             print(t, loss_new)
             dps = abs(loss_new - loss_old) 
             loss_old = loss_new
        optimier.zero_grad()
        loss.backward()
        optimier.step()

   t2 = time.time()
   print('time = ',t2-t1, 'sec')
   plt.plot(x.cpu().detach(), y_pred.cpu().detach(),lw=2,label='after training')
   plt.legend()
   plt.ylim([-0.5,1])
   plt.xlabel('x', fontsize=15)
   plt.ylabel('f(x)', fontsize=15)
  
   
   plt.show()
   plt.savefig('learn-one-function.jpg', dpi=500)