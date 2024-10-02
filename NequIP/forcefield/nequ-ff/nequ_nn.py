import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace
from e3nn.util.test import assert_equivariant
from radial_basis import distance_embedding
from nequ_nn_conv import Convolution

class NequIP(nn.Module):
   __constants__ = ['atomic_list', 'embedding_dim','cutoff','lmax' ]
   atomic_list: list[int]
   embedding_dim: list[int]
   cutoff: float 
   lmax: int 
   num_elements:int 
   edge_embed_dim:int 


   def __init__(self, atomic_list, embedding_dim=[4, 10], hidden_dim=10, cutoff=5.0, 
                lmax = 2 , device=None, dtype=None):
      factory_kwargs = {'device': device, 'dtype': dtype}
      super().__init__()
      
      # Store the list of atomic numbers for all atoms in the system
      self.atomic_list = atomic_list
      
      # Store the cutoff  
      self.cutoff = cutoff

      # Count the number of unique elements
      self.num_elements = len(atomic_list) 
      
      # Create an embedding layer for atom types
      # This converts atomic numbers into dense vector representations
      self.type_embedding = nn.Embedding(self.num_elements, embedding_dim[0])  
      # optional: embed a atom pair, this is can garrentess the perm symmetry 

      # Create a radial basis function for encoding distances
      # This helps in representing interatomic distances in a more expressive way
      self.r_embedding = distance_embedding(cutoff, bmax=embedding_dim[1], p = 6)

      print(1111111111111111111111111)
      edge_embed_dim = 2*embedding_dim[0] + embedding_dim[1]
      self.edge_embed_dim = edge_embed_dim
      self.embedding_dim = embedding_dim
      # direct sum of irreps up to lmax 
      self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

      self.irreps_atom = o3.Irreps([(embedding_dim[0], (0,1))])
      #self.irreps_edge_bare = o3.Irreps([(edge_embed_dim, (0,1))])
      print(22222222222222222222222222)
      gate1 = Gate(
          self.irreps_atom+"8x0o",            # the first 18x0e are for the first embedding 
          [torch.relu, torch.abs],                           # scalar
          "4x0e + 4x0o + 4x0e + 4x0o",
          [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars)
          "4x1o + 4x1e + 4x2e + 4x2o",  # gated tensors, num_irreps has to match with gates
      )
      print('!!!!!!!!!!!!!!!!!')
      # first 
      irreps_atom_embed = o3.Irreps([ (embedding_dim[0], (0,1)) ]) # atomic embedding dimension \times 
      print(33333333333333333333333)
      self.conv1 = Convolution(irreps_atom_embed, self.irreps_sh, gate1.irreps_in, edge_embed_dim, hidden_dim)
      #self.conv1 = Convolution(irreps_atom_embed, self.irreps_sh, "1x0e+1x1o+1x1e", edge_embed_dim, hidden_dim)
      print(44444444444444444444444)
      self.gate1 = gate1
      irreps = self.gate1.irreps_out
      
      print("in out of the first conv = ", irreps_atom_embed, irreps )
      gate2 = Gate(
          self.irreps_atom+"8x0o",              # the first 18x0e are for the first embedding 
          [torch.relu, torch.abs],                           # scalar
          "4x0e + 4x0o + 4x0e + 4x0o",
          [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars), all gate are even function! 
          "4x1o + 4x1e + 4x2e + 4x2o",  # gated tensors, num_irreps has to match with gates
      )
      
      self.conv2 = Convolution(irreps, self.irreps_sh, gate2.irreps_in, edge_embed_dim, hidden_dim)
      self.gate2 = gate2
      irreps = self.gate2.irreps_out
      print("in out of the second conv = ", irreps, gate2.irreps_in )

      # Final layer, is the energy, energy is 0e
      self.final = Convolution(self.gate2.irreps_out, self.irreps_sh, "1x0e", edge_embed_dim, hidden_dim)
      self.irreps_out = self.final.irreps_out

   def forward(self, atom_types, positions):
      num_nodes = 4  # typical number of nodes

      # every only one graph is passed in 
      # Convert atom types to integer indices
      type_indices = torch.tensor([self.atomic_list.index(at) for at in atom_types], device=positions.device)
      
      # Use radius_graph to create edge index for pairs within cutoff
      edge_index = radius_graph(positions, r=self.cutoff, batch=None)
      
      # Compute distances for the selected edges
      edge_src, edge_dest = edge_index
      edge_vec = positions[edge_dest] - positions[edge_src]   # of size [num_edges, 3]
      edge_length = torch.norm(edge_vec, dim=-1).unsqueeze(-1) # of size [num_edges, 1]
      
      # Compute initial feature: atomic embeddings + distance embedding 
      type_embed = self.type_embedding(type_indices) # of size [num_atoms, embedding_dim[0]]
      edge_length_embedded = self.r_embedding(edge_length) # of size [num_edges, embedding_dim[1]]
      edge_embedded = torch.cat([ type_embed[edge_dest], type_embed[edge_src], edge_length_embedded], dim=-1)
     
      edge_sh = o3.spherical_harmonics(l=self.irreps_sh, x=edge_vec, normalize=True, normalization="component")

      x0 = type_embed
      x1 = self.conv1(x0, edge_src, edge_dest, edge_sh, edge_embedded)

      x1 = self.gate1(x1)
      # residual link, x1 = x1 + x0 
      x1[:, :self.embedding_dim[0]] =  x1[:, :self.embedding_dim[0]] + x0 

      x2 = self.conv2(x1, edge_src, edge_dest, edge_sh, edge_embedded)
      x2 = self.gate2(x2)
      # residual link, x2 = x2 + x1, x1 and x2 of the same size 
      x2 = x2 + x1 
      x3 = self.final(x2, edge_src, edge_dest, edge_sh, edge_embedded)

      return torch.sum(x3)
   
   
   def compute_forces(self, atom_types, positions):
      # Enable gradient computation for positions
      positions.requires_grad_(True)
        
      # Compute total energy
      total_energy = self.forward(atom_types, positions)
        
      # Compute forces as negative gradient of energy with respect to positions
      forces = -torch.autograd.grad(total_energy, positions, create_graph=True, retain_graph=True)[0]
      #  forces = 0 
      return total_energy, forces


if __name__ == "__main__":
    #check equvariance of the model 
    import torch 
    from e3nn import o3
    import numpy as np 
    device = 'cuda'
    na = 10
    atom_types = torch.randint(high=3, size=[na] ).to(device)
    positions = torch.rand(size=(na,3) ).to(device)
    atomic_list = list(np.unique(atom_types.cpu().numpy()))
    print(atomic_list)

    model = NequIP(atomic_list, cutoff=4.0, lmax = 2).to(device)
    p1 = positions + 1 
    #model.zero_grad()
    predicted_energy, predicted_forces = model.compute_forces(atom_types, positions)

    R_x = o3.rand_matrix().to(device)
    positions_D = torch.tensordot(positions, R_x,dims=[[-1],[-1]])
    predicted_energy_r, predicted_forces_r = model.compute_forces(atom_types, positions_D)
    predicted_forces_r = torch.tensordot(predicted_forces_r,torch.inverse(R_x),dims=[[-1],[-1]])
    print('energy of original system = ',predicted_energy)
    print('energy of rotated system = ',predicted_energy_r)
    #comparising the force : F(x) = R^-1 F(Rx) 
    print( '|R^{-1} Force(Rx) - Force(x)| = ',torch.max(torch.abs(predicted_forces_r - predicted_forces)).item() )

