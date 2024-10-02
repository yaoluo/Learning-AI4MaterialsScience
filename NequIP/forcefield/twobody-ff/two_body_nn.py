import sys 
sys.path.append("../module/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_cluster import radius_graph
from radial_basis import distance_embedding
import numpy as np

# Define a two-body neural network model for predicting the energy and forces of a system of atoms
# This model is a graph neural network that can handle multiple types of atoms and multiple configurations

class TwoBodyNN(nn.Module):
   def __init__(self, atomic_list, embedding_dim=[8, 10], hidden_dim=10, cutoff=5.0, device=None, dtype=None):
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
      self.r_embedding = distance_embedding(cutoff, bmax=embedding_dim[1], p = 6, device=device, dtype=dtype)

      self.mlp = nn.Sequential(
          nn.Linear(2*embedding_dim[0]+embedding_dim[1], hidden_dim),
          nn.SiLU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.SiLU(),
          nn.Linear(hidden_dim, 1)
      )
      return
   def forward(self, atom_types, positions):
      # every only one graph is passed in 
      # Convert atom types to integer indices
      type_indices = torch.tensor([self.atomic_list.index(at) for at in atom_types], device=positions.device)
      
      # Use radius_graph to create edge index for pairs within cutoff
      edge_index = radius_graph(positions, r=self.cutoff, batch=None)
      
      # Compute distances for the selected edges
      edge_src, edge_dest = edge_index
      edge_vec = positions[edge_dest] - positions[edge_src]   # of size [num_edges, 3]
      edge_length = torch.norm(edge_vec, dim=-1).unsqueeze(-1) # of size [num_edges, 1]
      
      # Compute embeddings
      type_embed = self.type_embedding(type_indices) # of size [num_atoms, embedding_dim[0]]
      dist_embed = self.r_embedding(edge_length) # of size [num_edges, embedding_dim[1]]
      
      # Combine embeddings for each edge
      # type_embed[row] of size [num_edges, ]
      edge_features = torch.cat([ type_embed[edge_dest], type_embed[edge_src], dist_embed], dim=-1)  # of size [num_edges, 2*embedding_dim[0]+embedding_dim[1]]
      
      # Compute pair energies
      pair_energies = self.mlp(edge_features).squeeze(-1) # of size [num_edges, 1]
      
      # Sum pair energies to get total energy
      total_energy =0.5 * torch.sum(pair_energies,0)  # Factor 0.5 to avoid double counting
      
      return total_energy 
   
   def compute_forces(self, atom_types, positions):
      # Enable gradient computation for positions
      positions.requires_grad_(True)
        
      # Compute total energy
      total_energy = self.forward(atom_types, positions)
        
      # Compute forces as negative gradient of energy with respect to positions
      forces = -torch.autograd.grad(total_energy, positions, create_graph=True, retain_graph=True)[0]
        
      return total_energy, forces

