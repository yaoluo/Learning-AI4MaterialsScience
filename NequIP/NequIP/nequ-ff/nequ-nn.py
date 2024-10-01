import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

import sys 
sys.path.append("/Users/yaoluo/Literature/Research-idea/MachineLearn/Learning-AI4MaterialsScience-dev/NequIP/module")
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace
from e3nn.util.test import assert_equivariant
from radial_basis import distance_embedding


# you can change the internel dimension 
class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, dim_edge_embedding, hidden_dim = 256) -> None:
        """
        Initialize the Convolution module.

        Parameters:
        - irreps_in: o3.Irreps
            The irreducible representations of the input node features.

        - irreps_sh: o3.Irreps
            The irreducible representations of the spherical harmonics.

        - irreps_out: o3.Irreps
            The irreducible representations of the output features.

        - dim_edge_embedding: int 
            = 2 * dim of atomic embeding + distance embeding 
            The dimensionality of the edge embedding.
            This is the input size for the fully connected network that generates weights.
        """
        super().__init__()


 
        # Initialize a FullyConnectedTensorProduct for tensor operations
        # This will handle the combination of input features and spherical harmonics
        tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,    # Input irreducible representations
            irreps_in2=irreps_sh,    # Spherical harmonics irreps
            irreps_out=irreps_out,   # Output irreps
            internal_weights=False,  # Weights will be provided externally from a MLP
            shared_weights=False,    # Each output channel has its own weight
        )

        # Create a fully connected neural network to generate weights
        # This network takes edge embeddings and outputs weights for the tensor product
        self.weight_mlp = nn.Sequential(nn.Linear(in_features=dim_edge_embedding, out_features=256),
                                nn.silu(), 
                                nn.Linear(in_features=hidden_dim, out_features=tp.weight_numel))
        # Note: input dim is dim_edge_embeding, hidden layer has hidden_dim (=256 by default) units,
        # output dim is the number of weights needed for the tensor product

        self.tp = tp  # Store the tensor product object for use in forward pass
        self.irreps_out = self.tp.irreps_out  # Store the output irreps

    def forward(self, node_features, edge_src, edge_dst, edge_attr, edge_embedding) -> torch.Tensor:
        """
        Perform the convolution operation on the input graph.

        Parameters:
        - node_features: torch.Tensor
            Features of each node in the graph, is covariant, of size [# of nodes, F=dimension of node features]
        - edge_src: torch.Tensor
            Indices of the source nodes for each edge., of size [# of edges]
        - edge_dst: torch.Tensor
            Indices of the destination nodes for each edge., of size [# of edges]
        - edge_attr: torch.Tensor
            Attributes of each edge (e.g., spherical harmonics)., of size [# of edges, l_max^2]
        - edge_embedding: torch.Tensor
            Rotationally invariant embedding for each edge. for size [# of edges, embedding dimension]

        Returns:
        - torch.Tensor: Updated node features after convolution.
        """
        # Generate weights for the tensor product using the edge embeddings
        # weight_mlp maps edge_embedding to all the channels of (l_f,l_i,l_o) with multiplicity
        weight = self.weight_mlp(edge_embedding)

        # Perform the tensor product operation
        # Combines node features of source nodes, edge attributes, and generated weights
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)

        # Aggregate the edge features to update node features
        # Use scatter operation to sum features for each destination node
        # Normalize by the square root of the number of neighbors
        num_neighbors = len(edge_dst)/(len(node_features)+0.0)

        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)
        #node_features = scatter(edge_features, edge_dst, dim=0)
        
        return node_features
    



class NequIP(nn.Module):
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
      edge_embed_dim = 2*embedding_dim[0] + embedding_dim[1]
      self.edge_embed_dim = edge_embed_dim
      self.embedding_dim = embedding_dim
      # direct sum of irreps up to lmax 
      self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

      irreps = self.irreps_sh

      gate1 = Gate(
          [(edge_embed_dim, (0,1)),(8, (0, -1))],                                    # the first 18x0e are for the first embedding 
          [torch.silu, torch.abs],                           # scalar
          "4x0e + 4x0o + 4x0e + 4x0o",
          [torch.silu, torch.tanh, torch.silu, torch.tanh],  # gates (scalars)
          "4x1o + 4x1e + 4x2e + 4x2o",  # gated tensors, num_irreps has to match with gates
      )
      
      
      self.conv1 = Convolution(irreps, self.irreps_sh, gate1.irreps_in, edge_embed_dim, hidden_dim)
      self.gate1 = gate1
      irreps = self.gate1.irreps_out


      gate2 = Gate(
          "8x0e + 8x0o",
          [torch.silu, torch.abs],                           # scalar
          "4x0e + 4x0o + 4x0e + 4x0o",
          [torch.silu, torch.tanh, torch.silu, torch.tanh],  # gates (scalars)
          "4x1o + 4x1e + 4x2e + 4x2o",  # gated tensors, num_irreps has to match with gates
      )
      
      self.conv2 = Convolution(irreps, self.irreps_sh, gate2.irreps_in, edge_embed_dim, hidden_dim)
      self.gate2 = gate2
      irreps = self.gate2.irreps_out


      # Final layer, is the energy, energy is 0e
      self.final = Convolution(irreps, self.irreps_sh, "4x0e", edge_embed_dim, hidden_dim)
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
      edge_attr = o3.spherical_harmonics(l=self.irreps_sh, x=edge_vec, normalize=True, normalization="component")

      print('edge_embedded size = ',edge_embedded.size())
      # 
      x0 = type_embed
      x1 = self.conv1(x0, edge_src, edge_dest, edge_attr, edge_embedded)
      # residual link, x1 = x1 + x0 
      x1[:, :self.edge_embed_dim] =  x1[:, :self.edge_embed_dim] + x0 
      x1 = self.gate1(x1)
      x2 = self.conv2(x1, edge_src, edge_dest, edge_attr, edge_embedded)
      # residual link, x2 = x2 + x1, x1 and x2 of the same size  
      x2 = x2 + x1 
      x2 = self.gate2(x2)
      x3 = self.final(x2, edge_src, edge_dest, edge_attr, edge_embedded)

      return torch.sum(x3)
   
def compute_forces(self, atom_types, positions):
      # Enable gradient computation for positions
      positions.requires_grad_(True)
        
      # Compute total energy
      total_energy = self.forward(atom_types, positions)
        
      # Compute forces as negative gradient of energy with respect to positions
      forces = -torch.autograd.grad(total_energy, positions, create_graph=True, retain_graph=True)[0]
        
      return total_energy, forces






