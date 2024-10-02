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


# you can change the internel dimension 
class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, dim_edge_embedding, hidden_dim = 16) -> None:
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
        self.weight_mlp = nn.Sequential(nn.Linear(in_features=dim_edge_embedding, out_features=hidden_dim),
                                nn.SiLU(), 
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
        #print("edge_embedding.size = ",edge_embedding.size())
        weight = self.weight_mlp(edge_embedding)

        # Perform the tensor product operation
        # Combines node features of source nodes, edge attributes, and generated weights
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)

        # Aggregate the edge features to update node features
        # Use scatter operation to sum features for each destination node
        # Normalize by the square root of the number of neighbors
        #num_neighbors = len(edge_dst)/(len(node_features)+0.0)

        node_features = scatter(edge_features, edge_dst, dim=0)
        #node_features = scatter(edge_features, edge_dst, dim=0)
        
        return node_features
    