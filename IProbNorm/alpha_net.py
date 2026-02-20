import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaNet(nn.Module):
    """
    AlphaNet computes attention weights between keys and a query using scaled dot-product attention.

    This module projects keys and queries into a shared embedding space using separate
    learnable weight matrices, then computes attention scores via scaled dot-product.
    Inputs are L2-normalized before projection.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features for both keys and query.
    weight_dim : int
        Dimensionality of the projected embedding space for attention computation.

    Attributes
    ----------
    W_q : nn.Parameter
        Learnable weight matrix for query projection, shape (input_dim, weight_dim).
    W_k : nn.Parameter
        Learnable weight matrix for keys projection, shape (input_dim, weight_dim).

    Forward Inputs
    --------------
    keys : torch.Tensor
        Tensor of shape (n_keys, input_dim) containing key vectors.
    query : torch.Tensor
        Tensor of shape (n_query, input_dim) containing the query vector.

    Returns
    -------
    torch.Tensor
        Attention weights of shape (n_query, n_keys) after softmax normalization.
    """
    def __init__(self, input_dim, weight_dim):
        super(AlphaNet, self).__init__()

        attn_logits = torch.randn(input_dim, weight_dim, dtype=torch.float32, requires_grad=True)  # "сырые" веса
        self.W_q = nn.Parameter(torch.softmax(attn_logits, dim=-1)) #nn.Parameter(torch.eye(input_dim)*100)#
        attn_logits = torch.randn(input_dim, weight_dim, dtype=torch.float32, requires_grad=True)  # "сырые" веса
        self.W_k = nn.Parameter(torch.softmax(attn_logits, dim=-1)) #nn.Parameter(torch.eye(input_dim)*100) #torch.softmax(attn_logits, dim=-1))

        torch.nn.init.xavier_uniform_(self.W_q)
        torch.nn.init.xavier_uniform_(self.W_k)

    def forward(self, keys, query):
        keys = torch.nn.functional.normalize(keys, dim=1)
        query = torch.nn.functional.normalize(query, dim=1)

        keys_proj = torch.matmul(keys, self.W_k)
        query_proj = torch.matmul(query, self.W_q)

        weights = torch.matmul(query_proj, keys_proj.T) / (keys_proj.shape[-1] ** 0.5)

        if len(weights.shape) == 1:
            weights = weights.unsqueeze(dim=1)

        weights = F.softmax(weights, dim=1)

        return weights