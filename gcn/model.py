import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


"""
Architecture: 
    Tuple: (Type of Layer, Num of Output Channels)
"""

# torch.manual_seed(1)

ARCHITECTURE = [
    ('Conv',  32),
    ('Conv', 32),
    ('Conv', 32),
    ('Linear', 128), 
    ('Linear', 64), 
    ('Linear', 32), 
    ('Linear', 1), 
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(ConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, improved=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        conv = self.conv(x, edge_index)
        return self.batchnorm(self.leakyrelu(conv))
        # return self.leakyrelu(conv)

class FCN(nn.Module):
    def __init__(self, in_nodes, out_nodes, act=True):
        super(FCN, self).__init__()
        self.act = act
        self.linear = nn.Linear(in_nodes, out_nodes)
        if self.act:
            self.leakyrelu = nn.LeakyReLU(0.1)
            self.batchnorm = nn.BatchNorm1d(out_nodes)
        """Do I have to do BatchNorm here?"""

    def forward(self, x): 
        if self.act:
            # return self.batchnorm(self.leakyrelu(self.linear(x)))
            return self.leakyrelu(self.linear(x))
        else: 
            return self.linear(x)

class LatNet(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(LatNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size

        # hidden size is the embedding dim
        self.node_embedding = nn.Embedding(num_nodes, hidden_size)
        nn.init.normal_(self.node_embedding.weight, std=0.1)

        self.architecture = ARCHITECTURE
        self.layers = self._create_layers(self.architecture)

    def forward(self, data): 
        x = self.node_embedding.weight
        edge_index = data.edge_index
        # print(f"Time to go forward")
        for layer in self.layers:

            # print(f"Inside layer lol")
            if isinstance(layer, ConvBlock):
                print(f"\nNew Input X is {x.shape}")
                x = layer(x, edge_index)
                print(f"Output shape is {x.shape}")

            if isinstance(layer, FCN):
                x = layer(x.view(-1, layer.linear.in_features))

        return x

    def _create_layers(self, architecture):
        layers = []
        in_channels = self.hidden_size
        first_fcn_flag = True

        for idx, module in enumerate(architecture):
            block, out_channels = module

            if block == 'Conv':
                layers.append(ConvBlock(in_channels, out_channels))
                in_channels = out_channels

            if block == 'Linear':
                if first_fcn_flag:
                    in_channels = self.num_nodes*in_channels
                    first_fcn_flag = False

                if idx == len(architecture) - 1:
                    layers.append(FCN(in_channels, out_channels, act=False))
                else: 
                    layers.append(FCN(in_channels, out_channels))
                in_channels = out_channels

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = torch.device('cpu')
    model = LatNet(9, 16).to(device)
    learn_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of Learnable parameters: {learn_model_parameters}, Total Param: {total_params}")

    from .dataset import load_data
    data_loader, _ = load_data('training_data/data/training_data', batch_size=2)
    data_iter = iter(data_loader)
    first_batch = next(data_iter)

    # print(first_batch[0])
    # print(first_batch)
    # output = model(first_batch.to(device))
    # print(f"Output of the model is {output}")
    
    """
    Testing Conv for Batches
    """

    embedding_dim = 16
    conv_model = ConvBlock(embedding_dim,2)
    
    # For single data
    first_data = first_batch[0]
    first_data_edge_index = first_data.edge_index
    embedding = nn.Embedding(9, embedding_dim)
    nn.init.normal_(embedding.weight, std=0.1)
    
    print(f"\nSingle Data is {first_data}")
    print(f"Embedding shape {embedding.weight.shape}")
    conv_model(embedding.weight ,first_data_edge_index)

    # For Batch
    first_batch_edge_index = first_batch.edge_index
    embedding = nn.Embedding(18, embedding_dim)
    nn.init.normal_(embedding.weight, std=0.1)
    
    print(f"\nBatch Data is {first_batch}")
    print(f"Embedding shape {embedding.weight.shape}")
    conv_model(embedding.weight, first_batch_edge_index)


    # print(model)

