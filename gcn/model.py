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
    ('Conv',  128),
    ('Conv', 256),
    ('Conv', 128),
    # ('Linear', 512), 
    ('Linear', 256), 
    ('Linear', 128), 
    ('Linear', 64), 
    ('Linear', 32), 
    ('Linear', 1), 
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(ConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        # self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        conv = self.conv(x, edge_index)
        # return self.batchnorm(self.leakyrelu(conv))
        return self.leakyrelu(conv)

class FCN(nn.Module):
    def __init__(self, in_nodes, out_nodes, act=True):
        super(FCN, self).__init__()
        self.act = act
        self.linear = nn.Linear(in_nodes, out_nodes)
        if self.act:
            self.leakyrelu = nn.LeakyReLU(0.1)
            # self.batchnorm = nn.BatchNorm1d(out_nodes)
        """Do I have to do BatchNorm here?"""

    def forward(self, x): 
        if self.act:
            # return self.batchnorm(self.leakyrelu(self.linear(x)))
            return self.leakyrelu(self.linear(x))
        else: 
            return self.linear(x)

class LatNet(nn.Module):
    def __init__(self, num_nodes, in_features):
        super(LatNet, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features

        self.architecture = ARCHITECTURE
        self.layers = self._create_layers(self.architecture)

    def forward(self, data): 
        x, edge_index = data.x, data.edge_index
        # print("Input size:", x.size())  # Add this line to print the size of x
        for layer in self.layers:

            if isinstance(layer, ConvBlock):
                x = layer(x, edge_index)
                # print("ConvBlock output is :", x.size())

            if isinstance(layer, FCN):
                x = layer(x.view(-1, layer.linear.in_features))

        return x

    def _create_layers(self, architecture):
        layers = []
        in_channels = self.in_features
        first_fcn_flag = True

        for idx, module in enumerate(architecture):
            block, out_channels = module

            if block == 'Conv':
                layers.append(ConvBlock(in_channels, out_channels))
                in_channels = out_channels

            if block == 'Linear':
                if first_fcn_flag:
                    # in_channels bathth
                    in_channels = self.num_nodes*in_channels
                    # print(f"Number of neurons in the first linear layer: {in_channels}")
                    first_fcn_flag = False

                if idx == len(architecture) - 1:
                    layers.append(FCN(in_channels, out_channels, act=False))
                else: 
                    layers.append(FCN(in_channels, out_channels))
                in_channels = out_channels

        return nn.Sequential(*layers)


if __name__ == "__main__":
    
    from .dataset import load_data
    batch_size = 10
    data_loader, _ = load_data('training_data/data/training_data', batch_size=batch_size)
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    print(f"First Batch size is {len(first_batch)}")
    print(f"Number of Batches is {len(data_loader)}\n")

    device = torch.device('cpu')
    model = LatNet(32, 1).to(device)
    learn_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of Learnable parameters: {learn_model_parameters}, Total Param: {total_params}")


    # print(first_batch[0])
    # print(first_batch)
    output = model(first_batch.to(device))
    print(f"\nOutput of the model is {output}")
    
    """
    Testing Conv for Batches
    """

    # embedding_dim = 16
    # conv_model = ConvBlock(embedding_dim,2)
    
    # # For single data
    # first_data = first_batch[0]
    # first_data_edge_index = first_data.edge_index
    # embedding = nn.Embedding(9, embedding_dim)
    # nn.init.normal_(embedding.weight, std=0.1)
    
    # print(f"\nSingle Data is {first_data}")
    # print(f"Embedding shape {embedding.weight.shape}")
    # conv_model(embedding.weight ,first_data_edge_index)

    # # For Batch
    # first_batch_edge_index = first_batch.edge_index
    # embedding = nn.Embedding(18, embedding_dim)
    # nn.init.normal_(embedding.weight, std=0.1)
    
    # print(f"\nBatch Data is {first_batch}")
    # print(f"Embedding shape {embedding.weight.shape}")
    # conv_model(embedding.weight, first_batch_edge_index)


    # print(model)

