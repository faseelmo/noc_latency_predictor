import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

"""
Architecture: 
    Tuple: (Type of Layer, Num of Output Channels)
"""

torch.manual_seed(1)

ARCHITECTURE = [
    ('Conv', 2),
    ('Conv', 3),
    ('Linear', 4), 
    ('Linear', 1), 
]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(ConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        conv = self.conv(x, edge_index, edge_attr)
        return self.batchnorm(self.leakyrelu(conv))

class FCN(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(FCN, self).__init__()
        self.linear = nn.Linear(in_nodes, out_nodes)
        self.leakyrelu = nn.LeakyReLU(0.1)
        """Do I have to do BatchNorm here?"""

    def forward(self, x): 
        return self.leakyrelu(self.linear(x))

class LatNet(nn.Module):
    def __init__(self, in_features, num_nodes):
        super(LatNet, self).__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.architecture = ARCHITECTURE
        self.layers = self._create_layers(self.architecture)

    def forward(self, data): 
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for layer in self.layers:
            if isinstance(layer, ConvBlock):
                x = layer(x, edge_index, edge_attr)

            if isinstance(layer, FCN):
                x = layer(x.view(-1))

        return x

    def _create_layers(self, architecture):
        layers = []
        in_channels = self.in_features
        first_fcn_flag = True

        for module in architecture:
            block, out_channels = module

            if block == 'Conv':
                layers.append(ConvBlock(in_channels, out_channels))
                in_channels = out_channels

            if block == 'Linear':
                if first_fcn_flag:
                    in_channels = self.num_nodes*in_channels
                    first_fcn_flag = False

                layers.append(FCN(in_channels, out_channels))
                in_channels = out_channels

        return nn.Sequential(*layers)


def test(): 
    print("\n---- Testing GCN Conv----")
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    
    x = torch.tensor([[-1, 2, 3, 1, 1, 1, 1, 1], [-1, 2, 3, 1, 1, 1, 1, 1], [-1, 2, 3, 1, 1, 1, 1, 1]], dtype=torch.float)

    """Edge Attr should only contain +ve values"""
    edge_attr = torch.randn((edge_index.size(1), 1), dtype=torch.float)

    test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    test_conv_block = ConvBlock(8, 5)
    output_conv_block = test_conv_block(
        test_data.x, test_data.edge_index, test_data.edge_attr
    )

    print(f"Conv Block Output is {output_conv_block}")

    print("\n---- Testing FCN----")
    test_fcn = FCN(100,10)
    test_input = torch.randn(100)
    print(f"Test Output is {test_fcn(test_input)}")


    print("\n---- Testing the final Model----")
    latency_estimator = LatNet(8, 3)
    print(latency_estimator(test_data))

    print(f"\n\n{latency_estimator}")


test()


