import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
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

class GCN(nn.Module):
    def __init__(self, num_features, num_nodes=32,hidden_size_1=256, hidden_size_2=512, hidden_size_3=128):
        super(GCN, self).__init__()

        self.num_nodes = num_nodes
        # input = nn.Parameter(torch.FloatTensor(num_features, 1))
        # Define the three graph convolutional layers
        self.conv1 = GCNConv(num_features, hidden_size_1)
        self.conv2 = GCNConv(hidden_size_1, hidden_size_2)
        self.conv3 = GCNConv(hidden_size_2, hidden_size_3)

        # Define Batch Normalization for each convolutional layer
        # self.bn1 = nn.BatchNorm1d(hidden_size_1)
        # self.bn2 = nn.BatchNorm1d(hidden_size_2)
        # self.bn3 = nn.BatchNorm1d(hidden_size_3)

        # Define three fully connected layers
        self.fc1 = nn.Linear(num_nodes*hidden_size_3, 1)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 1)

    def forward(self, x, edge_index):
        
        # With Batch Normalization
        # x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        # x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        # x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))

        # Without Batch Normalization
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))

        # print(f"Shape of x after Conv is {x.shape}")
        x = x.view(-1, self.num_nodes*x.shape[1] )
        # print(f"Reshaped x is {x.shape}")
        # x = torch.mean(x, dim=1)

        x = F.relu(self.fc1(x))

        # x = F.relu(self.fc2(x))

        # x = self.fc3(x)

        return x

if __name__ == "__main__":
    
    batch_size = 2

    # Loading the Model
    device = torch.device('cpu')
    model = GCN(1).to(device)
    learn_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of Learnable parameters: {learn_model_parameters}, Total Param: {total_params}")
    

    # Loading the dataset
    from .dataset import load_data
    data_loader, _ = load_data('training_data/data/training_data', batch_size=batch_size)
    for i, data in enumerate(data_loader):
        print(f"\nBatch {i} has {len(data)} data points")
        print(f"Data is {data}")
        x = data.x
        target = data.y
        edge_index = data.edge_index
        output = model(x, edge_index).squeeze(1)
        print(f"Shape of output is {output.shape} and target is {target.shape}")
        loss = F.mse_loss(output, data.y)
        print(f"Output is {output}")


        if i == 1: break
    

    # data_iter = iter(data_loader)
    # first_batch = next(data_iter)
    # print(f"First Batch size is {len(first_batch)}")
    # print(f"Number of Batches is {len(data_loader)}\n")



    # print(first_batch[0])
    # print(first_batch)
    # data = first_batch.to(device)
    # print(f"Data is {data}")
    # x = data.x
    # edge_index = data.edge_index
    # print(f"Data is {data}")
    # print(f"Data.x is {x}")
    # print(f"Data.edge_index is {edge_index}")
    # output = model(x, edge_index)
    # print(f"Shape of output is {output.shape}")
    # print(f"Shape of target is {data.y.shape}")
    # print(f"\nOutput of the model is {output}")
    # loss = F.mse_loss(output, data.y)
    # print(f"Loss is {loss}")
    
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

