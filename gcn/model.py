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
    def __init__(self, num_features, hidden_size_1=256, hidden_size_2=512, hidden_size_3=256):
        super(GCN, self).__init__()

        # Define the three graph convolutional layers
        self.conv1 = GCNConv(num_features, hidden_size_1)
        self.conv2 = GCNConv(hidden_size_1, hidden_size_2)
        self.conv3 = GCNConv(hidden_size_2, hidden_size_3)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(hidden_size_3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer for regression

    def forward(self, data):
        # Apply the first graph convolutional layer
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        
        # Apply the second graph convolutional layer
        x = F.relu(self.conv2(x, edge_index))

        # Apply the third graph convolutional layer
        x = F.relu(self.conv3(x, edge_index))

        # Global pooling (e.g., mean) to aggregate node features
        x = torch.mean(x, dim=0, keepdim=True)  # Keep the dimension for batch size

        # Fully connected layers for regression
        x = F.relu(self.fc1(x.view(1, -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    
    from .dataset import load_data
    batch_size = 1
    data_loader, _ = load_data('training_data/data/training_data', batch_size=batch_size)
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    print(f"First Batch size is {len(first_batch)}")
    print(f"Number of Batches is {len(data_loader)}\n")

    device = torch.device('cpu')
    model = GCN(1).to(device)
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

