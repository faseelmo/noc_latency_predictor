import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, global_add_pool

# torch.manual_seed(1)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=512, num_node_features=1):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        # print(f"\nInput Size is {x.shape}")
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # print(f"1st GCN Output Size is {x.shape}")

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # print(f"2nd GCN Output Size is {x.shape}")

        x = self.conv3(x, edge_index)
        # print(f"3rd GCN Output Size is {x.shape}")

        # 2. Readout layer
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        # print(f"Global Mean Pooling Output Size is {x.shape}")

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin2(x)
        x = F.relu(x)

        x = self.lin3(x)
        x = F.relu(x)

        x = self.lin4(x)
        
        return x

if __name__ == "__main__":
    
    # Loading the dataset
    from .dataset import load_data
    features = 3
    batch_size = 1
    data_loader, _ = load_data('training_data/data/task_from_graph_tensor', batch_size=batch_size)

    # Loading the Model
    device = torch.device('cpu')
    model = GCN(num_node_features=features).to(device)
    learn_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of Learnable parameters: {learn_model_parameters}, Total Param: {total_params}")

    print(f"\nModel is {model}")  

    # Forward Pass
    for i, data in enumerate(data_loader):
        print(f"\n--------Batch {i} has {len(data)} data points--------")
        print(f"Data is {data}")
        # print(f"Data batch is {data.batch}")
        output = model(data.x, data.edge_index, data.batch).squeeze(1)
        # print(f"Shape of output is {output.shape} and target is {data.y.shape}")
        # loss = F.mse_loss(output, data.y)
        # print(f"Output is {output}")
        # print(f"-----------------------------------")

        if i == 1: break
    
