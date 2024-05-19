import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, global_add_pool

# torch.manual_seed(1)
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=512, num_node_features=1):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index)) 
        x2 = F.relu(self.conv2(x1, edge_index)) + x1
        x3 = self.conv3(x2, edge_index) + x2
        x4 = self.conv4(x3, edge_index) + x3
        return x4

class LatencyModel(torch.nn.Module):
    def __init__(self, hidden_channels=512, num_node_features=1):
        super(LatencyModel, self).__init__()
        self.gcn = GCN(hidden_channels, num_node_features)
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        gcn_output = self.gcn(x, edge_index)
        gcn_output = global_add_pool(gcn_output, batch)
        x = F.relu(self.lin1(gcn_output))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x

if __name__ == "__main__":
    
    # Loading the dataset
    from .dataset import load_data
    features = 6
    batch_size = 1
    data_loader, _ = load_data('training_data/data/task_from_graph_dag_over_network_corrected_train', batch_size=batch_size)

    # Loading the Model
    device = torch.device('cpu')
    model = LatencyModel(num_node_features=features).to(device)
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
    
