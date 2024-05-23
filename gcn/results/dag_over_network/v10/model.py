import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, global_add_pool, HeteroConv, to_hetero

# torch.manual_seed(1)


class GNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        # Try Graph Attention Network
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)  
        self.conv5 = GraphConv(hidden_channels, hidden_channels)  

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu() + x1  
        x3 = self.conv3(x2, edge_index).relu() + x2  
        x4 = self.conv4(x3, edge_index).relu() + x3  
        x5 = self.conv5(x4, edge_index) + x4  
        return x5


class LatencyModel(nn.Module):
    def __init__(self, num_features, hidden_channels, metadata, aggr='sum'):
        super(LatencyModel, self).__init__()
        gnn = GNN(num_features, hidden_channels)
        self.hetero_gnn = to_hetero(gnn, metadata, aggr=aggr)
        
        self.lin1 = nn.Linear(hidden_channels, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.lin2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.lin3 = nn.Linear(32, 1)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        gnn_output = self.hetero_gnn(x_dict, edge_index_dict)
        x = global_add_pool(gnn_output['link'], batch_dict)
        
        x = F.leaky_relu(self.bn1(self.lin1(x)))
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.lin2(x)))
        x = self.dropout2(x)
        
        x = self.lin3(x)
        return x


if __name__ == "__main__":

    # Loading the dataset
    from .dataset import load_data
    import torch_geometric.transforms as T

    FEATURES = 1
    BATCH_SIZE = 2
    HIDDEN_CHANNELS = 100

    data_loader, _ = load_data(
        'training_data/data/task_from_graph_dag_over_network_directed_hetero_train', batch_size=BATCH_SIZE)

    for i, data in enumerate(data_loader):
        print(
            f"\n\t\t\t\t\t--------Batch {i} has {len(data)} data points--------\n")

        undirected_data = T.ToUndirected()(data)

        print(f"Batch is {undirected_data.batch_dict['link']}")

        # """Homogenous"""
        # print(f"---- Homogenous Output ----\n ")
        # homogenous_data = undirected_data.to_homogeneous()
        # print(
        #     f"\nData is {homogenous_data}"
        #     f" with num of feature {homogenous_data.num_features}")

        # model = GNN(num_features=FEATURES, hidden_channels=HIDDEN_CHANNELS)
        # output = model(homogenous_data.x, homogenous_data.edge_index)

        # print(f"\nModel is {model}")
        # print(f"Output shape is {output.shape}")

        """Heterogenous"""
        print(f"\n")
        metadata = undirected_data.metadata() # metadata -> [[node_types], [edge_types]
        # model_hetero = to_hetero(model, metadata)
        model_hetero = LatencyModel(
            num_features=FEATURES, hidden_channels=HIDDEN_CHANNELS, metadata=metadata, aggr='sum')

        output_hetero = model_hetero(
            undirected_data.x_dict, undirected_data.edge_index_dict, undirected_data.batch_dict['link'])

        print(f"Output is {output_hetero} w/ shape is {output_hetero.shape}")

        # print(f"---- Hetero Output ----\n ")
        # for key, value in output_hetero.items():
        #     print(f"{key} is {value.shape}")

        break
