import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, global_add_pool, HeteroConv, to_hetero

# torch.manual_seed(1)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleDict()
        self.num_layers = num_layers
        connection_types = [
            ('router', 'to', 'router'),
            ('router', 'to', 'pe'),
            ('pe', 'to', 'router'),
            ('task', 'to', 'pe'),
            ('start_task', 'to', 'task'),
            ('task', 'to', 'task'),
            ('task', 'to', 'end_task'),
            ('task', 'to', 'link'),
            ('start_task', 'to', 'link'),
            ('link', 'to', 'router'),
            ('link', 'to', 'task'),
            ('link', 'to', 'end_task'),
            ('end_task', 'to', 'pe'),
            ('start_task', 'to', 'pe'),
        ]

        for connection in connection_types:
            conv = HeteroConv({
                (connection): GraphConv(-1, hidden_channels)
            }, aggr='sum')

            self.convs[f'{connection[0]}_{connection[1]}_{connection[2]}'] = conv

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # print(f"Shape of x_dict is {x_dict['task'].shape}")
    
        for _ in range(self.num_layers):
            for connection in edge_index_dict.keys():
                print(f"Connection is {connection}")
                out_dict = self.convs[f'{connection[0]}_{connection[1]}_{connection[2]}'](x_dict, edge_index_dict)
                for key in out_dict.keys():
                    x_dict[key] = F.relu(out_dict[key])
    
        return x_dict

if __name__ == "__main__":

    # Loading the dataset
    from .dataset import load_data
    features = 1
    batch_size = 1

    data_loader, _ = load_data(
        'training_data/data/task_from_graph_dag_over_network_directed_hetero_train', batch_size=batch_size)

    metadata = next(iter(data_loader))[0].metadata()
    """metadata -> [[node_types], [edge_types]"""

    data = next(iter(data_loader))[0]
    # print(f"X dict is {data.x_dict}")
    print(f"Edge Index dict is {data.edge_index_dict}")
    for key, value in data.edge_index_dict.items():
        print(f"\nEdge type is {key}")
        print(value[0])
        print(value[1])

    model = HeteroGNN(hidden_channels=16, out_channels=1, num_layers=3)
    print(f"Model is {model}")

    output = model(data)
