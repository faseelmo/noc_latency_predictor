import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, global_add_pool, HeteroConv, to_hetero

# torch.manual_seed(1)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero, global_add_pool
from torch.nn.utils.rnn import pad_sequence


class GNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.lin1 = nn.Linear(num_features, hidden_channels)

        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index) + self.lin1(x) 
        x1 = x1.relu()

        x2 = self.conv2(x1, edge_index) + self.lin2(x1)
        x2 = x2.relu()

        x3 = self.conv3(x2, edge_index) + self.lin3(x2)
        return x3


class LatencyModel(nn.Module):
    def __init__(self, num_features, hidden_channels, metadata, aggr='sum'):
        super(LatencyModel, self).__init__()
        gnn = GNN(num_features, hidden_channels)
        self.hetero_gnn = to_hetero(gnn, metadata, aggr=aggr)

        # Linear layers to sum channels to a single value for each node type
        self.channel_sum_link = nn.Linear(hidden_channels, 1)
        self.channel_sum_start_task = nn.Linear(hidden_channels, 1)
        self.channel_sum_end_task = nn.Linear(hidden_channels, 1)
        self.channel_sum_task = nn.Linear(hidden_channels, 1)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=1, hidden_size=64,
                            num_layers=1, batch_first=True)

        # Final linear layer
        self.lin = nn.Linear(64, 1)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        gnn_output = self.hetero_gnn(x_dict, edge_index_dict)

        # Sum the channels of each specified node type
        link_output = gnn_output['link']
        summed_channels_link = self.channel_sum_link(link_output)
        batch_dict_link = batch_dict['link']
        nodes_per_batch_link = torch.bincount(batch_dict_link)
        # print(f"\nLink")
        # print(
        #     f"Shape {summed_channels_link.shape}, nodes_per_batch {nodes_per_batch_link}")

        start_task_output = gnn_output['start_task']
        summed_channels_start_task = self.channel_sum_start_task(
            start_task_output)
        batch_dict_start_task = batch_dict['start_task']
        nodes_per_start_task = torch.bincount(batch_dict_start_task)
        # print(f"\nStart Task")
        # print(
        #     f"Shape {summed_channels_start_task.shape}, nodes_per_batch {nodes_per_start_task}")

        end_task_output = gnn_output['end_task']
        summed_channels_end_task = self.channel_sum_end_task(end_task_output)
        batch_dict_end_task = batch_dict['end_task']
        nodes_per_end_task = torch.bincount(batch_dict_end_task)
        # print(f"\nEnd Task")
        # print(
        #     f"Shape {summed_channels_end_task.shape}, nodes_per_batch {nodes_per_end_task}")

        task_output = gnn_output['task']
        summed_channels_task = self.channel_sum_task(task_output)
        batch_dict_task = batch_dict['task']
        nodes_per_task = torch.bincount(batch_dict_task)
        # print(f"\nTask")
        # print(
        #     f"Shape {summed_channels_task.shape}, nodes_per_batch {nodes_per_task}")

        lstm_input = []
        link_idx, start_task_idx, end_task_idx, task_idx = 0, 0, 0, 0
        for link, start_task, end_task, task in zip(nodes_per_batch_link, nodes_per_start_task, nodes_per_end_task, nodes_per_task):
            # print(
            #     f"Link: {link}, Start Task: {start_task}, End Task: {end_task}, Task: {task}")
            batch_input = torch.cat([
                summed_channels_start_task[start_task_idx:start_task_idx + start_task],
                summed_channels_task[task_idx:task_idx + task],
                summed_channels_link[link_idx:link_idx + link],
                summed_channels_end_task[end_task_idx:end_task_idx + end_task]
            ], dim=0)
            lstm_input.append(batch_input)

            link_idx += link
            start_task_idx += start_task
            end_task_idx += end_task
            task_idx += task

            
        # Ensure lstm_input has a batch dimension
        padded_lstm_input = pad_sequence(lstm_input, batch_first=True)

        # Process through LSTM
        lstm_output, (h_n, c_n) = self.lstm(padded_lstm_input)
        
        # Use the last output of the LSTM
        lstm_output = lstm_output[:, -1, :]
        
        # Pass through final linear layer
        x = self.lin(lstm_output)
        
        return x


if __name__ == "__main__":

    # Loading the dataset
    from .dataset import load_data
    import torch_geometric.transforms as T

    FEATURES = 1
    BATCH_SIZE = 3
    HIDDEN_CHANNELS = 100

    torch.manual_seed(1)

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
        # metadata -> [[node_types], [edge_types]
        metadata = undirected_data.metadata()
        # model_hetero = to_hetero(model, metadata)
        model_hetero = LatencyModel(
            num_features=FEATURES, hidden_channels=HIDDEN_CHANNELS, metadata=metadata, aggr='sum')

        output_hetero = model_hetero(
            undirected_data.x_dict, undirected_data.edge_index_dict, undirected_data.batch_dict)

        print(f"Output is {output_hetero} w/ shape is {output_hetero.shape}")

        # print(f"---- Hetero Output ----\n ")
        # for key, value in output_hetero.items():
        #     print(f"{key} is {value.shape}")

        break
