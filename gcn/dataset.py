import os
import pickle
from natsort import natsorted

import utils

import torch 
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, random_split, DataLoader

class CustomData(Dataset): 
    def __init__(self, pickle_dir): 
        entries = os.listdir(pickle_dir)
        self.data_dir = pickle_dir
        self.file_list = natsorted([entry for entry in entries if os.path.isfile(os.path.join(pickle_dir, entry))])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx]) 
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)

        task_graph = data_dict['task_graph']
        map_graph = data_dict['map_graph']
        task_duration = data_dict['duration']
        task_processing_time = data_dict['processing_time']

        #selecting the CPU demand from (CPU,Memory)
        task_demands = [demands[0] for demands in data_dict['demand']]

        task_node_list = list(task_graph.nodes)       
        map_node_list = list(map_graph.nodes)       
        map_edge_list = list(map_graph.edges)       

        """Since there is no demand/duration for Start and Exit Node"""
        task_demands.insert(0,0)
        task_demands.append(0)
        task_duration.insert(0,0)
        task_duration.append(0)

        """
            node_mapping={mapped_node: task_node, .... }
        """
        node_mapping = {}
        for task_index, map_index in zip(task_node_list, map_node_list):
            if task_index == 'Start': 
                node_mapping[map_index] = 0
            elif task_index == 'Exit': 
                node_mapping[map_index] = len(task_node_list) - 1
            else: 
                node_mapping[map_index] = task_index

        """
        Edge Index is for PyG
            map_edge_list = [(src_node, dest_node), ...]
            edge[0] -> src_node
            edge[1] -> dest_node
        """
        edge_index = torch.tensor([
            (node_mapping[edge[0]], node_mapping[edge[1]]) for edge in map_edge_list]
        ).t().contiguous()


        """Node Level Features"""
        # pe = torch.tensor(task_node_list).view(-1,1).float()
        demand_feature = torch.tensor(task_demands).view(-1,1).float()
        duration_feature = torch.tensor(task_duration).view(-1,1).float()
        x = torch.cat([demand_feature, duration_feature], dim=1)

        """Edge Level Features"""
        distance_list = []
        for edge in map_edge_list:
            src_node = edge[0]
            dest_node = edge[1]

            src_node_loc = utils.net_3x3[src_node]
            dest_node_loc = utils.net_3x3[dest_node]

            distance = utils.manhattan_distance(src_node_loc, dest_node_loc)
            distance_list.append(distance)
        edge_attr = torch.tensor(distance_list).view(-1,1).float()

        """Target Label"""
        target_label = torch.tensor([task_processing_time]).float()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target_label)
        
        """un-comment for debug"""
        # new_pos = utils.convertTaskPosToPygPos(data_dict['task_graph_pos'])
        # utils.visGraph(task_graph, pos=data_dict['task_graph_pos'])
        # utils.visGraph(map_graph, pos=data_dict['map_graph_pos'])
        # utils.visualize_pyG(data, pos=new_pos)

        return data

def custom_collate(data_list):
    return Batch.from_data_list(data_list)

def load_data(data_dir, batch_size=100):
    dataset = CustomData(data_dir)
    test_size = int(0.1 * len(dataset))
    train_dataset, test_dataset = random_split(
        dataset, [len(dataset) - test_size, test_size] 
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, collate_fn=custom_collate)

    return train_loader, test_loader

def test():
    pickle_dir = '../training_data/data/task_7'
    dataset = CustomData(pickle_dir)
    print(f"Dataset size {len(dataset)}")
    data = dataset[3500]
    print(data)

    print(f"\nInput Feature is \n{data.x}")
    print(f"\nEdge Feature is \n{data.edge_attr}")
    print(f"\nOuput Label {data.y}")

    print(f"Nodes {data.node_attrs}")

# test()