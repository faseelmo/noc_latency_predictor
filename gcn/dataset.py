import os
import pickle
from natsort import natsorted

from .utils import *
import sys
sys.path.append('training_data')

import torch 
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, random_split, DataLoader

class CustomData(Dataset): 
    def __init__(self, pickle_dir): 
        entries = os.listdir(pickle_dir)
        self.data_dir = pickle_dir
        self.file_list = natsorted([entry for entry in entries if os.path.isfile(os.path.join(pickle_dir, entry))])
        self.distance_max = 6

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx]) 
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)

        task_dag = data_dict['task_dag']
        task_processing_time = float(data_dict['network_processing_time'])
        target_value = torch.tensor([task_processing_time]).float()

        task_graph = task_dag.graph
        edge_index = list(task_graph.edges)
        
        total_tasks = len(task_graph.nodes)
        last_task = len(task_graph.nodes) - 1
        
        converted_edge_index = convert_edge_index(edge_index, last_task)
        converted_edge_index_torch = torch.tensor(converted_edge_index, dtype=torch.int).t().contiguous()

        dummy_input = torch.ones(total_tasks).view(-1,1)
        data = Data(x=dummy_input,edge_index=converted_edge_index_torch, y=target_value)
        # data.num_nodes = num_of_tasks + 1


        # """Target Label"""
        # target_label = torch.tensor([task_processing_time]).float()
        # data = Data(edge_index=edge_index, y=target_label)
        
        """un-comment for debug"""
        # new_pos = utils.convertTaskPosToPygPos(data_dict['task_graph_pos'])
        # utils.visGraph(task_graph, pos=data_dict['task_graph_pos'])
        # utils.visGraph(map_graph, pos=data_dict['map_graph_pos'])
        # # printask_t(dag.pitionos)
        # visualize_pyG(data)
        # task_dag.plot(show_node_attrib=False)


        # pass
        return data

def convert_edge_index(edge_index, num_of_tasks):
    converted_edge_index = []
    node_mapping = {'Start': 0, 'Exit': num_of_tasks}

    for src, dest in edge_index:
        if src == 'Start':
            src = node_mapping[src]

        if dest == 'Exit':
            dest = node_mapping[dest]

        converted_edge_index.append((src, dest))

    return converted_edge_index

def custom_collate(data_list):
    return Batch.from_data_list(data_list)

def min_max_scaler(x, min, max):
    return ((x - min ) / (max - min))
    

def load_data(data_dir, batch_size=100):
    dataset = CustomData(data_dir)
    valid_size = int(0.1 * len(dataset))
    train_dataset, test_dataset = random_split(
        dataset, [len(dataset) - valid_size, valid_size] 
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, collate_fn=custom_collate
    )
    valid_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, collate_fn=custom_collate)

    return train_loader, valid_loader

if __name__ == "__main__":
    pickle_dir = 'training_data/data/training_data'
    dataset = CustomData(pickle_dir)

    print(f"Dataset size {len(dataset)}")
    data = dataset[100]
    print(type(data))

    print(f"Graph is Valid: {data.validate(raise_on_error=True)}")
    print(f"Input Feature ({data.x.shape})is \n{data.x}")
    print(f"\nEdge Feature is \n{data.edge_attr}")
    print(f"\nOuput Label {data.y}")
    print(f"\nNodes {data.node_attrs}")
