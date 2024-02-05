import os
import pickle
from natsort import natsorted

# from .utils import *

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
        # map_graph = data_dict['map']
        task_processing_time = data_dict['network_processing_time']

        task_graph = task_dag.graph
        edge_index = torch.tensor(list(task_graph.edges)).t().contiguous()         

        """Target Label"""
        target_label = torch.tensor([task_processing_time]).float()
        data = Data(edge_index=edge_index, y=target_label)
        
        """un-comment for debug"""
        # new_pos = utils.convertTaskPosToPygPos(data_dict['task_graph_pos'])
        # utils.visGraph(task_graph, pos=data_dict['task_graph_pos'])
        # utils.visGraph(map_graph, pos=data_dict['map_graph_pos'])
        # utils.visualize_pyG(data, pos=new_pos)

        return data

def custom_collate(data_list):
    return Batch.from_data_list(data_list)

def min_max_scaler(x, min, max):
    return ((x - min ) / (max - min))
    

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

if __name__ == "__main__":
    pickle_dir = 'training_data/data/task_7'
    dataset = CustomData(pickle_dir)

    print(f"Dataset size {len(dataset)}")
    data = dataset[3500]
    print(data)

    # print(f"\nInput Feature is \n{data.x}")
    # print(f"\nEdge Feature is \n{data.edge_attr}")
    # print(f"\nOuput Label {data.y}")
    # print(f"Nodes {data.node_attrs}")
