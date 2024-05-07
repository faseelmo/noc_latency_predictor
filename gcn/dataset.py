import os
import pickle
from natsort import natsorted

from .utils import * 
# import sys
# sys.path.append('training_data')

import torch 
# import random 
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
        return torch.load(file_path)

def custom_collate(data_list):
    return Batch.from_data_list(data_list)


def load_data(data_dir, batch_size=100):
    dataset = CustomData(data_dir)

    valid_size = int(0.01 * len(dataset))
    train_dataset, test_dataset = random_split(
        dataset, [len(dataset) - valid_size, valid_size] 
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, collate_fn=custom_collate, pin_memory=True, num_workers=4
    )
    print(f"\nNumber of batches in train_loader: {len(train_loader)}, total data points: {len(train_loader.dataset)}")

    valid_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        drop_last=True, collate_fn=custom_collate, pin_memory=True, num_workers=4
    )

    print(f"Number of batches in valid_loader: {len(valid_loader)}, total data points: {len(valid_loader.dataset)}")

    return train_loader, valid_loader

if __name__ == "__main__":
    # pickle_dir = 'training_data/data/unique_graphs_with_links'
    pickle_dir = 'training_data/data/task_from_graph_dag_over_network_train'
    dataset = CustomData(pickle_dir)

    print(f"Dataset size {len(dataset)}")
    data = dataset[100]
    print(type(data))
    print(data)

    print(f"Graph is Valid: {data.validate(raise_on_error=True)}")
    print(f"Edge Index is {data.edge_index.t()}")
    print(f"Input Feature ({data.x.shape})is \n{data.x.t()}")
    print(f"\nEdge Feature is \n{data.edge_attr}")
    print(f"\nOuput Label {data.y}")
    print(f"\nNodes {data.node_attrs}")
