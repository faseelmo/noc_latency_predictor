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
            data = pickle.load(file)

        data.x = data.x.float()

        return data

def custom_collate(data_list):
    return Batch.from_data_list(data_list)


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
        test_dataset, batch_size=batch_size, shuffle=False, 
        drop_last=True, collate_fn=custom_collate)

    return train_loader, valid_loader

if __name__ == "__main__":
    pickle_dir = 'training_data/data/training_data_tensor'
    dataset = CustomData(pickle_dir)

    print(f"Dataset size {len(dataset)}")
    data = dataset[0]
    print(type(data))

    print(f"Graph is Valid: {data.validate(raise_on_error=True)}")
    print(f"Edge Index is {data.edge_index.t()}")
    print(f"Input Feature ({data.x.shape})is \n{data.x.t()}")
    print(f"\nEdge Feature is \n{data.edge_attr}")
    print(f"\nOuput Label {data.y}")
    print(f"\nNodes {data.node_attrs}")
