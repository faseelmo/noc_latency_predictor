import torch
import numpy as np
import networkx as nx 
from gcn.model import LatNet
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from gcn.utils import visGraph, manhattan_distance, net_4x4
from gcn.dataset import min_max_scaler

NETWORK = net_4x4
MODEL_PATH = 'gcn/results/epoch_500_L1/LatNet_200.pth'

# 0 -> Start Node
# 8 -> Exit Node

TASK = {
    0     : [1, 3],
    1     : [2],
    2     : [5], 
    3     : [4], 
    4     : [6], 
    5     : [6], 
    6     : [7], 
    7     : [8], 
}

MAP = {
    0     : 18,
    1     : 25,
    2     : 17, 
    3     : 31,  
    4     : 16,   
    5     : 28, 
    6     : 29, 
    7     : 24, 
    8     : 20, 
}

DEMAND = {
    0     : 0,
    1     : 1,
    2     : 81, 
    3     : 1,  
    4     : 53,   
    5     : 99, 
    6     : 14, 
    7     : 71, 
    8     : 0, 
}

DURATION = {
    0     : 0,
    1     : 74,
    2     : 13, 
    3     : 48,  
    4     : 16,   
    5     : 67, 
    6     : 83, 
    7     : 1, 
    8     : 0, 
}

def main():

    # Initializing the Model
    model_state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model = LatNet(in_features=4, num_nodes=9).to("cpu")
    model.load_state_dict(model_state_dict)

    # Initializing the Graph
    task_graph = nx.from_dict_of_lists(TASK, create_using=nx.DiGraph)
    
    # Get Edge Index 
    edge_list = list(task_graph.edges())

    # Get Demand and Duration Feature
    demand_feature = []
    duration_feature = []
    for demand_node, duration_node in zip(DEMAND, DURATION): 
        demand_feature.append(DEMAND[demand_node])
        duration_feature.append(DURATION[duration_node])
    demand_feature = min_max_scaler(np.array(demand_feature), 0, 100)
    duration_feature = min_max_scaler(np.array(duration_feature), 0, 100)

    # Get Manhattan Distance between nodes
    distance_list = []
    for (src_task, dest_task) in edge_list:
        src_pe = MAP[src_task]
        dest_pe = MAP[dest_task]
        
        src_loc = NETWORK[src_pe]
        dest_loc = NETWORK[dest_pe]

        distance = manhattan_distance(src_loc, dest_loc)
        distance_list.append(distance)


    # Get PE Position 
    x_pos = []
    y_pos = []
    mesh_size = int(len(NETWORK) ** 0.5) 
    for node in task_graph.nodes:
        pe = MAP[node]
        x,y = NETWORK[pe] 
        x_pos.append(x/(mesh_size-1))
        y_pos.append(y/(mesh_size-1))

    # Converting Everything to Torch
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    demand_feature = torch.tensor(demand_feature).view(-1,1).float()
    duration_feature = torch.tensor(duration_feature).view(-1,1).float()
    x_pos = torch.tensor(x_pos).view(-1,1).float()
    y_pos = torch.tensor(y_pos).view(-1,1).float()
    pe_pos = torch.cat([x_pos, y_pos], dim=1)

    x = torch.cat([pe_pos, demand_feature, duration_feature], dim=1)
    edge_attr = torch.tensor(distance_list).view(-1,1).float()
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    pred = model(data)
    print(f"Latency Prediction is {pred.item()}")

if __name__ == "__main__":
    main()


