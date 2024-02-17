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
        edge_index = task_graph.edges()
        
        total_tasks = len(task_graph.nodes)
        last_task = len(task_graph.nodes) - 1
        
        converted_edge_index = convert_edge_index(edge_index, last_task)
        mapped_graph = get_mapped_graph(converted_edge_index, data_dict['map'], pe_4x4, router_4x4)

        mapped_edge_index = list(mapped_graph.edges())

        converted_edge_index_torch = torch.tensor(mapped_edge_index, dtype=torch.int).t().contiguous()

        total_num_of_nodes = 32
        dummy_input = torch.ones(total_num_of_nodes).view(-1,1)
        data = Data(x=dummy_input,edge_index=converted_edge_index_torch, y=target_value)


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

def xy_routing(source, target):
    """
        1. Determine the step for the X direction (1 if source is left of target, -1 if it's right)
        2. Move in the X direction
        3. Update source to the last point in the X direction
        4. Determine the step for the Y direction (1 if source is below target, -1 if it's above)
        5. Move in the Y direction

        Returns a list of (x, y) coordinates from source to target
    """
    path = []
    step_x = 1 if source[0] <= target[0] else -1
    for x in range(source[0], target[0] + step_x, step_x):
        path.append((x, source[1]))
    source = path[-1]
    step_y = 1 if source[1] <= target[1] else -1
    for y in range(source[1] + step_y, target[1] + step_y, step_y):
        path.append((target[0], y))
    return path

def get_nework_path(source_mapped, target_mapped, pe_net, router_net): 
    """
    Arg: (source, target) is the element of edge_index list 
    Function: 
        1. Maps the PE to the router
        2. Finds the XY path from source to target in coordinate-space
           Using XY routing algorithm
        3. Then converts the path from coordinate-space to PE-space

    return a list of nodes in the path from source to target in PE-space. 
    """
    source_xy = pe_net[source_mapped]
    target_xy = pe_net[target_mapped]
    xy_path = xy_routing(source_xy, target_xy)
    node_path = []
    for xy_node in xy_path:
        node_path.append(router_net.inverse[xy_node])
    node_path.insert(0, pe_net.inverse[source_xy])
    node_path.append(pe_net.inverse[target_xy])
    return node_path

def create_edge_index(node_path_list):
    """
    Given a list of node paths, create a list of edges suitable for creating a graph
    """
    edge_index = []
    for node_path in node_path_list:
        for idx, node in enumerate(node_path):
            if idx == len(node_path) - 1:
                break
            edge = (node, node_path[idx+1])
            # if edge not in edge_index:
            #     print(f"Edge already exists: {edge}")
            edge_index.append(edge)

    return edge_index

def get_mapped_graph(edge_index, map, pe_net, router_net):
    node_path_list = []
    for edge in edge_index:
        
        source_mapped = map[edge[0]]
        target_mapped = map[edge[1]]
        node_path = get_nework_path(source_mapped, target_mapped, pe_net, router_net)
        node_path_list.append(node_path)

    new_edge_index = create_edge_index(node_path_list)
    # print(f"New Edge Index: {new_edge_index}")
    return nx.DiGraph(new_edge_index)

def rename_nodes(graph):
    mapping = {}
    for idx, old_label in enumerate(graph.nodes()):
        new_label = idx
        mapping[old_label] = new_label

    new_graph = nx.relabel_nodes(graph, mapping)
    return new_graph

def convert_edge_index(edge_index, num_of_tasks):
    """
    Changes the Start and Exit nodes to 0 and num_of_tasks respectively
    """
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
