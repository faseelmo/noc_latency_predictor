import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from bidict import bidict

pe_4x4 = bidict({
    28: (0, 3), 29: (1, 3), 30: (2, 3), 31: (3, 3),
    24: (0, 2), 25: (1, 2), 26: (2, 2), 27: (3, 2),
    20: (0, 1), 21: (1, 1), 22: (2, 1), 23: (3, 1),
    16: (0, 0), 17: (1, 0), 18: (2, 0), 19: (3, 0)
})

router_4x4 = bidict({
    12: (0, 3), 13: (1, 3), 14: (2, 3), 15: (3, 3),
    8: (0, 2), 9: (1, 2), 10: (2, 2), 11: (3, 2),
    4: (0, 1), 5: (1, 1), 6: (2, 1), 7: (3, 1),
    0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)
})


pe_3x3 = bidict({
    15: (0, 2), 16: (1, 2), 17: (2, 2),
    12: (0, 1), 13: (1, 1), 14: (2, 1),
    9: (0, 0), 10: (1, 0), 11: (2, 0)
})

router_3x3 = bidict({
    6: (0, 2), 7: (1, 2), 8: (2, 2),
    3: (0, 1), 4: (1, 1), 5: (2, 1),
    0: (0, 0), 1: (1, 0), 2: (2, 0)
})


def visGraph(graph, pos=None):
    """
    usage: visGraph(data_dict['task_graph'], pos=data_dict['task_graph_pos'])
    """
    if pos is None:
        # pos = nx.spring_layout(graph, seed=42)
        pos = nx.spectral_layout(graph)
        # pos['Start'] = np.array([-1,0])
        # pos['Exit'] = np.array([1,0])
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue',
            font_size=10, font_color='black', font_weight='bold', arrowsize=20)
    plt.show()


def visGraphGrid(edges, network_grid):
    """
    usage: visGraphGrid(data_dict['task_graph'].edges, net_3x3)
    """
    nodes = list(range(len(network_grid), 2*len(network_grid), 1))
    # print(f"\nNum of PE's = {len(network_grid)}")

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    nx.draw(G, network_grid, with_labels=True, node_size=700, node_color='skyblue',
            font_size=10, font_color='black', font_weight='bold', arrowsize=20)
    plt.title('4x4 Grid Graph')
    plt.show()


def visualize_pyG(data, pos=None):
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().tolist())

    if pos is None:
        pos = nx.spring_layout(graph)

    # node_features = {node: f"R: {data['x'][node].tolist()[0]:.0f}\nD: {data['x'][node].tolist()[1]:.0f}" for node in graph.nodes}
    # nx.draw(graph, pos, with_labels=True, labels=node_features, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', arrowsize=20)
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue',
            font_size=10, font_color='black', font_weight='bold', arrowsize=20)

    if 'edge_attr' in data:
        edge_labels = {tuple(e): str(int(attr.item())) for e, attr in zip(
            data.edge_index.t().tolist(), data.edge_attr)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        nx.draw(graph, pos, node_color='skyblue',
                edgelist=data.edge_index.t().tolist())
    plt.show()


def manhattan_distance(src, dst):
    x1, y1 = src
    x2, y2 = dst
    return abs(x2 - x1) + abs(y2 - y1)


def convertTaskPosToPygPos(task_pos):
    data_pyg_pos = {}
    print(f"task_pos is {task_pos}")
    print(f"task_pos type is {type(task_pos)}")
    for key, value in task_pos.items():
        if key == 'Start':
            data_pyg_pos[0] = value
        elif key == 'Exit':
            data_pyg_pos[len(task_pos) - 1] = value
        else:
            data_pyg_pos[key] = value
    print(f"New Pos is {data_pyg_pos}")
    return data_pyg_pos


def load_data(dir_path, num_mappings):
    from natsort import natsorted
    import os
    import torch

    if not os.path.exists(dir_path):
        print('file not found')
        return

    entries = os.listdir(dir_path)
    files = natsorted(entries)

    list_of_graphs = []
    list_of_iso_graph = []

    for idx, file in enumerate(files):
        if file.endswith('.pt'):
            data = torch.load(os.path.join(dir_path, file))
            list_of_iso_graph.append(data)
            if (idx + 1) % num_mappings == 0:
                # print(f"Condition is True")
                list_of_graphs.append(list_of_iso_graph)
                list_of_iso_graph = []

    return list_of_graphs


def do_inference(data, MODEL, model_path, input_features, is_hetero=False):
    import torch
    model_state_dict = torch.load(
        model_path,  map_location=torch.device('cpu'))
    
    if is_hetero:
        metadata = data.metadata()
        print(data.batch_dict)
        model = MODEL(num_features=input_features, hidden_channels=512, metadata=metadata).to(torch.device('cpu'))
        x = data.x_dict
        edge_index = data.edge_index_dict
        batch = data.batch_dict['link']
        pred_latency = MODEL(x, edge_index, batch).squeeze(1)
    else:
        model = MODEL(num_node_features=input_features).to(torch.device('cpu'))
        model.load_state_dict(model_state_dict)
        pred_latency = model(data.x, data.edge_index, data.batch)

    actual_latency = data.y
    # model.load_state_dict(model_state_dict)
    return pred_latency.item(), actual_latency.item()


def get_tau_result(list_of_graphs, MODEL, model_path, input_features, show_plot=False, is_hetero=False):
    from scipy.stats import kendalltau
    tau_list = []
    for idx, list_of_iso in enumerate(list_of_graphs):

        list_of_pred = []
        list_of_actual = []
        for data in list_of_iso:
            if is_hetero:
                pred, actual = do_inference(
                    data, MODEL=MODEL, model_path=model_path, input_features=input_features, is_hetero=True)
            else: 
                pred, actual = do_inference(
                    data, MODEL=MODEL, model_path=model_path, input_features=input_features)
            list_of_pred.append(pred)
            list_of_actual.append(actual)

        tau, _ = kendalltau(list_of_pred, list_of_actual)
        tau_list.append([idx, tau])

        if show_plot:
            plt.plot(list_of_pred, label='Predicted')
            plt.plot(list_of_actual, label='Actual')
            plt.legend()
            plt.text(0.5, 0.9, f"Tau: {tau}", fontsize=12,
                     transform=plt.gcf().transFigure)
            plt.show()

    return np.array(tau_list)

def get_tau_from_dataloader(dataloader, model_with_state_dict, num_mapping): 
    from scipy.stats import kendalltau
    tau_list = []
    
    list_of_pred = []
    list_of_actual = []

    for idx, data in enumerate(dataloader, start=1):
        pred = model_with_state_dict(data.x_dict, data.edge_index_dict, data.batch_dict)
        list_of_pred.append(pred.item())
        list_of_actual.append(data.y.item())

        if idx % num_mapping == 0:
            tau, _ = kendalltau(list_of_pred, list_of_actual)
            tau_list.append([idx, tau])
            list_of_pred = []
            list_of_actual = []

    return np.array(tau_list)

def get_tau_from_dataloader(dataloader, model_with_state_dict, num_mapping): 
    from scipy.stats import kendalltau
    tau_list = []
    
    list_of_pred = []
    list_of_actual = []

    for idx, data in enumerate(dataloader):
        pred = model_with_state_dict(data.x_dict, data.edge_index_dict, data.batch_dict)
        list_of_pred.append(pred.item())
        list_of_actual.append(data.y.item())

        if idx % num_mapping == 0:
            print(f"List of Pred is {list_of_pred}")
            print(f"List of Actual is {list_of_actual}")
            tau, _ = kendalltau(list_of_pred, list_of_actual)
            print(f"Tau is {tau}")
            tau_list.append([idx, tau])
            list_of_pred = []
            list_of_actual = []

    return np.array(tau_list)

from torch.utils.data import Dataset, DataLoader
class TestDataset(Dataset): 
    def __init__(self, list_of_graphs): 
        self.list_of_graphs = list_of_graphs    

    def __len__(self):
        num_of_graphs = len(self.list_of_graphs)
        num_of_mapping_per_graph = len(self.list_of_graphs[0])
        return num_of_graphs*num_of_mapping_per_graph

    def __getitem__(self, idx):
        graph_idx = idx // len(self.list_of_graphs[0])
        mapping_idx = idx % len(self.list_of_graphs[0])
        return self.list_of_graphs[graph_idx][mapping_idx]

from torch_geometric.data import Batch
def custom_collate(data_list):
    return Batch.from_data_list(data_list)