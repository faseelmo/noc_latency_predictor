import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 

net_4x4 = {
    16: (0, 3), 17: (1, 3), 18: (2, 3), 19: (3, 3),
    20: (0, 2), 21: (1, 2), 22: (2, 2), 23: (3, 2),
    24: (0, 1), 25: (1, 1), 26: (2, 1), 27: (3, 1),
    28: (0, 0), 29: (1, 0), 30: (2, 0), 31: (3, 0)
}

net_3x3 = {
    9: (0, 2), 10: (1, 2), 11: (2, 2),
    12: (0, 1), 13: (1, 1), 14: (2, 1),
    15: (0, 0), 16: (1, 0), 17: (2, 0)
}

def visGraph(graph, pos=None):
    """
    usage: visGraph(data_dict['task_graph'], pos=data_dict['task_graph_pos'])
    """
    if pos is None:
        pos = nx.spring_layout(graph, seed=42)
        pos['Start'] = np.array([-1,0])
        pos['Exit'] = np.array([1,0])
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', arrowsize=20)
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
    
    nx.draw(G, network_grid, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', arrowsize=20)
    plt.title('4x4 Grid Graph')
    plt.show()


def visualize_pyG(data, pos=None):
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().tolist())

    if pos is None: pos = nx.spring_layout(graph)  

    node_features = {node: f"R: {data['x'][node].tolist()[0]:.0f}\nD: {data['x'][node].tolist()[1]:.0f}" for node in graph.nodes}
    nx.draw(graph, pos, with_labels=True, labels=node_features, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', arrowsize=20)  

    if 'edge_attr' in data:
        edge_labels = {tuple(e): str(int(attr.item())) for e, attr in zip(data.edge_index.t().tolist(), data.edge_attr)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        nx.draw(graph, pos,node_color='skyblue' , edgelist=data.edge_index.t().tolist())
    plt.show()



def manhattan_distance(src,dst):
    x1, y1 = src
    x2, y2 = dst
    return abs(x2 - x1) + abs(y2 - y1)

def convertTaskPosToPygPos(task_pos):
    data_pyg_pos = {}
    print(f"task_pos is {task_pos}")
    print(f"task_pos type is {type(task_pos)}")
    for key,value in task_pos.items():
        if key == 'Start': 
            data_pyg_pos[0] = value
        elif key == 'Exit': 
            data_pyg_pos[len(task_pos) - 1] = value
        else: 
            data_pyg_pos[key] = value
    print(f"New Pos is {data_pyg_pos}")
    return data_pyg_pos
