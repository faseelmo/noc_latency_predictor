import copy
from bidict import bidict

import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx, to_networkx

"""
For usage of this class, refer to the main function at the bottom of the file

Member functions:

    init_network(self, network) -> tuple:

    dag_on_network(self, dag, map) -> nx.Graph:

    generate_tensor(self, nx_graph, num_node_types, target, debug=False) -> Data:

    generate_network_graph(self, processing_element, router) -> nx.Graph:

    create_rename_map(self, dag, map) -> tuple:

    visualize_network_3d(self, graph_=None) -> None:
"""


class GraphUtils():
    def __init__(self, network):

        self.router_z = 0
        self.processing_element_z = 0.5
        self.dag_z = 1
        # of a 4x4 network, used for normalization of task coordinates
        self.max_network_coordinate_value = 3

        processing_element, router = self.init_network(network)
        self.last_pe_index = max(processing_element.keys())

        self.network_graph = self.generate_network_graph(
            processing_element, router)

    def generate_tensor(self, nx_graph, num_node_types, target, debug=False) -> Data:
        """nx_graph modified in this function."""

        """ Dictionary for unique node types """
        category_dict = {
            'router': 0,
            'pe': 1,
            'task': 2
        }

        """ Creating a list of node features (for one-hot encoding) """
        node_features = []
        for node in nx_graph.nodes():
            node_type = nx_graph.nodes[node]['type']
            node_features.append(category_dict[node_type])
            # Clearing node attributes for converting to PyTorch Geometric later
            nx_graph.nodes[node].clear()

        """ One-hot encoding Node types """
        node_features_one_hot = F.one_hot(
            torch.tensor(node_features), num_classes=num_node_types).to(torch.float)

        """ Converting NetworkX graph to PyTorch Geometric """
        graph_tensor = from_networkx(nx_graph)
        graph_tensor.y = torch.tensor([float(target)])
        graph_tensor.x = node_features_one_hot

        """
            for peace of mind, you can check if the edge index is the 
            same by converting it back to networkx (if not converted and are using the original 
            the order of edges might be different)
            check if graph_tensor.edge_index.t() == nx_graph.edges

            graph_networkx = to_networkx(graph_tensor)
            for graph_edge, tensor_edge in zip(graph_networkx.edges, graph_tensor.edge_index.t()):
                assert graph_edge == tuple(tensor_edge.tolist())
        """
        if debug:
            graph_networkx = to_networkx(graph_tensor)
            for graph_edge, tensor_edge in zip(graph_networkx.edges, graph_tensor.edge_index.t()):
                assert graph_edge == tuple(tensor_edge.tolist())

        return graph_tensor

    def dag_on_network(self, dag, map) -> nx.Graph:
        """ type(dag) -> DAG() """

        """ Need to rename otherwise network graph
        and dag graph will have same node names
        task names 0-8, 
        router name 0-15, 
        pe names 16-31, 

        so, we rename the task nodes to 32-40
        """
        rename_dict, new_pos, new_map = self.create_rename_map(dag, map)
        dag_graph = nx.relabel_nodes(dag.graph, rename_dict)
        # print(f"Renamed Nodes: {rename_dict}")
        # print(f"Old nodes with postion: {dag.position}")
        # print(f"New nodes with postion: {new_pos}")
        # print(f"Old map: {map}")
        # print(f"New map: {new_map}")
        # print(f"Old graph: {dag.graph.nodes()}")
        # print(f"New graph: {dag_graph.nodes()}")
        # print(f"Node with Attributes {dag_graph.nodes(data=True)}")
        # print(f"Node with Edge Attributes {dag_graph.edges(data=True)}")

        """Adding Z coordinates to the new_pos
        Need to normalize the coordinates of the dag so that 
        they all fit in the same network graph
        """
        max_current_coordinate = max(max(pos) for pos in new_pos.values())
        scaling_factor = self.max_network_coordinate_value / max_current_coordinate

        new_pos_with_z = {}
        for node, pos in new_pos.items():
            new_pos_with_z[node] = (
                pos[0]*scaling_factor, pos[1]*scaling_factor, self.dag_z)

        """Code below adds "pos" as a node attribute in the dag_graph
        along with delay"""
        nx.set_node_attributes(dag_graph, new_pos_with_z, 'pos')

        """Demand is an edge attribute right now, think about how to use it
        Maybe assign this to a node (reprsenting link) inserted inbetween exisiting nodes"""

        """Adding the dag to the network graph"""
        network_graph = copy.deepcopy(self.network_graph)
        for node in dag_graph.nodes():
            """type='task' could be changed later to 3 different types
            (start, task, end)"""
            network_graph.add_node(node, pos=new_pos_with_z[node], type='task')

        """Connecting the edges of the dag"""
        for node in dag_graph.nodes():
            for neighbor in dag_graph.neighbors(node):
                network_graph.add_edge(node, neighbor)

        """Connecting the dag to the network graph"""
        for node, pe in new_map.items():
            network_graph.add_edge(node, pe)

        return network_graph

    def generate_network_graph(self, processing_element, router) -> nx.Graph:

        G = nx.Graph()

        """ Adding the nodes with their coordinates and type """
        for node, coordinates in router.items():
            G.add_node(node, pos=coordinates, type='router')

        for node, coordinates in processing_element.items():
            G.add_node(node, pos=coordinates, type='pe')

        """ Adding Edges (Mesh Topology)
        Maybe there's a better way to do this. But its not real time critical.
        Only used for visualization"""
        for src_node, src_coordinates in router.items():
            """ coordinates -> (x, y, z)"""
            for dest_node, dest_coordinates in router.items():
                euc_dist = ((src_coordinates[0] - dest_coordinates[0]) **
                            2 + (src_coordinates[1] - dest_coordinates[1])**2)**0.5
                if euc_dist == 1:
                    G.add_edge(src_node, dest_node)

        """ Adding Edges between PE and Router"""
        for src_node, src_coordinates in processing_element.items():
            src_x, src_y, src_z = src_coordinates
            for dest_node, dest_coordinates in router.items():
                dest_x, dest_y, dest_z = dest_coordinates
                if src_x == dest_x and src_y == dest_y:
                    G.add_edge(src_node, dest_node)

        return G

    def create_rename_map(self, dag, map) -> tuple:
        """ Renaming the nodes of the dag_graph
        - Renames the position dictionary accordingly
        - Renames mapping dictionary accordingly
        Why do we do this? 
        check dag_on_network() comments for more info
        """

        dag_graph = dag.graph
        dag_position = dag.position

        rename_dict = {}
        new_map = {}

        num_of_nodes = dag_graph.number_of_nodes()
        last_pe_index = self.last_pe_index

        for node in dag_graph.nodes():
            if node == "Start":
                rename_dict[node] = last_pe_index + 1
            elif node == "Exit":
                rename_dict[node] = last_pe_index + num_of_nodes
            else:
                rename_dict[node] = last_pe_index + node + 1

        new_pos = {}
        for old_key, value in dag_position.items():
            new_key = rename_dict[old_key]
            new_pos[new_key] = value

        for task, pe in map.items():
            if task == 0:
                updated_node = rename_dict['Start']
            elif task == num_of_nodes - 1:
                updated_node = rename_dict['Exit']
            else:
                updated_node = rename_dict[task]

            new_map[updated_node] = pe

        return rename_dict, new_pos, new_map

    def visualize_network_3d(self, graph_=None) -> None:

        if graph_ is None:
            G = self.network_graph
        else:
            G = graph_

        pos = nx.get_node_attributes(G, 'pos')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for node, coordinates in pos.items():
            node_type = G.nodes[node]['type']
            num_node_connection = len(G.edges(node))
            if node_type == 'router':
                ax.scatter(*coordinates, color='b')
            elif node_type == 'pe':
                if num_node_connection > 1:
                    ax.scatter(*coordinates, color='r')  # Regular red
                else:
                    ax.scatter(*coordinates, color=(0.4, 0, 0))  # Darker red
            elif node_type == 'task':
                ax.scatter(*coordinates, color='g')
            else:
                raise NotImplementedError(
                    "Node type not supported in visualization_network_3d()")

            ax.text(*coordinates, f'{node}', color='black')

        for edge in G.edges():
            source_node, destination_node = edge
            source_node_type = G.nodes[source_node]['type']
            destination_node_type = G.nodes[destination_node]['type']

            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]

            if source_node_type == 'task' and destination_node_type == 'task':
                ax.plot(x, y, z, color='g')  # Green
            elif set((source_node_type, destination_node_type)) == {'router', 'pe'}:
                ax.plot(x, y, z, color='0.75')  # Light gray
            elif source_node_type == 'router' and destination_node_type == 'router':
                ax.plot(x, y, z, color='0.75')  # Light gray
            elif set((source_node_type, destination_node_type)) == {'task', 'pe'}:
                ax.plot(x, y, z, color=('y'))  # Yellow
            else:
                raise NotImplementedError(
                    "Edge type not supported in visualization_network_3d()")
        plt.show()

    def init_network(self, network) -> tuple:
        if network == int(4):

            pe_z = self.processing_element_z
            processing_element = bidict({
                28: (0, 3, pe_z), 29: (1, 3, pe_z), 30: (2, 3, pe_z), 31: (3, 3, pe_z),
                24: (0, 2, pe_z), 25: (1, 2, pe_z), 26: (2, 2, pe_z), 27: (3, 2, pe_z),
                20: (0, 1, pe_z), 21: (1, 1, pe_z), 22: (2, 1, pe_z), 23: (3, 1, pe_z),
                16: (0, 0, pe_z), 17: (1, 0, pe_z), 18: (2, 0, pe_z), 19: (3, 0, pe_z)
            })

            router_z = self.router_z
            router = bidict({
                12: (0, 3, router_z), 13: (1, 3, router_z), 14: (2, 3, router_z), 15: (3, 3, router_z),
                8:  (0, 2, router_z), 9:  (1, 2, router_z), 10: (2, 2, router_z), 11: (3, 2, router_z),
                4:  (0, 1, router_z), 5:  (1, 1, router_z), 6:  (2, 1, router_z), 7:  (3, 1, router_z),
                0:  (0, 0, router_z), 1:  (1, 0, router_z), 2:  (2, 0, router_z), 3:  (3, 0, router_z)
            })

        else:
            raise NotImplementedError("Network size not supported")

        return processing_element, router


if __name__ == '__main__':
    """Takes in the argument of the index of the dag to be visualized"""

    import pickle
    import sys

    default_index = 1
    index = int(sys.argv[1]) if len(sys.argv) > 1 else default_index

    """Loading the simulation data from the pickle file"""
    dag_dir = f'data/task_7/{index}.pickle'
    dag = pickle.load(open(dag_dir, 'rb'))

    """
    Extracting data from the loaded pickle file
    required data
        1. network
        2. task graph (task_dag)
        3. mapping dictionary (map) (if task on top network)
        4. network processing time (for regression target)
    """
    data_network = dag['network']

    data_dag = dag['task_dag']
    data_map = dag['map']
    # data_dag.plot(show_node_attrib=False) # Uncomment to visualize the task graph in 2d

    data_target = dag['network_processing_time']

    graph = GraphUtils(data_network)
    # graph.visualize_network_3d() # Uncomment to visualize the network in 3d
    dag_on_network = graph.dag_on_network(data_dag, data_map)
    # Uncomment to visualize the network with the dag in 3d
    graph.visualize_network_3d(dag_on_network)
    graph_tensor = graph.generate_tensor(
        dag_on_network, num_node_types=3, target=data_target)

    print(f"Graphs is directed: {graph_tensor.is_directed()}")
    print(f"Graph has self loops: {graph_tensor.has_self_loops()}")
    print(f"Graph has isolated nodes: {graph_tensor.has_isolated_nodes()}")
    print(f"Graph is valid: {graph_tensor.validate()}")
