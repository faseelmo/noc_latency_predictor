import copy
from bidict import bidict

import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx, to_networkx
import mpl_toolkits.mplot3d

"""
Usage ( for more info check main() ):

    graph                       =   GraphUtils              ( dag['network'] )

    dag_on_network, new_map     =   graph.dag_on_network    ( dag['task_dag'], dag['map'] )
    graph_with_link_nodes       =   graph.create_link_nodes ( dag_on_network, new_map )

    graph_tensor                =   graph.generate_tensor   (
                                                              dag_on_network (or) graph_with_link_nodes,
                                                              target_            =   dag['network_processing_time']
                                                            )

Member functions:

    init_network( network_ )                                -> tuple    ( processing_element, router )

    generate_network_graph( )                               -> nx.Graph

    dag_on_network( dag_, map_ )                            -> tuple    ( network_graph, new_map )

    create_rename_map( dag_, map_ )                         -> tuple    ( rename_dict, new_pos, new_map )

    create_link_nodes( graph_, map_ )                       -> nx.Graph

    generate_tensor( nx_graph_, target_, debug_=False )     -> Data

    visualize_network_3d( graph_=None )                     -> None

"""


class GraphUtils():
    def __init__(self, network_):

        self.router_z = 0
        self.processing_element_z = 0.5
        self.dag_z = 1
        """ of a 4x4 network, used for normalization of task coordinates """
        self.max_network_coordinate_value = 3
        self.processing_element = None
        self.router = None

        self.processing_element, self.router = self.init_network(network_)
        self.last_pe_index = max(self.processing_element.keys())

        self.network_graph = self.generate_network_graph()

    def generate_tensor(self, nx_graph_, target_, debug_=False) -> Data:
        """nx_graph modified in this function."""

        """ Dictionary for unique node types """
        category_dict = {
            'router': 0,
            'pe': 1,
            'task': 2,
            'start_task': 3,
            'end_task': 4,
            'link': 5
        }

        """ Creating a list of node features (for one-hot encoding) """
        node_features = []
        for node in nx_graph_.nodes():
            node_type = nx_graph_.nodes[node]['type']
            node_features.append(category_dict[node_type])
            # Clearing node attributes for converting to PyTorch Geometric later
            nx_graph_.nodes[node].clear()

        """ One-hot encoding Node types """
        node_features_one_hot = F.one_hot(
            torch.tensor(node_features), num_classes=len(category_dict)).to(torch.float)

        """ Converting NetworkX graph to PyTorch Geometric """
        graph_tensor = from_networkx(nx_graph_)
        graph_tensor.y = torch.tensor([float(target_)])
        graph_tensor.x = node_features_one_hot

        """
            for peace of mind, you can check if the edge index is the
            same by converting it back to networkx (if not converted and are using the original
            the order of edges might be different)
            (post coding thought: convert the edgeview to list
            and then sort?, look into self.create_link_nodes() for more info)
        """
        if debug_:
            graph_networkx = to_networkx(graph_tensor)
            for graph_edge, tensor_edge in zip(graph_networkx.edges, graph_tensor.edge_index.t()):
                assert graph_edge == tuple(tensor_edge.tolist(
                )), "Edge index mismatch between graph and tensor"

        return graph_tensor

    def dag_on_network(self, dag_, map_) -> tuple:
        """ type(dag) -> DAG() """

        """ Need to rename otherwise network graph
        and dag graph will have same node names
        task names 0-8,
        router name 0-15,
        pe names 16-31,

        so, we rename the task nodes to 32-40
        """
        rename_dict, new_pos, new_map = self.create_rename_map(dag_, map_)
        dag_graph = nx.relabel_nodes(dag_.graph, rename_dict)
        # print(f"Renamed Nodes: {rename_dict}")
        # print(f"Old nodes with postion: {dag_.position}")
        # print(f"New nodes with postion: {new_pos}")
        # print(f"Old map: {map_}")
        # print(f"New map: {new_map}")
        # print(f"Old graph: {dag_.graph.nodes()}")
        # print(f"New graph: {dag_graph.nodes()}")
        # print(f"Node with Attributes {dag_graph.nodes(data=True)}")
        # print(f"Node with Edge Attributes {dag_graph.edges(data=True)}")

        """Adding Z coordinates to the new_pos
        Need to normalize the coordinates of the dag so that
        they all fit in the same network graph
        max_dag_coordinate is the maximum coordinate (pos) value of the dag 
        """
        max_dag_coordinate = max(max(pos) for pos in new_pos.values())
        scaling_factor = self.max_network_coordinate_value / max_dag_coordinate

        new_pos_with_z = {}
        y_offset = self.max_network_coordinate_value // 2
        for node, pos in new_pos.items():
            new_pos_with_z[node] = (
                pos[0]*scaling_factor, pos[1]*scaling_factor + y_offset, self.dag_z)

        """Code below adds "pos" as a node attribute in the dag_graph
        along with delay"""
        nx.set_node_attributes(dag_graph, new_pos_with_z, 'pos')

        """Demand is an edge attribute right now, think about how to use it
        Maybe assign this to a node (reprsenting link) inserted inbetween exisiting nodes"""

        """Adding the dag to the network graph"""
        network_graph = copy.deepcopy(self.network_graph)
        for node in dag_graph.nodes():
            """Task can be  3 types (start_task, task, end_task)"""
            if node == self.last_pe_index + 1:
                network_graph.add_node(
                    node, pos=new_pos_with_z[node], type='start_task')
            elif node == self.last_pe_index + len(dag_graph.nodes()):
                network_graph.add_node(
                    node, pos=new_pos_with_z[node], type='end_task')
            else:
                network_graph.add_node(
                    node, pos=new_pos_with_z[node], type='task')

        """Connecting the edges of the dag"""
        for node in dag_graph.nodes():
            for neighbor in dag_graph.neighbors(node):
                network_graph.add_edge(node, neighbor)

        """Connecting the dag to the network graph"""
        for node, pe in new_map.items():
            network_graph.add_edge(node, pe)

        return network_graph, new_map

    def create_link_nodes(self, graph_, map_, visualize_=False) -> nx.Graph:
        """
        arg graph: nx.Graph (usually the output of dag_on_network() or create_link_nodes())
        Inserting link nodes between the task nodes
        Connect these link nodes to relevant routers
        How to find the relevant routers?
            - Check the src and dest task nodes of the link node
            - Check the mapping of the src and dest task nodes (to a pe node)
            - Find the XY Routing between the src and dest pe nodes
                - Get the coordinates of the src and dest pe
                - Move in the X direction till the x coordinates of src
                    and dest match
                - Then move in the Y direction till the current coordinate and
                    the dest coordinate match
            - Connect the link node to the routers in the XY Routing
        """
        graph = copy.deepcopy(graph_)

        task_types = {'task', 'start_task', 'end_task'}

        if visualize_:
            """ Creating a new graph for visualization purposes
                - Removing the edges between the pe and task nodes"""
            graph_for_vis = copy.deepcopy(graph_)
            for edge in graph_for_vis.edges():
                source_node, destination_node = min(edge), max(edge)
                is_source_pe = graph_for_vis.nodes[source_node].get(
                    'type') == 'pe'
                is_dest_task = graph_for_vis.nodes[destination_node].get(
                    'type') in task_types
                if is_source_pe and is_dest_task:
                    graph_for_vis.remove_edge(source_node, destination_node)

        link_node_index = max(graph.nodes()) + 1
        edges = list(graph.edges)
        edges.sort()

        for edge in edges:
            """
            src is always less than dest ? In our application yes?
            """
            src_node, dest_node = min(edge), max(edge)

            """ Checking if the nodes are task nodes"""
            if graph.nodes[src_node]['type'] in task_types and graph.nodes[dest_node]['type'] in task_types:

                """ Adding link nodes between task nodes"""
                link_pos_x = (graph.nodes[src_node]['pos']
                              [0] + graph.nodes[dest_node]['pos'][0]) / 2
                link_pos_y = (graph.nodes[src_node]['pos']
                              [1] + graph.nodes[dest_node]['pos'][1]) / 2
                link_pos = (link_pos_x, link_pos_y, self.dag_z)
                graph.add_node(link_node_index, pos=link_pos, type='link')

                """ Finding XY Routing between the src and dest pe nodes"""
                src_pe = map_[src_node]
                dest_pe = map_[dest_node]

                src_pe_coordinates = graph.nodes[src_pe]['pos']
                dest_pe_coordinates = graph.nodes[dest_pe]['pos']

                """ Need to change the z coordinate from pe to router """
                router_coordinates = list(src_pe_coordinates)
                router_coordinates[2] = self.router_z

                path_in_coordinates = []

                list_router_link_connection = []
                list_router_link_connection.append(
                    self.router.inverse[tuple(router_coordinates)])

                """if src is on the left of dest, step_x is 1, else -1"""
                step_x = 1 if src_pe_coordinates[0] < dest_pe_coordinates[0] else -1

                """Adding the src router coordinates to the path"""
                path_in_coordinates.append(tuple(router_coordinates))
                while router_coordinates[0] != dest_pe_coordinates[0]:
                    """ trasnverse in x direction """
                    router_coordinates[0] += step_x
                    path_in_coordinates.append(tuple(router_coordinates))
                    list_router_link_connection.append(
                        self.router.inverse[tuple(router_coordinates)])

                """if src is below dest, step_y is 1 (i.e we want to move up), else -1"""
                step_y = 1 if router_coordinates[1] < dest_pe_coordinates[1] else -1
                while router_coordinates[1] != dest_pe_coordinates[1]:
                    """ trasnverse in y direction """
                    router_coordinates[1] += step_y
                    path_in_coordinates.append(tuple(router_coordinates))
                    list_router_link_connection.append(
                        self.router.inverse[tuple(router_coordinates)])

                """Connecting the link node to the routers in the XY Routing"""
                for routers in list_router_link_connection:
                    graph.add_edge(link_node_index, routers)

                if visualize_:
                    """Creating a list of edges to add and then removing them after visualization"""
                    edges_to_add = [(src_node, src_pe), (dest_node, dest_pe)] + [
                        (link_node_index, router) for router in list_router_link_connection]
                    graph_for_vis.add_edges_from(edges_to_add)
                    graph_for_vis.add_node(
                        link_node_index, pos=link_pos, type='link')

                    self.visualize_network_3d(graph_for_vis, full_screen_=True)

                    graph_for_vis.remove_edges_from(edges_to_add)
                    graph_for_vis.remove_node(link_node_index)

                """Incrementing the link node index for the next link node"""
                link_node_index += 1

        return graph

    def generate_network_graph(self) -> nx.Graph:

        G = nx.Graph()

        """ Adding the nodes with their coordinates and type """
        for node, coordinates in self.router.items():
            G.add_node(node, pos=coordinates, type='router')

        for node, coordinates in self.processing_element.items():
            G.add_node(node, pos=coordinates, type='pe')

        """ Adding Edges (Mesh Topology)
        Maybe there's a better way to do this. But its not real time critical.
        Only used for visualization"""
        for src_node, src_coordinates in self.router.items():
            """ coordinates -> (x, y, z)"""
            for dest_node, dest_coordinates in self.router.items():
                euc_dist = ((src_coordinates[0] - dest_coordinates[0]) **
                            2 + (src_coordinates[1] - dest_coordinates[1])**2)**0.5
                if euc_dist == 1:
                    G.add_edge(src_node, dest_node)

        """ Adding Edges between PE and Router"""
        for src_node, src_coordinates in self.processing_element.items():
            src_x, src_y, src_z = src_coordinates
            for dest_node, dest_coordinates in self.router.items():
                dest_x, dest_y, dest_z = dest_coordinates
                if src_x == dest_x and src_y == dest_y:
                    G.add_edge(src_node, dest_node)

        return G

    def create_rename_map(self, dag_, map_) -> tuple:
        """ Renaming the nodes of the dag_graph
        - Renames the position dictionary accordingly
        - Renames mapping dictionary accordingly
        Why do we do this?
        check dag_on_network() comments for more info
        """
        dag_graph = dag_.graph
        dag_position = dag_.position

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
                rename_dict[node] = last_pe_index + int(node) + 1

        new_pos = {}
        for old_key, value in dag_position.items():
            new_key = rename_dict[old_key]
            new_pos[new_key] = value

        for task, pe in map_.items():
            if task == 0:
                updated_node = rename_dict['Start']
            elif task == num_of_nodes - 1:
                updated_node = rename_dict['Exit']
            else:
                updated_node = rename_dict[str(task)]

            new_map[updated_node] = pe

        return rename_dict, new_pos, new_map

    def visualize_network_3d(self, graph_=None, full_screen_=False) -> None:

        if graph_ is None:
            G = self.network_graph
        else:
            G = graph_

        pos = nx.get_node_attributes(G, 'pos')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        """Plotting Nodes"""
        node_colors = {  # (r,g,b) 1 -> light, 0 -> dark
            'router': 'b',
            'pe': lambda num_connections: 'r' if num_connections > 1 else (0.4, 0, 0),
            'task': (0, 0.5, 0),
            'link': (0, 1, 0),
            'start_task': (0, 0.8, 0),
            'end_task': (0, 0.8, 0),
        }

        for node, coordinates in pos.items():
            node_type = G.nodes[node]['type']
            num_node_connection = len(G.edges(node))

            if node_type not in node_colors:
                raise NotImplementedError(
                    "Node type not supported in visualization_network_3d()")

            color = node_colors[node_type]

            if callable(color):
                """For PE nodes, color is based on the number of connections"""
                color = color(num_node_connection)

            ax.scatter(*coordinates, color=color)
            ax.text(*coordinates, f'{node}', color='black')

        """Plotting Edges"""
        edge_colors = {
            frozenset(['task', 'task']): 'g',
            frozenset(['start_task', 'task']): 'g',
            frozenset(['end_task', 'task']): 'g',
            frozenset(['router', 'pe']): '0.75',
            frozenset(['router', 'router']): '0.75',
            frozenset(['task', 'pe']): 'y',
            frozenset(['end_task', 'pe']): 'y',
            frozenset(['start_task', 'pe']): 'y',
            frozenset(['link', 'router']): 'c',
        }

        for edge in G.edges():
            source_node, destination_node = edge
            source_node_type = G.nodes[source_node]['type']

            destination_node_type = G.nodes[destination_node]['type']

            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]

            edge_type = frozenset([source_node_type, destination_node_type])
            if edge_type not in edge_colors:
                raise NotImplementedError(
                    f"Edge type not supported in visualization_network_3d() "
                    f"for {source_node_type} -> {destination_node_type}")

            color = edge_colors[edge_type]
            if edge_type == frozenset(['link', 'router']):
                linestyle = ':'
                alpha = 0.5
            else:
                linestyle = '-'
                alpha = 1

            ax.plot(x, y, z, color=color, linestyle=linestyle, alpha=alpha)

        if full_screen_:
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())

        plt.draw()
        while True:
            """Close the plot on button press"""
            if plt.waitforbuttonpress(0):
                plt.close()
                break


    def init_network(self, network_) -> tuple:
        if network_ == int(4):

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
    dag_dir = f'data/task_from_graph/{index}.pickle'
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
    dag_on_network, new_map = graph.dag_on_network(data_dag, data_map)

    graph_with_link_nodes = graph.create_link_nodes(
        dag_on_network, new_map, visualize_=False)

    """Visualization of the network graph in 3D"""
    # graph.visualize_network_3d()
    # graph.visualize_network_3d(dag_on_network)
    graph.visualize_network_3d(graph_with_link_nodes)

    graph_tensor = graph.generate_tensor(
        graph_with_link_nodes, target_=data_target, debug_=True)

    """Checking the graph tensor"""
    print(f"Graph Tensor: {graph_tensor}")
    print(f"Number of nodes: {graph_tensor.num_nodes}")
    print(f"Number of edges: {graph_tensor.num_edges}")
    print(f"Number of node features: {graph_tensor.num_node_features}")
    print(f"Number of edge features: {graph_tensor.num_edge_features}")
    print(f"Node features: \n{graph_tensor.x}")

    """Checking the validity of the graph"""
    print(f"\nGraphs is directed: {graph_tensor.is_directed()}")
    print(f"Graph has self loops: {graph_tensor.has_self_loops()}")
    print(f"Graph has isolated nodes: {graph_tensor.has_isolated_nodes()}")
    print(f"Graph is valid: {graph_tensor.validate()}")

    """Checking the adjacency matrix"""
    # from torch_geometric.utils import to_dense_adj
    # print(f"Adjacency matrix: {to_dense_adj(graph_tensor.edge_index).squeeze(0)}")
