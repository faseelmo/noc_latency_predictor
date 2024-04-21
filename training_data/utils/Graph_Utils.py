from bidict import bidict
import networkx as nx
import matplotlib.pyplot as plt
import copy


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

    def dag_on_network(self, dag, map) -> nx.Graph:
        """ type(dag) -> DAG() """

        """ Need to rename otherwise network graph
        and dag graph will have same node names"""
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

        for node in dag_graph.nodes():
            for neighbor in dag_graph.neighbors(node):
                network_graph.add_edge(node, neighbor)

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
        Renames the position dictionary accordingly
        Renames mapping dictionary accordingly"""

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

        # Set the limits of the axes
        # ax.set_xlim([0, 15])
        # ax.set_ylim([0, 15])
        # ax.set_zlim([0, 2])

        for node, coordinates in pos.items():
            node_type = G.nodes[node]['type']
            if node_type == 'router':
                ax.scatter(*coordinates, color='b')
            elif node_type == 'pe':
                ax.scatter(*coordinates, color='r')
            elif node_type == 'task':
                ax.scatter(*coordinates, color='g')
            else:
                raise NotImplementedError(
                    "Node type not supported in visualization_network_3d()")

        # Draw edges
        for edge in G.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, color='k')

        plt.show()
        # plt.waitforbuttonpress(0)
        # plt.close()

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

    def visualize_dag_3d(self, graph):
        print("Visualizing DAG in 3D")


if __name__ == '__main__':
    import pickle
    dag_dir = 'data/task_7/1.pickle'
    dag = pickle.load(open(dag_dir, 'rb'))

    data_network = dag['network']
    data_dag = dag['task_dag']
    data_dag.plot(show_node_attrib=False)
    data_map = dag['map']

    graph = GraphUtils(data_network)
    # graph.visualize_network_3d()
    dag_on_network = graph.dag_on_network(data_dag, data_map)
    graph.visualize_network_3d(dag_on_network)
