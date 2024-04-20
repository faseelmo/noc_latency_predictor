from bidict import bidict
import networkx as nx
import matplotlib.pyplot as plt
import copy


class GraphUtils():
    def __init__(self, network):

        self.router_z = 0
        self.processing_element_z = 0.5
        self.dag_z = 2

        processing_element, router = self.init_network(network)
        self.network_graph = self.generate_network_graph(
            processing_element, router)

    def dag_on_network(self, dag, map) -> nx.Graph:

        print(dag.position)
        dag_graph = dag.graph

        new_pos = {}
        # Adding Z coordinate to the nodes
        for node in dag.position:
            new_pos[node] = (dag.position[node][0],
                             dag.position[node][1], self.dag_z)
        
        nx.set_node_attributes(dag_graph, new_pos, 'pos')

        # pos = nx.get_node_attributes(dag_graph, 'pos')
        # type = nx.get_node_attributes(dag_graph, 'type')

        # Adding dag to network graph
        network_graph = copy.deepcopy(self.network_graph)

        for node in dag_graph.nodes():
            network_graph.add_node(node, pos=dag_graph.nodes[node]['pos'], type='dag')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        pos  = nx.get_node_attributes(network_graph, 'pos')
        for node, coordinates in pos.items():
            ax.scatter(*coordinates, color='g')

        for edge in network_graph.edges(): 
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, color='k')

        plt.show()
        

        return network_graph

    def generate_network_graph(self, processing_element, router) -> nx.Graph:

        G = nx.Graph()

        # Adding the nodes with their coordinates adn type
        for node, coordinates in router.items():
            G.add_node(node, pos=coordinates, type='router')

        for node, coordinates in processing_element.items():
            G.add_node(node, pos=coordinates, type='pe')

        # Adding Edges (Mesh Topology)
        # Maybe there's a better way to do this. But its not real time critical.
        # Only used for visualization
        for src_node, src_coordinates in router.items():
            # coordinates -> (x, y, z)
            for dest_node, dest_coordinates in router.items():
                euc_dist = ((src_coordinates[0] - dest_coordinates[0]) **
                            2 + (src_coordinates[1] - dest_coordinates[1])**2)**0.5
                if euc_dist == 1:
                    G.add_edge(src_node, dest_node)

        # Adding Edges between PE and Router
        for src_node, src_coordinates in processing_element.items():
            src_x, src_y, src_z = src_coordinates
            for dest_node, dest_coordinates in router.items():
                dest_x, dest_y, dest_z = dest_coordinates
                if src_x == dest_x and src_y == dest_y:
                    G.add_edge(src_node, dest_node)

        return G

    def visualize_network_3d(self, graph_=None) -> None:
        # 3D drawing

        if graph_ is None:
            G = self.network_graph
        else:
            G = graph_

        pos = nx.get_node_attributes(self.network_graph, 'pos')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for node, coordinates in pos.items():
            node_type = G.nodes[node]['type']
            if node_type == 'router':
                ax.scatter(*coordinates, color='b')
            elif node_type == 'pe':
                ax.scatter(*coordinates, color='r')
            elif node_type == 'task':
                ax.scatter(*coordinates, color='g')
            else: 
                raise NotImplementedError("Node type not supported in visualization_network_3d()")

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

    dag_network = dag['network']
    dag_graph = dag['task_dag']
    dag_map = dag['map']

    graph = GraphUtils(dag_network)
    # graph.visualize_network_3d()
    dag_on_network = graph.dag_on_network(dag_graph, map)
    # graph.visualize_network_3d(dag_on_network)

    print(dag_network)
    print(dag)
