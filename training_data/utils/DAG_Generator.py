import random, math 
import numpy as np 
import uuid
import json 
import matplotlib.pyplot as plt 
import networkx as nx 

# seed = 1
# random.seed(seed)
# np.random.seed(seed)

"""
1.  
    If initialized with a graph, new graph will not be generated. 
    nodes -> Tasks excluding Start and Exit Node 
    graph -> path to the .edgelist file 
    sepecifiying the demand rage dosent affect the 'Start' and 'Exit' Node 

2. 
    alpha controls the depth of the DAG, 
        length = math.sqrt(num_of_nodes)/alpha
        smaller alpha -> thinner and longer DAG
        bigger alphs -> fatter and denser DAG

    beta controls the width of the DAG, 
        width randomly set to normal distribution 
        mean = num_of_nodes/length 
        std = beta 
        larger the beta, the more irrgular the graph. 

    Parameterization: 
        max_out = [1,2,3,4,5]       #max out_degree of one node
        alpha = [0.5,1.0,1.5]       #DAG shape
        beta = [0.0,0.5,1.0,2.0]    #DAG regularity

source: https://github.com/Livioni/DAG_Generator

"""
class DAG:

    def __init__(
        self, 
        nodes=None, 
        graph_path=None, 
        graph_pos_path=None, 
        max_out = 3, 
        alpha = 1.0, 
        beta = 1.0, 
        demand_range=(1,10)
    ):

        self.num_nodes = nodes  # Number of Non-Start and Non-Exit nodes
        self.max_out = max_out 
        self.alpha = alpha  
        self.beta = beta
        self.position = {'Start':(0,4),'Exit':(10,4)}

        if graph_path is None:
            assert self.num_nodes is not None, "Number of nodes should be specified"
            self.graph = self.graph_generator()

        else: 
            self.graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph())
            self.max_out = None 
            self.alpha = None
            self.beta = None
            self.num_nodes = self.graph.number_of_nodes() - 2
            if graph_pos_path is not None: 
                self.position = json.load(open(graph_pos_path))
            else: 
                self.position = nx.circular_layout(self.graph)

        low, high = demand_range
        self.add_demand(low, high)
        self.add_delay(low, high)
        self.id = str(uuid.uuid4())

        self.edge_attr = nx.get_edge_attributes(self.graph, 'demand')
        self.node_attr = nx.get_node_attributes(self.graph, 'delay')

    def add_demand(self, low=1, high=100) -> None: 
        for (src, dest) in self.graph.edges:
            self.graph[src][dest]['demand'] = random.randint(low,high)

    def add_delay(self, low=1, high=100) -> None: 
        for node in self.graph.nodes:
            if node == 'Start' or node == 'Exit':
                self.graph.nodes[node]['delay'] = 1
            else: 
                self.graph.nodes[node]['delay'] = random.randint(low, high)

    def graph_generator(self) -> nx.DiGraph:
        # Function from  https://github.com/Livioni/DAG_Generator

        n = self.num_nodes 
        max_out = self.max_out
        alpha = self.alpha
        beta = self.beta

        length = math.floor(math.sqrt(n)/alpha)
        mean_value = n/length
        random_num = np.random.normal(loc = mean_value, scale = beta,  size = (length,1))    

        generate_num = 0
        dag_num = 1
        dag_list = [] 
        for i in range(len(random_num)):
            dag_list.append([]) 
            for j in range(math.ceil(random_num[i])):
                dag_list[i].append(j)
            generate_num += len(dag_list[i])

        if generate_num != n:
            if generate_num<n:
                for i in range(n-generate_num):
                    index = random.randrange(0,length,1)
                    dag_list[index].append(len(dag_list[index]))
            if generate_num>n:
                i = 0
                while i < generate_num-n:
                    index = random.randrange(0,length,1)
                    if len(dag_list[index])<=1:
                        continue
                    else:
                        del dag_list[index][-1]
                        i += 1

        dag_list_update = []
        pos = 1
        max_pos = 0
        for i in range(length):
            dag_list_update.append(list(range(dag_num,dag_num+len(dag_list[i]))))
            dag_num += len(dag_list_update[i])
            pos = 1
            for j in dag_list_update[i]:
                self.position[j] = (3*(i+1),pos)
                pos += 5
            max_pos = pos if pos > max_pos else max_pos
            self.position['Start']=(0,max_pos/2)
            self.position['Exit']=(3*(length+1),max_pos/2)

        # Link
        into_degree = [0]*n            
        out_degree = [0]*n             
        edges = []                          
        pred = 0

        for i in range(length-1):
            sample_list = list(range(len(dag_list_update[i+1])))
            for j in range(len(dag_list_update[i])):
                od = random.randrange(1,max_out+1,1)
                od = len(dag_list_update[i+1]) if len(dag_list_update[i+1])<od else od
                bridge = random.sample(sample_list,od)
                for k in bridge:
                    edges.append((dag_list_update[i][j],dag_list_update[i+1][k]))
                    into_degree[pred+len(dag_list_update[i])+k]+=1
                    out_degree[pred+j]+=1 
            pred += len(dag_list_update[i])

        # Create start node and exit node
        for node,id in enumerate(into_degree):# Add entry nodes as father to all nodes with no entry edges
            if id ==0:
                edges.append(('Start',node+1))
                into_degree[node]+=1

        for node,od in enumerate(out_degree):# Add exit nodes as sons to all nodes with no exit edges
            if od ==0:
                edges.append((node+1,'Exit'))
                out_degree[node]+=1

        graph = nx.DiGraph()
        graph.add_edges_from(edges)

        return graph 

    def plot(self, show_node_attrib=True) -> None:
        node_labels = not show_node_attrib
        nx.draw(self.graph,  pos=self.position, with_labels=node_labels,  font_weight='bold', node_color='skyblue', edge_color='gray', node_size=800)
        nx.draw_networkx_edge_labels(self.graph, pos=self.position, edge_labels=self.edge_attr)
        if show_node_attrib:
            nx.draw_networkx_labels(self.graph, pos=self.position, labels=self.node_attr)  
        plt.show()

    def is_isomorphic(self, graph) -> bool:
        return nx.is_isomorphic(self.graph, graph)

if __name__ == "__main__": 
    # dag = DAG(nodes=5, alpha=0.5)
    dag = DAG(
        graph_path='data/non_iso_graphs/graphs/500.edgelist', 
        graph_pos_path='data/non_iso_graphs/positions/500.json', 
        demand_range=(10,10))
    print(f"Position is: \n{dag.position} ")
    print(f"Edge Attributes: \n{dag.edge_attr} ")
    print(f"Node Attributes: \n{dag.node_attr} ")
    print(f"Params {dag.max_out, dag.alpha, dag.beta}")
    print(f"Number of node (non exit and start) {dag.num_nodes}")
    dag.plot(show_node_attrib=False)



