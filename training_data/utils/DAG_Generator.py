"""
 graph_generator() from https://github.com/Livioni/DAG_Generator
"""
import random, math 
import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx 

seed = 1
random.seed(seed)
np.random.seed(seed)

class DAG:

    def __init__(self, nodes, max_out = 3, alpha = 1.0, beta = 1.0, demand_range=(1,10)):
        self.num_nodes = nodes  # Number of Non-Start and Non-Exit nodes
        self.max_out = max_out 
        self.alpha = alpha  
        self.beta = beta
        self.position = {'Start':(0,4),'Exit':(10,4)}

        graph = self.graph_generator()

        low, high = demand_range
        self.add_demand(graph, low, high)
        self.add_delay(graph, low, high)

        self.graph = graph
        self.edge_attr = nx.get_edge_attributes(graph, 'demand')
        self.node_attr = nx.get_node_attributes(graph, 'delay')

    def add_demand(self, graph, low=1, high=100): 
        for (src, dest) in graph.edges:
            graph[src][dest]['demand'] = random.randint(low,high)

    def add_delay(self, graph, low=1, high=100): 

        for node in graph.nodes:
            if node == 'Start' or node == 'Exit':
                graph.nodes[node]['delay'] = 1
            else: 
                graph.nodes[node]['delay'] = random.randint(low, high)

    def graph_generator(self):

        n = self.num_nodes 
        max_out = self.max_out
        alpha = self.alpha
        beta = self.beta

        length = math.floor(math.sqrt(n)/alpha)
        mean_value = n/length
        random_num = np.random.normal(loc = mean_value, scale = beta,  size = (length,1))    

        # Division
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

def test(): 
    dag = DAG(nodes=5)
    node_labels = False
    nx.draw(dag.graph,  pos=dag.position, with_labels=node_labels,  font_weight='bold', node_color='skyblue', edge_color='gray', node_size=800)
    print(f"Edge Attributes: \n{dag.edge_attr} ")
    print(f"Node Attributes: \n{dag.node_attr} ")
    nx.draw_networkx_edge_labels(dag.graph, pos=dag.position, edge_labels=dag.edge_attr)
    if not node_labels: 
        nx.draw_networkx_labels(dag.graph, pos=dag.position, labels=dag.node_attr)  
    plt.show()

# test()