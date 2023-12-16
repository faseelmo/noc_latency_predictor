"""
 Code forked and modified from https://github.com/Livioni/DAG_Generator 
"""
import random, math 
import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx 
import sys

class DAG:
    def __init__(self, nodes, max_out = 3, alpha = 1.0, beta = 1.0, withDemand=False,  isLowDemand=False, isMediumDemand=False, isHighDemand=False, withDuration=False):
        self.nodes = nodes 
        self.max_out = max_out 
        self.alpha = alpha  
        self.beta = beta
        self.duration = []
        self.withDemand = withDemand
        self.withDuration = withDuration
        self.demand = []
        self.position = {'Start':(0,4),'Exit':(10,4)}

        if withDemand:
            self.getValidDemand(isLowDemand, isMediumDemand, isHighDemand)

        self.generator()

    
    def getValidDemand(self, isLowDemand, isMediumDemand, isHighDemand):
        
        all_demand_true = (isLowDemand and isMediumDemand and isHighDemand)

        #Condition to check if only one boolean is true. ^ -> XOR
        if (isLowDemand ^ isMediumDemand ^ isHighDemand) and not all_demand_true:
            if isLowDemand:
                start_number, end_number = 1, 100
            elif isMediumDemand:
                start_number, end_number = 100, 1000
            elif isHighDemand:
                start_number, end_number = 1000, 3000

        elif not all_demand_true: # Condition where no demand criteria is mentioned
            start_number, end_number = 1, 30
        else: 
            raise Exception("Multiple Demands True.")

        self.demand_range = (start_number, end_number)


    def getRandomDemand(self, low_num, high_num):
        mu = low_num + ((high_num - low_num) / 2 )
        return abs(np.random.normal(mu, mu/4))


    def generator(self):
        '''Randomly generates a DAG task and randomly assigns its duration and (CPU, Memory) requirements'''
        prob = 1
        t = 10  # s   time unit
        r = 100  # resource unit
        self.DAGs_generate()

        # Initialization Duration
        for i in range(len(self.into_degree)):
            if self.withDuration:
                if random.random() < prob:
                    # duration.append(random.uniform(t,3*t))
                    self.duration.append(random.sample(range(0, 3 * t), 1)[0])
                else:
                    # duration.append(random.uniform(5*t,10*t))
                    self.duration.append(random.sample(range(5 * t, 10 * t), 1)[0])
            else: 
                self.duration.append(-1)

        # Initialize resource requirements
        for i in range(len(self.into_degree)):
            if self.withDemand:
                if random.random() < 0.5:
                    cpu_demand = random.uniform(self.demand_range[0], self.demand_range[1])
                    mem_demand = random.uniform(0.05 * r, 0.01 * r)
                    self.demand.append((cpu_demand, mem_demand))
                else:
                    cpu_demand = random.uniform(self.demand_range[0], self.demand_range[1])
                    mem_demand = random.uniform(0.25 * r, 0.5 * r)
                    self.demand.append((cpu_demand, mem_demand))
            else: 
                self.demand.append((1, 1))

    
    def DAGs_generate(self):

        n = self.nodes 
        max_out = self.max_out
        alpha = self.alpha
        beta = self.beta

        length = math.floor(math.sqrt(n)/alpha)
        mean_value = n/length
        random_num = np.random.normal(loc = mean_value, scale = beta,  size = (length,1))    
        ###############################################division#############################################
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

        ############################################link#####################################################
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


        ######################################create start node and exit node################################
        for node,id in enumerate(into_degree):# Add entry nodes as father to all nodes with no entry edges
            if id ==0:
                edges.append(('Start',node+1))
                into_degree[node]+=1

        for node,od in enumerate(out_degree):# Add exit nodes as sons to all nodes with no exit edges
            if od ==0:
                edges.append((node+1,'Exit'))
                out_degree[node]+=1

        #############################################plot###################################################
        self.edges = edges
        self.into_degree = into_degree
        self.out_degree = out_degree

    def plot_DAG(self, g1, position):
        # To do: Add labels to Nodes
        # print(f"Position is {position}")
        # print(f"Nodes is {g1.nodes}")
        nx.draw_networkx(g1, arrows=True, pos=position)
        plt.savefig("DAG.png", format="PNG")
        return plt.clf

    def getInfo(self):
        print(f"\nNo.of Nodes: {self.nodes}, No.of edges = {len(self.edges)}\n")
        for i in range(self.nodes):
            print(f"\nNode: {i+1} \nProcessing: Duration = {self.duration[i]}, \nRequire: CPU = {self.demand[i][0]}, Memory = {self.demand[i][1]}")
        print("\nEdges")
        print(self.edges)
        print(f"\nInto Degree: {self.into_degree}")
        print(f"Out Degree: {self.out_degree}")
        print(f"Max Out: {self.max_out}")
        print("\nDemand")
        print(self.demand)

    def getGraph(self):
        g1 = nx.DiGraph()
        g1.add_edges_from(self.edges)
        return g1

