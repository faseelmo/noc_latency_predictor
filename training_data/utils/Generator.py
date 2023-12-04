import random
import subprocess
import pandas as pd

from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG
from utils.Map_Generator import MapGenerator

class Generator:
    def __init__(self, num_of_tasks=4, meshSize=4, maps_per_task=1, runsim=False, randomDAG=True, allDAG=False):
        self.num_of_tasks = num_of_tasks 
        self.runsim = runsim
        self.randomDAG = randomDAG
        self.sim_path = 'ratatoskr/config/'
        self.network = str(meshSize)
        self.maps_per_task = maps_per_task

        self.set_max_out = [1,2,3,4,5]      #max out_degree of one node
        self.set_alpha = [0.5,1.0,1.5]      #DAG shape
        self.set_beta = [0.0,0.5,1.0,2.0]   #DAG regularity

        self.task = None
        self.mapper = None

        if allDAG:
            self.runAllSim()

    def runAllSim(self):
        for max_out in self.set_max_out:
            for alpha in self.set_alpha:
                for beta in self.set_beta:
                    self.generate(max_out, alpha, beta)

    def saveResults(self, max_out, alpha, beta):
        """
        Data Type could be a dictionary? 
        One folder could then contain all the dictionaries? 

        Things to Save 
        self.task.task_graph
        self.mapper.map_graph
        self.task.demand
        self.task.duration
        max_out, alpha, beta 
        num_of_tasks

        latency 
        """
        return 



    def generate(self, max_out, alpha, beta):
        dag  = self.generateDAGandTask(max_out, alpha, beta)
        """Support for Multiple Random Mapping for a single Task"""
        for i in range(self.maps_per_task):
            print(f"Mapping Iteration is {i}")
            self.singleMappingAndSim(self.task.num_of_tasks, self.task.task_graph, dag.position)

    def generateDAGandTask(self, max_out, alpha, beta):
        dag = DAG(self.num_of_tasks, max_out, alpha, beta, withDemand=False, withDuration=False) #withDuration sim dosent work
        self.task = TaskGenerator(dag, self.sim_path + 'data.xml')
        return dag
    
    def singleMappingAndSim(self, num_of_tasks, task_graph, dag_position):
        self.mapper= MapGenerator(num_of_tasks, self.network, self.sim_path + 'map.xml' )
        self.mapper.plotTaskAndMap(task_graph, dag_position)
        if self.runsim:
            print("Running Simulator")
            self.doSim(dag_position, showSimOutput=True)

    def getRandomDAGParameters(self, showParam=False):
        max_out = random.choice(self.set_max_out) 
        alpha = random.choice(self.set_alpha) 
        beta = random.choice(self.set_beta) 
        if showParam:
            print("\n----DAG Parameters----")
            print(f"Max Out: {max_out}, Alpha: {alpha}, Beta: {beta}\n")
        return max_out, alpha, beta

    def doSim(self,dag_position, showSimOutput=False): 

        sim_successfull_flag = False 
        last_node = self.mapper.map[-1][1]
        successfull_string = '[ProcessingElementVC:startSending]  Node' + str(last_node)
        command = "cd ratatoskr/ && ./sim"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        with open('sim_output.txt', 'w') as file:
            for line in process.stdout:
                if showSimOutput:
                    print(line, end=' ')
                file.write(line + ' ')
                if successfull_string in line:
                    sim_successfull_flag = True

        if sim_successfull_flag:
            print("\n----Sim Successfull----")
            result = pd.read_csv('ratatoskr/results/report_Performance.csv', header=None)
            result_values = result.values.tolist()

            avg_flit_lat = result_values[0][1]
            avg_packet_lat = result_values[1][1]
            avg_network_lat = result_values[2][1]

            latency_list = [avg_flit_lat, avg_packet_lat, avg_network_lat]
            self.mapper.plotTaskAndMap(self.task.task_graph, dag_position, latency_list)


        

