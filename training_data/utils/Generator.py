import random
import subprocess
import pandas as pd


from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG
from utils.Map_Generator import MapGenerator

class Generator:
    def __init__(self, num_of_tasks=4, meshSize=4, maps_per_task=1,sim_count=0, runsim=False, randomDAG=True, allDAG=False):

        self.num_of_tasks = num_of_tasks 
        self.runsim = runsim
        self.randomDAG = randomDAG
        self.allDAG = allDAG

        self.sim_path = 'ratatoskr/config/'
        self.network = str(meshSize)
        self.maps_per_task = maps_per_task

        self.set_max_out = [1,2,3,4,5]      #max out_degree of one node
        self.set_alpha = [0.5,1.0,1.5]      #DAG shape
        self.set_beta = [0.0,0.5,1.0,2.0]   #DAG regularity
        
        self.sim_count = sim_count
        self.map_count = 0

        self.dag = None
        self.task = None
        self.mapper = None
        self.latency_list = []
        self.sim_successfull_flag = False

        if self.allDAG:
            self.runAllSim()

    def runAllSim(self):
        for max_out in self.set_max_out:
            for alpha in self.set_alpha:
                for beta in self.set_beta:
                    self.generate(max_out, alpha, beta) # Need to pass these arguments for DAG Generator

    def generate(self, max_out, alpha, beta):
        self.generateDAGandTask(max_out, alpha, beta) # arg for DAG Generator
        """Support for Multiple Random Mapping for a single Task"""
        self.map_count = 0
        for i in range(self.maps_per_task):
            self.map_count += 1
            self.singleRandomMappingAndSim()

    def generateDAGandTask(self, max_out, alpha, beta):
        self.dag = DAG(self.num_of_tasks, max_out, alpha, beta, withDemand=False, withDuration=False) #withDuration sim dosent work
        self.task = TaskGenerator(self.dag, self.sim_path + 'data.xml')
    
    def singleRandomMappingAndSim(self):
        self.mapper= MapGenerator(self.task.num_of_tasks, self.network, self.sim_path + 'map.xml' )
        self.mapper.plotTaskAndMap(self.task.task_graph, self.dag.position)

        if self.runsim:
            showSimOutput = not self.allDAG #Sim results not shown when allDAG is true
            self.doSim(showSimOutput)
            self.saveResults()


    def doSim(self, showSimOutput=False): 
        self.sim_successfull_flag = False 
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
                    self.sim_successfull_flag = True

        if self.sim_successfull_flag:
            # print("\n----Sim Successfull----")
            result = pd.read_csv('ratatoskr/results/report_Performance.csv', header=None)
            result_values = result.values.tolist()

            avg_flit_lat = result_values[0][1]
            avg_packet_lat = result_values[1][1]
            avg_network_lat = result_values[2][1]

            self.latency_list = [avg_flit_lat, avg_packet_lat, avg_network_lat]
            self.mapper.plotTaskAndMap(self.task.task_graph, self.dag.position, self.latency_list)

    def saveResults(self):
        result = {}
        result["max_out"] = self.dag.max_out
        result["alpha"] = self.dag.alpha
        result["beta"] = self.dag.beta
        result['network'] = self.network
        # result["task_graph"] = self.task.task_graph
        # result["map_graph"] = self.mapper.map_graph
        # result["demand"] = self.task.demand
        # result["duration"] = self.task.duration
        result["task_num"] = self.task.num_of_tasks
        result["sim_successful"] = self.sim_successfull_flag
        result['avg_flit_lat'] = self.latency_list[0]
        result['avg_packet_lat'] = self.latency_list[1]
        result['avg_network_lat'] = self.latency_list[2]

        self.sim_count += 1

        print(f"\n[{self.sim_count}]Task_Num: {result['task_num']}, Map_Count: {self.map_count}")
        # print(result)
        return 

    def getRandomDAGParameters(self, showParam=False):
        max_out = random.choice(self.set_max_out) 
        alpha = random.choice(self.set_alpha) 
        beta = random.choice(self.set_beta) 
        if showParam:
            print("\n----DAG Parameters----")
            print(f"Max Out: {max_out}, Alpha: {alpha}, Beta: {beta}\n")
        return max_out, alpha, beta
            
        

