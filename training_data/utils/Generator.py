import random
import subprocess
import pandas as pd
import os
import pickle


from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG
from utils.Map_Generator import MapGenerator

class Generator:
    def __init__(self, result_path='', num_of_tasks=4, mesh_size=4, maps_per_task=1,sim_count=0, runsim=False, random_dag=True, all_dag=False, save_results=False):

        self.allDAG = all_dag
        self.runsim = runsim
        self.randomDAG = random_dag
        self.network = str(mesh_size)
        self.num_of_tasks = num_of_tasks 
        self.maps_per_task = maps_per_task
        self.save_results = save_results

        self.result_path = result_path
        self.sim_path = 'ratatoskr/config/'

        self.set_max_out = [1,2,3,4,5]      #max out_degree of one node
        self.set_alpha = [0.5,1.0,1.5]      #DAG shape
        self.set_beta = [0.0,0.5,1.0,2.0]   #DAG regularity
        
        self.map_count = 0
        self.sim_count = sim_count

        self.dag = None
        self.task = None
        self.mapper = None
        self.latency_list = []
        self.sim_time_string = ''
        self.total_sims_required = 0
        self.sim_successfull_flag = False

        if self.allDAG:
            if result_path == '':
                raise ValueError("If allDAG=True, must specify valid result_path argument")
            self.showSimInfo()
            self.checkResultPath()
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
            self.doSim()
            self.saveResults()


    def doSim(self, showSimOutput=False): 
        self.sim_successfull_flag = False 
        last_node = self.mapper.map[-1][1]
        successfull_string = '[ProcessingElementVC:startSending]  Node' + str(last_node)
        command = "cd ratatoskr/ && ./sim"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        sim_output_path = 'ratatoskr/results/sim_output.txt'
        with open(sim_output_path, 'w') as file:
            for line in process.stdout:
                if showSimOutput:
                    print(line, end=' ')
                file.write(line + ' ')
                if successfull_string in line:
                    self.sim_successfull_flag = True
                if 'Execution time' in line: 
                    start_index = line.index(':')
                    self.sim_time_string = line[start_index+2:]

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
        result["task_graph"] = self.task.task_graph
        result["map_graph"] = self.mapper.map_graph
        result["demand"] = self.task.demand
        result["duration"] = self.task.duration
        result["task_num"] = self.task.num_of_tasks
        result["sim_successful"] = self.sim_successfull_flag
        result['avg_flit_lat'] = self.latency_list[0]
        result['avg_packet_lat'] = self.latency_list[1]
        result['avg_network_lat'] = self.latency_list[2]

        self.sim_count += 1

        print(f"[{self.sim_count}/{self.total_sims_required}]Task_Num: {result['task_num']}, Map_Count: {self.map_count}, Sim_Successful: {self.sim_successfull_flag}, Sim_Time: {self.sim_time_string} ")

        file_name = str(self.sim_count) + '.pickle'
        file_path = os.path.join(self.result_path,file_name)

        if self.save_results:
            with open(file_path, 'wb') as file:
                pickle.dump(result, file)

        return 

    def getRandomDAGParameters(self, showParam=False):
        max_out = random.choice(self.set_max_out) 
        alpha = random.choice(self.set_alpha) 
        beta = random.choice(self.set_beta) 
        if showParam:
            print("\n----DAG Parameters----")
            print(f"Max Out: {max_out}, Alpha: {alpha}, Beta: {beta}\n")
        return max_out, alpha, beta
            

    def checkResultPath(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print(f"Directory '{self.result_path}' created successfully.")
        else:
            print(f"Directory '{self.result_path}' already exists.")

    def showSimInfo(self):
        self.total_sims_required = len(self.set_alpha) * len(self.set_beta) * len(self.set_max_out) * self.maps_per_task        
        print(f"Total Number of Simulation Required: {self.total_sims_required}")

