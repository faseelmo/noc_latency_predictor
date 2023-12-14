import random
import subprocess
import pandas as pd
import os
import pickle


from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG
from utils.Map_Generator import MapGenerator

class Generator:
    def __init__(self, result_path='', num_of_tasks=4, mesh_size=4, maps_per_task=1,sim_count=0, run_sim=True):

        self.runsim = run_sim
        self.network = str(mesh_size)
        self.num_of_tasks = num_of_tasks 
        self.maps_per_task = maps_per_task
        self.demand_requirement = None
# 
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
        self.current_demand = None
        self.save_dict_flag = False
        self.total_sims_required = 1
        self.sim_successfull_flag = False

    def generateAllDag(self):
        if self.result_path == '':
            raise ValueError("If allDAG=True, must specify valid result_path argument")

        self.save_dict_flag = True
        self.showSimCount()
        self.checkResultPath()
        self.generate_with_demand()

    def runAllSim(self, demand=(False, False, False)):
        print("Running All Sims\n\n")
        for max_out in self.set_max_out:
            for alpha in self.set_alpha:
                for beta in self.set_beta:
                    self.generate(max_out, alpha, beta, demand) # Need to pass these arguments for DAG Generator

    def generate_with_demand(self):
        """Need to Add a for loop here for different list of demands.
            Low List, Medium List, High List
        """
        demand_all_none = all(element is None for element in self.demand_requirement)

        if demand_all_none:
            print("Without Sim Demand Requirement")
            self.runAllSim()

        if self.demand_requirement is not None:
            print("With Demand Requirement")
            demand_index = 0
            for demand_iter in self.demand_requirement: # Iterating for each demand
                if demand_iter != None:
                    for i in range(demand_iter):
                        self.current_demand = demand_index
                        if demand_index == 0 : self.runAllSim(demand=(True, False, False))
                        elif demand_index == 1 : self.runAllSim(demand=(False, True, False))
                        elif demand_index == 2 : self.runAllSim(demand=(False, False, True))

                demand_index += 1

        
    def generate(self, max_out, alpha, beta, demand=(False, False, False)):
        self.generateDAGandTask(max_out, alpha, beta, demand) # arg for DAG Generator
        """Support for Multiple Random Mapping for a single Task"""
        self.map_count = 0
        for i in range(self.maps_per_task):
            self.map_count += 1
            self.singleRandomMappingAndSim()

    def generateDAGandTask(self, max_out, alpha, beta, demand):
        self.dag = DAG(self.num_of_tasks, max_out, alpha, beta, withDemand=True, 
                       isLowDemand=demand[0], isMediumDemand=demand[1], isHighDemand=demand[2]) 
        self.task = TaskGenerator(self.dag, self.sim_path + 'data.xml')
    
    def singleRandomMappingAndSim(self):
        self.mapper= MapGenerator(self.task.num_of_tasks, self.network, self.sim_path + 'map.xml' )
        self.mapper.plotTaskAndMap(self.task.task_graph, self.dag.position)

        if self.runsim:
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
        result["task_graph_pos"] = self.dag.position
        result["map_graph"] = self.mapper.map_graph
        result["map_graph"] = self.mapper.position
        result["demand"] = self.task.demand
        result["duration"] = self.task.duration
        result["task_num"] = self.task.num_of_tasks
        result["sim_successful"] = self.sim_successfull_flag
        result['avg_flit_lat'] = self.latency_list[0]
        result['avg_packet_lat'] = self.latency_list[1]
        result['avg_network_lat'] = self.latency_list[2]
        result['demand_level'] = self.current_demand

        self.sim_count += 1

        print(f"[{self.sim_count}/{self.total_sims_required}] Demand_Level: {self.current_demand}, Task_Param: {(self.dag.max_out, self.dag.alpha,self.dag.beta)}, Map_Count: {self.map_count}, Sim_Successful: {self.sim_successfull_flag}, Latency:{self.latency_list[1]} , Sim_Time: {self.sim_time_string} ")

        if not self.save_dict_flag:
            print("Not Saving Results")
            return

        file_name = str(self.sim_count) + '.pickle'
        file_path = os.path.join(self.result_path,file_name)

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

    def showSimCount(self):
        total_param_count = len(self.set_alpha) * len(self.set_beta) * len(self.set_max_out) * self.maps_per_task
        total_demand = (self.demand_requirement[0] or 0) + (self.demand_requirement[1] or 0) + (self.demand_requirement[2] or 0)

        if total_demand == 0: total_demand = 1
        self.total_sims_required = total_param_count * total_demand        

        print(f"Total Number of Simulation Required: {self.total_sims_required}")

