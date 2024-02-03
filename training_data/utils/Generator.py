import os
import pickle
import subprocess

from .TASK_Generator import TaskGenerator
from .DAG_Generator import DAG
from .Map_Generator import MapGenerator

class Generator:
    def __init__(self, result_path='', num_of_tasks=4, demand_range=(1,100), network_mesh_size=4, maps_per_task=1, sim_count=0):
        
        self.max_out_list = [1,2,3,4,5]     # Max out_degree of one node
        self.alpha_list = [0.5,1.0,1.5]     # DAG shape
        self.beta_list  = [0.0,0.5,1.0,2.0] # DAG regularity
        self.num_of_task = num_of_tasks     # This is excluding the Start and Exit Node

        self.result_path = result_path
        self.sim_count = sim_count
        
        self.demand_range = demand_range
        self.network = network_mesh_size
        self.maps_per_task = maps_per_task
        self.total_sim_count = len(self.max_out_list) * len(self.alpha_list) * len(self.beta_list) * maps_per_task

        self.checkResultPath()

    def generate_all_dag(self):
        for max_out in self.max_out_list:
            print(f"    Simulating for max out: {max_out}")
            for alpha in self.alpha_list:
                print(f"     Simulating for alpha: {alpha}")
                for beta in self.beta_list:
                    print(f"      Simulating for beta: {beta}")
                    self.generate(dag_param=(max_out, alpha, beta))

    def generate(self, dag_param=(3, 1.0, 1.0)):
        max_out, alpha, beta = dag_param
        dag = DAG(
            nodes=self.num_of_task,
            max_out=max_out, 
            alpha=alpha,
            beta=beta, 
            demand_range=self.demand_range
        )
        TaskGenerator(dag, 'ratatoskr/config/data.xml')     # Creates the relevant data.xml file in config dir
        for i in range(self.maps_per_task):                 # For multiple maps per task
            mapper = MapGenerator(
                dag=dag, 
                network=self.network, 
                file_location='ratatoskr/config/map.xml'
            )                                               # Assigns a random map for the dag and creates the map.xml file in config dir

            sim_results = self.doSim(mapper, showSimOutput=False)
            if sim_results['sim_successfull_flag'] is False: print("[Warning] Sim not Successfull")
            else: print(f"      [{i+1}/{self.maps_per_task}] Latency = {sim_results['network_processing_time']} ")
            self.saveResults(dag, mapper.map, sim_results)

        print(f" All Mapping Combination done\n")

    def doSim(self, mapper, showSimOutput=False): 
        sim_successfull_flag = False 
        last_node = mapper.map[self.num_of_task+1]
        successfull_string = '[ProcessingElementVC:startSending]  Node' + str(last_node)
        command = "cd ratatoskr/ && ./sim"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        processing_time_string = 'Node' + str(last_node) 
        for line in process.stdout:
            if showSimOutput:
                print(line, end=' ')
            if successfull_string in line:
                sim_successfull_flag = True
            if processing_time_string in line and 'Receive Flit' in line: 
                totat_processing_time = line.split()[0][:-3]
            if 'Execution time' in line: 
                start_index = line.index(':')
                sim_time_string = line[start_index+2:].strip()
            if 'Lost Packets' in line: 
                start_index = line.index(':')
                lost_packets = line[start_index+2:].strip()
            if 'Average flit latency' in line: 
                start_index = line.index(':')
                avg_flit_lat = line[start_index+2:-4].strip()
            if 'Average packet latency' in line: 
                start_index = line.index(':')
                avg_packet_lat = line[start_index+2:-4].strip()
            if 'Average network latency' in line: 
                start_index = line.index(':')
                avg_network_lat = line[start_index+2:-4].strip()

        sim_result = {
            'sim_successfull_flag'      : sim_successfull_flag, 
            'network_processing_time'   : totat_processing_time, 
            'sim_exec_time'             : sim_time_string, 
            'lost_packets'              : lost_packets, 
            'avg_flit_lat'              : avg_flit_lat, 
            'avg_packet_lat'            : avg_packet_lat, 
            'avg_network_lat'           : avg_network_lat
        }

        return sim_result

    def saveResults(self, dag, map, sim_result):
        result = {
            'task_dag'  : dag, 
            'network'   : self.network, 
            'map'       : map, 
        }

        result.update(sim_result) # append the result dict with sim_results

        self.sim_count += 1
        file_name = str(self.sim_count) + '.pickle'
        file_path = os.path.join(self.result_path,file_name)

        with open(file_path, 'wb') as file:
            pickle.dump(result, file)


    def checkResultPath(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print(f"Directory '{self.result_path}' created successfully.")
        else:
            print(f"Directory '{self.result_path}' already exists.")
            input("Press Enter to Proceed ")

def test(): 
    gen = Generator(result_path= 'skraa', num_of_tasks=10, demand_range=(10,100))
    gen.generate_all_dag()
# test()