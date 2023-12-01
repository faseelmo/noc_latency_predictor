import argparse
import subprocess
import pandas as pd

from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG
from utils.Map_Generator import MapGenerator

parser = argparse.ArgumentParser(description='Creates random tasks along with the required congfig files to run on ratatoskr')
parser.add_argument('--tasks', type=int, default=2, help='Number of tasks (excluding Start and Exit Task)')
parser.add_argument('--runSim', type=bool, default=False, help='If True, the script will also run the ratatoskr simulation')

args = parser.parse_args()

num_of_nodes = args.tasks
run_sim = args.runSim


network = '4' # Select '4' for 4x4 Mesh and '2' for 2x2 Mesh
max_out = 3
alpha = 1.0
beta = 1.0

"""
    Generate a random Direct Acyclic Graph (DAG)
    set_max_out = [1,2,3,4,5]                            #max out_degree of one node
    set_alpha = [0.5,1.0,1.5]                            #DAG shape
    set_beta = [0.0,0.5,1.0,2.0]                         #DAG regularity
"""
dag = DAG(num_of_nodes, max_out, alpha, beta, 
          withDemand=False, withDuration=False) #withDuration sim dosent work
task_graph = dag.getGraph()
# dag.getInfo()


"""
    Convert DAG to Tasks (data.xml) for ratatoskr Simulator
"""
sim_path = 'ratatoskr/config/'
task = TaskGenerator(dag, sim_path + 'data.xml')


"""
    Assign the tasks to nodes (Processing Elements in Network).
    The constructor creates the map.xml file in /config. 
    It also copies the right network.xml file from /misc to 
    the config folder. 
"""
mapper = MapGenerator(task.num_of_tasks, network, sim_path + 'map.xml' )
rename_dict, new_pos = mapper.getRenameDict(dag.position)
map_graph = mapper.doGraphRemapping(task_graph, rename_dict)
mapper.plotTaskAndMap(task_graph, dag.position, map_graph, new_pos)


"""
    Automating The Pipeline
"""
sim_successfull_flag = False 
successfull_string = '[ProcessingElementVC:startSending]  Node' + str(rename_dict['Exit'])

if run_sim:
    command = "cd ratatoskr/ && ./sim"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end=' ')
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
    mapper.plotTaskAndMap(task_graph, dag.position, map_graph, new_pos, latency_list)




