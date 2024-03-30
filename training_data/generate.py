import os 
import sys
import shutil
import argparse

from utils.Generator import Generator
    
parser = argparse.ArgumentParser(
    description='Generate a random task that can be mapped to a NoC ' 
    'and estimate then save latency results'
)
parser.add_argument(
    '--gen', 
    action='store_true', 
    help='Generate all combinations of data'
)
parser.add_argument(
    '--gen_from_graph', 
    action='store_true',
    help='Use already generated graph from /data/non_iso_graph'
)
parser.add_argument(
    '--tasks_num', 
    type=int, 
    default=None, 
    help='Number of tasks excluding Start and Exit'
)
parser.add_argument(
    '--sim_count', 
    type=int, 
    default=0, 
    help='saving results starts from this arg'
)
parser.add_argument(
    '--maps_per_task', 
    type=int, 
    default=1, 
    help='Number of mapping done per task'
)
parser.add_argument(
    '--low_range', 
    type=int, 
    default=1, 
    help='Low limit for Demand and Delay'
)
parser.add_argument(
    '--high_range', 
    type=int, 
    default=100, 
    help='High Limit for Demand and Delay'
)
parser.add_argument(
    '--iteration', 
    type=int, 
    default=1, 
    help='Number of Iteration of data generation'
)


args = parser.parse_args()

NETWORK_SIZE = 4

all_gen = args.gen
gen_from_graph = args.gen_from_graph
tasks = args.tasks_num
maps = args.maps_per_task
sim_count = args.sim_count
low_demand = args.low_range
high_demand = args.high_range
sim_iteration = args.iteration

# result_path = 'data/task_' + str(tasks)
result_path = 'data/task_from_graph'
network_config_path = 'ratatoskr/config/'
network_file_path = 'ratatoskr/config/misc/networks/' 
if NETWORK_SIZE == 4:
    network_file_name = '4x4_Mesh.xml' 
elif NETWORK_SIZE == 2:
    network_file_name = '2x2_Mesh.xml' 
elif NETWORK_SIZE == 3:
    network_file_name = '3x3_Mesh.xml' 
else: 
    print("[Warning] Not a Valid Network Choice")
    sys.exit()

shutil.copy(
    network_file_path + network_file_name, 
    network_config_path + network_file_name
)
os.rename(
    network_config_path + network_file_name, 
    network_config_path + 'network.xml'
) # This deletes the original file

data_generator = Generator(
    result_path=result_path, 
    num_of_tasks=tasks, 
    demand_range=(low_demand, high_demand),
    network_mesh_size=NETWORK_SIZE,
    maps_per_task=maps, 
    sim_count=sim_count
)

first_iter_flag = True

if all_gen:
    for i in range(sim_iteration):
        print(f"Iteartion {i+1}")
        data_generator.generate_all_dag()
elif gen_from_graph:
    print('Usings graphs from /data/non_iso_graphs')
    for i in range(sim_iteration):
        print(f"Iteartion {i+1}")
        data_generator.generate_from_graph()
    
else: 
    data_generator.generate()

"""
    Argument list for Generator Class 

    num_of_tasks    : Number of random tasks that will be generated in the data.xml
    mesh_size       : 2D Mesh size of the Network 
    maps_per_task   : Number of mappings done per generated task file (data.xml)
    sim_count       : Default is 0. Only use when you'd like to appends the same dataset 
    result_path     : Path where the generated data will be stored. If specified path is not
                      valid, then it will create the path. 
"""


