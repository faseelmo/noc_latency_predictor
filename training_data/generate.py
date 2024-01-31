import argparse
from utils.Generator import Generator
    
parser = argparse.ArgumentParser(description='Generate a random task that can be mapped to a NoC and estimate then save latency results')
parser.add_argument('--gen', action='store_true', help='Generate all combinations of data')
parser.add_argument('--tasksNum', type=int, default=4, help='Number of tasks excluding Start and Exit')
parser.add_argument('--simCount', type=int, default=0, help='saving results starts from this arg')
parser.add_argument('--mapsPerTask', type=int, default=1, help='Number of mapping done per task')
parser.add_argument('--low_range', type=int, default=1, help='Low limit for Demand and Delay')
parser.add_argument('--high_range', type=int, default=100, help='High Limit for Demand and Delay')
parser.add_argument('--iteration', type=int, default=1, help='Number of Iteration of data generation')

args = parser.parse_args()

NETWORK_SIZE = 4

all_gen = args.gen
tasks = args.tasksNum
maps = args.mapsPerTask
sim_count = args.simCount
low_demand = args.low_range
high_demand = args.high_range
sim_iteration = args.iteration
result_path = 'data/task_' + str(tasks)

data_generator = Generator(
    result_path=result_path, 
    num_of_tasks=tasks, 
    demand_range=(low_demand, high_demand),
    network_mesh_size=NETWORK_SIZE,
    maps_per_task=maps, 
    sim_count=sim_count
)

total_simulation_count = data_generator.total_sim_count 

if all_gen:
    print(f"Total datapoint that'll be generated: {total_simulation_count*sim_iteration}")
    data_generator.generate_all_dag()
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


