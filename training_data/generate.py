import argparse
from utils.Generator import Generator
    
parser = argparse.ArgumentParser(description='Generate a random task that can be mapped to a NoC and estimate then save latency results')
parser.add_argument('--gen', action='store_true', help='Generate all combinations of data')
parser.add_argument('--tasksNum', type=int, default=4, help='Number of tasks')
parser.add_argument('--mapsPerTask', type=int, default=1, help='Number of mapping done per task')
parser.add_argument('--lowDemandCount', type=int, default=None, help='Number of Iteration for Low Demand')
parser.add_argument('--medDemandCount', type=int, default=None, help='Number of Iteration for Medium Demand')
parser.add_argument('--highDemandCount', type=int, default=None, help='Number of Iteration for High Demand')

args = parser.parse_args()

all_gen = args.gen
tasks = args.tasksNum
maps = args.mapsPerTask
user_demand_requirement = (args.lowDemandCount, args.medDemandCount, args.highDemandCount)

if all_gen:
    data_generator = Generator(result_path='data/task_5', num_of_tasks=tasks, maps_per_task=maps)
    data_generator.demand_requirement = user_demand_requirement 
    data_generator.generateAllDag()
else: 
    data_generator = Generator(num_of_tasks=tasks)
    max_out, alpha, beta = data_generator.getRandomDAGParameters(True)
    data_generator.generate(max_out, alpha, beta)

"""
    Argument list for Generator Class 

    num_of_tasks  : Number of random tasks that will be generated in the data.xml
    mesh_size     : 2D Mesh size of the Network 
    maps_per_task : Number of mappings done per generated task file (data.xml)
    sim_count     : Default is 0. Only use when you'd like to appends the same dataset 
    rum_sim       : if True, the generated files (data.xml, map.xml, network.xml) 
                    is passed to the ratatoskr simulator to get the latency information 
    result_path   : Path where the generated data will be stored. If specified path is not
                    valid, then it will create the path. 
"""


