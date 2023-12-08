import argparse
from utils.Generator import Generator
    
    
parser = argparse.ArgumentParser(description='Generate a random task that can be mapped to a NoC and estimate then save latency results')
parser.add_argument('--gen', action='store_true', help='Generate all combination of data')
parser.add_argument('--tasks', type=int, default=4, help='Number of tasks')
parser.add_argument('--maps', type=int, default=1, help='Number of mapping done per task')

args = parser.parse_args()

all_gen = args.gen
tasks = args.tasks
maps = args.maps

if all_gen:
    data_generator = Generator(result_path='data/test', run_sim=True, 
                               num_of_tasks=tasks, maps_per_task=maps,
                               all_dag=True, save_results=True)
else: 
    data_generator = Generator(run_sim=True, num_of_tasks=tasks, all_dag=False)
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
    all_dag       : If True, for the given number of tasks all combination of configuration 
                    of DAGs are generated based on the parameters "set_max_out", "set_alpha" 
                    and "set_beta"
    save_results  : Data is saved as a dictionary contatining all the required parameters for 
                    training the latency model. Results are saved in user defined 'result_path'
    result_path   : Path where the generated data will be stored. If specified path is not
                    valid, then it will create the path. 

"""


