import argparse

from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG
from utils.Map_Generator import MapGenerator

parser = argparse.ArgumentParser(description='Creates random tasks along with the required congfig files to run on ratatoskr')
parser.add_argument('--tasks', type=int, default=2, help='Number of tasks (excluding Start and Exit Task)')
args = parser.parse_args()

num_of_nodes = args.tasks
network = '4' # Select '4' for 4x4 Mesh and '2' for 2x2 Mesh
max_out = 3
alpha = 0.5
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