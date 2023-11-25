from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG

"""
    set_max_out = [1,2,3,4,5]                            #max out_degree of one node
    set_alpha = [0.5,1.0,1.5]                            #DAG shape
    set_beta = [0.0,0.5,1.0,2.0]                         #DAG regularity
"""

num_of_nodes = 2
dag = DAG(nodes=num_of_nodes, max_out=3, withDemand=False, withDuration=False) #withDuration sim dosent work
dag.plot_DAG()
dag.getInfo()

# TaskGenerator(dag, 'dev_output.xml')
TaskGenerator(dag, 'ratatoskr/config/data.xml')

