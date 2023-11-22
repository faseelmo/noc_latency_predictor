from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG

num_of_nodes = 2

# set_dag_size = [20,30,40,50,60,70,80,90]             #random number of DAG  nodes       
# set_max_out = [1,2,3,4,5]                            #max out_degree of one node
# set_alpha = [0.5,1.0,1.5]                            #DAG shape
# set_beta = [0.0,0.5,1.0,2.0]                         #DAG regularity

dag = DAG(nodes=num_of_nodes, max_out=3)
dag.plot_DAG()
dag.getInfo()

TaskGenerator(dag.edges, num_of_nodes, dag.duration, dag.demand,'dev_output.xml')

