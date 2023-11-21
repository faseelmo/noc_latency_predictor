# import utils.DAG_Generator as dag
from utils.TASK_Generator import TaskGenerator
from utils.DAG_Generator import DAG

num_of_nodes = 4

# set_dag_size = [20,30,40,50,60,70,80,90]             #random number of DAG  nodes       
# set_max_out = [1,2,3,4,5]                            #max out_degree of one node
# set_alpha = [0.5,1.0,1.5]                            #DAG shape
# set_beta = [0.0,0.5,1.0,2.0]                         #DAG regularity

dg = DAG(nodes=num_of_nodes)
dg.plot_DAG()
print(dg.position)


# print(f"\nNo.of Nodes: {len(duration)}, No.of edges = {len(edges)}\n")

# for i in range(len(duration)):
#     print(f"Node: {i+1}, Processing Duration: {duration[i]}, Require: {demand[i]}")

TaskGenerator(dg.edges, num_of_nodes,'dev_output.xml')
