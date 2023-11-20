import utils.DAG_Generator as dag
from utils.TASK_Generator import TaskGenerator

num_of_nodes = 4
edges, duration, demand, position = dag.workflows_generator(mode='default', 
                                                            n=num_of_nodes,
                                                            alpha= 1,
                                                            beta= 2.0,
                                                            max_out=2)

print(f"\nNo.of Nodes: {len(duration)}, No.of edges = {len(edges)}\n")
for i in range(len(duration)):
    print(f"Node: {i+1}, Processing Duration: {duration[i]}, Require: {demand[i]}")

TaskGenerator(edges, num_of_nodes,'dev_output.xml')

# task.getXML(edges)

# root_node = task.initRoot('data')

# data_types_node = ET.SubElement(self.root_node, 'dataTypes') # Including the header name <datatypes>
# task.add_data_type_node(root_node, )

# task.writeFile(root_node, 'output')