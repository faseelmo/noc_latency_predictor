import utils.DAG_Generator as dag
from utils.TASK_Generator import TaskGenerator

edges, duration, demand, position = dag.workflows_generator(mode='default', 
                                                            n=4,
                                                            alpha= 1,
                                                            beta= 2.0,
                                                            max_out=3)

print(f"\nNo.of Nodes: {len(duration)}, No.of edges = {len(edges)}\n")
for i in range(len(duration)):
    print(f"Node: {i+1}, Processing Duration: {duration[i]}, Require: {demand[i]}")

TaskGenerator(edges, 'dev_output')

# task.getXML(edges)

# root_node = task.initRoot('data')

# data_types_node = ET.SubElement(self.root_node, 'dataTypes') # Including the header name <datatypes>
# task.add_data_type_node(root_node, )

# task.writeFile(root_node, 'output')