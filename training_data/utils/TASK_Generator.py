from .XML_Generator import XMLGenerator 

class TaskGenerator:
    def __init__(self, dag, output_file):
        self.num_nodes = len(dag.graph.nodes)
        self.nodes = list(dag.graph.nodes)
        self.edges = list(dag.graph.edges)

        self.delay = dag.node_attr
        self.demand = dag.edge_attr

        self.num_of_tasks = 0 # Gets Incremented each time createTask is called.

        self.xml_gen = XMLGenerator('data', output_file)
        self.root = self.xml_gen.initRoot()

        self.addDataType()
        self.addTasks()

        self.xml_gen.writeFile(self.root)

    def addDataType(self):
        data_type_nodes = self.xml_gen.addChild(self.root, 'dataTypes')
        """Modify the following code to add More Data Types"""
        data_type_node = self.xml_gen.addChild(data_type_nodes, 'dataType', ['id'], ['0'])
        self.xml_gen.addChild(data_type_node, 'name', ['value'], ['bit'])

    def addTasks(self):
        task_parent = self.xml_gen.addChild(self.root, 'tasks') 
        self.createRequireAndGenerate(task_parent)

    def createTaskHeader(self, task_parent, duration = -1):
        task_node = self.xml_gen.addChild(task_parent, 'task', ['id'], [str(self.num_of_tasks)])
        self.xml_gen.addChild(task_node, 'start', ['min', 'max'], ["0", "0"])
        self.xml_gen.addChild(task_node, 'duration', ['min', 'max'], [str(duration), str(duration)])
        self.xml_gen.addChild(task_node, 'repeat', ['min', 'max'], ["1", "1"])
        self.num_of_tasks += 1
        return task_node

    def createGenerate(self, task_node, generate_dest, demand, delay, firstGenerate, destination_id):
        """Header for Generate"""
        if firstGenerate:
            generate_node = self.xml_gen.addChild(task_node, 'generates')
            possibility_node = self.xml_gen.addChild(generate_node, 'possibility', ['id'], ["0"])
            self.xml_gen.addChild(possibility_node, 'probability', ['value'], ["1"])
            destinations_node = self.xml_gen.addChild(possibility_node, 'destinations')
        else: 
            destinations_node = task_node.find('generates/possibility/destinations')

        destination_node = self.xml_gen.addChild(destinations_node, 'destination' ,['id'], [str(destination_id)])
        self.xml_gen.addChild(destination_node, 'delay', ['min', 'max'], [str(delay), str(delay)])
        self.xml_gen.addChild(destination_node, 'interval', ['min', 'max'], ["0", "0"])
        self.xml_gen.addChild(destination_node, 'count', ['min', 'max'], [str(demand), str(demand)])
        self.xml_gen.addChild(destination_node, 'type', ['value'], ["0"])
        self.xml_gen.addChild(destination_node, 'task', ['value'], [str(generate_dest)])
        destination_id += 1

    def createRequire(self, task_node, require_src, demand, firstRequire, requirement_id):
        """Header for Require"""
        if firstRequire:
            requires_node = self.xml_gen.addChild(task_node, 'requires')
        else: 
            requires_node = task_node.find('requires')

        requirement_node = self.xml_gen.addChild(requires_node, 'requirement' ,['id'], [str(requirement_id)])
        self.xml_gen.addChild(requirement_node, 'type', ['value'], ["0"])
        self.xml_gen.addChild(requirement_node, 'source', ['value'], [str(require_src)])
        self.xml_gen.addChild(requirement_node, 'count', ['min', 'max'], [str(demand), str(demand)])

    def createRequireAndGenerate(self, task_parent):

        for i in range(self.num_nodes): 
            if i == 0:                      node = 'Start'
            elif i == (self.num_nodes-1):   node = 'Exit'
            else:                           node = i
            
            task_node = self.createTaskHeader(task_parent)

            first_require = first_generate = True
            require_id = generate_id = 0
            for (src, dst) in self.edges:
                if node == dst:
                    """This Node needs require from src"""
                    demand = self.demand[(src, dst)]
                    if src == 'Start': src = 0

                    self.createRequire(
                        task_node, src, 
                        demand=demand, 
                        firstRequire=first_require, 
                        requirement_id=require_id
                    )
                    first_require=False
                    require_id += 1

                if node == src: 
                    """This Node need Generate to dst"""
                    demand = self.demand[(src, dst)]

                    if src == 'Start': delay = 0 # Zero Delay for the Start Node
                    else : delay = self.delay[src]

                    if dst == 'Exit': dst = self.num_nodes - 1

                    self.createGenerate(
                        task_node=task_node, 
                        generate_dest=dst, 
                        demand=demand, 
                        delay=delay, 
                        firstGenerate=first_generate,
                        destination_id=generate_id
                    )
                    first_generate=False
                    generate_id+=1
        
# def print_xml(root):
#     xml_string = ET.tostring(root, encoding="utf-8")
#     import xml.dom.minidom
#     dom = xml.dom.minidom.parseString(xml_string)
#     pretty_xml_string = dom.toprettyxml(indent="  ")
#     print(pretty_xml_string)
#     import sys
#     sys.exit(0)

# def test(): 
#     from DAG_Generator import DAG
#     dag = DAG(nodes=4, max_out=2, demand_range=(1,100))

    # import networkx as nx
    # import matplotlib.pyplot as plt
    # nx.draw(dag.graph,  pos=dag.position, with_labels=True,  font_weight='bold', node_color='skyblue', edge_color='gray', node_size=800)
    # nx.draw_networkx_edge_labels(dag.graph, pos=dag.position, edge_labels=dag.edge_attr)
    # plt.show()
    # task = TaskGenerator(dag, 'test_data.xml')

# test()