import xml.etree.ElementTree as ET
from xml.dom import minidom

class TaskGenerator:
    def __init__(self, edges, output):
        self.edges = edges
        self.num_of_tasks = 0 # Gets Incremented each time createTask is called.

        self.root = self.initRoot()
        self.addDataType()
        self.addTasks()



        self.writeFile(output)

    def initRoot(self):
        root_node = ET.Element('data') 
        root_node.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance') 
        return root_node

    def addDataType(self):
        data_type_nodes = self.addChild(self.root, 'dataTypes')

        """Modify the following code to add More Data Types"""
        data_type_node = self.addChild(data_type_nodes, 'dataTypes', ['id'], ['0'])
        self.addChild(data_type_node, 'name', ['value'], ['bit'])

    def addTasks(self):

        start_edges, node_edges, exit_edges = self.organizeEdges()
        task_parent = self.addChild(self.root, 'tasks') 
        """
            Create 'Start' Task.
                Single Task having 1 generate with single destination id 
                but with multiple task ids [destination]
        """
        self.createTask(self.num_of_tasks, task_parent, start_edges, toRequire=False, toGenerate=True)

        """
            Create 'Exit' Task.
                Single Task having only requirement. 
                Could have multiple sources. 
        """
        self.createTask(self.num_of_tasks, task_parent, exit_edges, toRequire=True, toGenerate=False)

        """
            Create 'Node' Task.
                Single Task possibly having multiple generate & requirement. 
        """
        self.createTask(self.num_of_tasks, task_parent, exit_edges, toRequire=True, toGenerate=True)






        print(f"Start Edge is {start_edges}")
        print(f"Node Edge is {node_edges}")
        print(f"Exit Edge is {exit_edges}")

    def createTask(self, id, task_parent, edges, toRequire=False, toGenerate=False):

        """Header for Task"""
        task_node = self.addChild(task_parent, 'task', ['id'], [str(id)])
        self.addChild(task_node, 'start', ['min', 'max'], ["0", "0"])
        self.addChild(task_node, 'duration', ['min', 'max'], ["-1", "-1"])
        self.addChild(task_node, 'repeat', ['min', 'max'], ["1", "1"])
        self.num_of_tasks += 1

        if toGenerate:
            self.createGenerate(task_node, edges)
        if toRequire:
            self.createRequire(task_node, edges)
        if toGenerate and toRequire:
            print("Its complicated Lmao")

    def createGenerate(self, task_node, edges):
        """Header for Generate"""
        generate_node = self.addChild(task_node, 'generates')
        possibility_node = self.addChild(generate_node, 'possibility', ['id'], ["0"])
        self.addChild(possibility_node, 'probability', ['value'], ["1"])
        destinations_node = self.addChild(possibility_node, 'destinations')
        destination_id = 0
        for edge in edges:
            destination_node = self.addChild(destinations_node, 'destination' ,['id'], [str(destination_id)])
            self.addChild(destination_node, 'delay', ['min', 'max'], ["0", "0"])
            self.addChild(destination_node, 'interval', ['min', 'max'], ["0", "0"])
            self.addChild(destination_node, 'count', ['min', 'max'], ["1", "1"])
            self.addChild(destination_node, 'type', ['value'], ["0"])
            self.addChild(destination_node, 'task', ['value'], [str(edge[1])])
            destination_id += 1

    def createRequire(self, task_node, edges):
        """Header for Generate"""
        requires_node = self.addChild(task_node, 'requires')
        requirement_id = 0
        for edge in edges:
            requirement_node = self.addChild(requires_node, 'requirement' ,['id'], [str(requirement_id)])
            self.addChild(requirement_node, 'type', ['value'], ["0"])
            self.addChild(requirement_node, 'source', ['value'], [str(edge[0])])
            self.addChild(requirement_node, 'count', ['min', 'max'], ["1", "1"])

    def addChild(self, parent, child_name, key_list=[], value_list=[]):
        node = ET.SubElement(parent, child_name)
        if len(key_list) != 0:
            for i in range(len(key_list)):
                node.set(key_list[i], value_list[i])
        return node

    def organizeEdges(self):
        start_edges = []
        exit_edges = []
        node_edges = []

        """Organizing Edges"""
        for edge in self.edges:
            if edge[0] == 'Start':
                start_edges.append(edge)
            elif edge[1] == 'Exit':
                exit_edges.append(edge)
            else: 
                node_edges.append(edge)

        return start_edges, node_edges, exit_edges



    def writeFile(self, output_file):
        rough_string = ET.tostring(self.root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        data = reparsed.toprettyxml(indent="  ")
        of = open(output_file, 'w')
        of.write(data)
        of.close()



