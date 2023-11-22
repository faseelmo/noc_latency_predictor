import xml.etree.ElementTree as ET
from xml.dom import minidom

class TaskGenerator:
    def __init__(self, edges, nodes, duration, output):
        self.edges = edges
        self.num_of_tasks = 0 # Gets Incremented each time createTask is called.
        self.nodes = nodes
        self.duration = duration

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
        task_parent = self.addChild(self.root, 'tasks') 
        """
            Create 'Start' Task.
                Single Task having 1 generate with single destination id 
                but with multiple task ids [destination]
        """
        self.createTask(self.num_of_tasks, task_parent, toRequire=False, toGenerate=True)

        """
            Create 'Node' Task.
                Single Task possibly having multiple generate & requirement. 
        """
        self.createTask(self.num_of_tasks, task_parent, toRequire=True, toGenerate=True)

        """
            Create 'Exit' Task.
                Single Task having only requirement. 
                Could have multiple sources. 
        """
        self.createTask(self.num_of_tasks, task_parent, toRequire=True, toGenerate=False)

    def createTask(self, id, task_parent, toRequire=False, toGenerate=False):
        start_edges, _, exit_edges = self.organizeEdges(self.edges)
        if toGenerate and not toRequire:
            """Creating Start Node"""
            task_node = self.createTaskHeader(task_parent)
            self.createGenerate(task_node, start_edges)
        if toRequire and not toGenerate:
            """Creating Exit Node"""
            task_node = self.createTaskHeader(task_parent)
            self.createRequire(task_node, exit_edges)
        if toGenerate and toRequire:
            """Creating Normal Nodes"""
            self.createRequireAndGenerate(task_parent)

    def createTaskHeader(self, task_parent, duration = -1):
        task_node = self.addChild(task_parent, 'task', ['id'], [str(self.num_of_tasks)])
        self.addChild(task_node, 'start', ['min', 'max'], ["0", "0"])
        self.addChild(task_node, 'duration', ['min', 'max'], [str(duration), str(duration)])
        self.addChild(task_node, 'repeat', ['min', 'max'], ["1", "1"])
        self.num_of_tasks += 1
        return task_node

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

    def createRequireAndGenerate(self, task_parent):
        for i in range(self.nodes):
            duration = self.duration[i]
            task_node = self.createTaskHeader(task_parent, duration=duration)
            node = i+1
            node_edges = self.getNodeInfoFromEdges(node)
            start_edges, node_edges, exit_edges = self.organizeEdges(node_edges)

            if start_edges: 
                edge = [(0, start_edges[0][1])] 
                self.createRequire(task_node, edge)

            if node_edges: 
                for node_edge in node_edges:
                    if node_edge[1] == node:
                        self.createRequire(task_node, [node_edge])
                    elif node == node_edge[0]:
                        self.createGenerate(task_node, [node_edge])

            if exit_edges: 
                edge = [(exit_edges[0][0], self.nodes + 1 )] 
                self.createGenerate(task_node, edge)


    def addChild(self, parent, child_name, key_list=[], value_list=[]):
        node = ET.SubElement(parent, child_name)
        if len(key_list) != 0:
            for i in range(len(key_list)):
                node.set(key_list[i], value_list[i])
        return node

    def getNodeInfoFromEdges(self, node):
        return [(x, y) for x, y in self.edges if node in (x, y)]

    def organizeEdges(self, edges):
        start_edges = []
        exit_edges = []
        node_edges = []

        """Organizing Edges"""
        for edge in edges:
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



