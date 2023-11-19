import xml.etree.ElementTree as ET
from xml.dom import minidom

class TaskGenerator:
    def __init__(self, edges, output):
        self.root = self.initRoot()
        self.addDataType()
        self.addTasks(edges)



        self.writeFile(output)

    def initRoot(self):
        root_node = ET.Element('data') 
        root_node.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance') 
        return root_node

    def addDataType(self):
        data_type_nodes = ET.SubElement(self.root, 'dataTypes')

        """Modify the following code to add More Data Types"""
        data_type_node = ET.SubElement(data_type_nodes, 'dataTypes')
        data_type_node.set('id', '1')

        name_node = ET.SubElement(data_type_node, 'name')
        name_node.set('value', 'bit')

    def addTasks(self, edges):
        start_edges = []
        node_edges = []
        exit_edges = []

        """Organizing Edges"""
        for edge in edges:
            if edge[0] == 'Start':
                start_edges.append(edge)
            elif edge[1] == 'Exit':
                exit_edges.append(edge)
            else: 
                node_edges.append(edge)

        """
            Create Start Task.j
                Single Task having 1 generate with single destination id 
                but with multiple task ids [destination]
        
        """


        print(f"Start Edge is {start_edges}")
        print(f"Node Edge is {node_edges}")
        print(f"Exit Edge is {exit_edges}")


    def writeFile(self, output_file):
        rough_string = ET.tostring(self.root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        data = reparsed.toprettyxml(indent="  ")
        of = open(output_file, 'w')
        of.write(data)
        of.close()



