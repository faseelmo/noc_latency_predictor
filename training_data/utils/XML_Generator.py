import xml.etree.ElementTree as ET
from xml.dom import minidom

class XMLGenerator:
    def __init__(self, root_name, output_file):
        self.root_name = root_name
        self.output_file = output_file

    def initRoot(self):
        root_node = ET.Element(self.root_name) 
        root_node.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance') 
        return root_node

    def addChild(self, parent, child_name, key_list=[], value_list=[]):
        node = ET.SubElement(parent, child_name)
        if len(key_list) != 0:
            for i in range(len(key_list)):
                node.set(key_list[i], value_list[i])
        return node

    def writeFile(self, root):
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        data = reparsed.toprettyxml(indent="  ")
        of = open(self.output_file, 'w')
        of.write(data)
        of.close()
