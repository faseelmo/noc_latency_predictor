import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

def findNodelModel(root):
    node_models = []
    for node_type in root.findall('nodeTypes'):
        for model in node_type:
            id_ = model.attrib['id'] # Dict
            for element in model:
                if element.tag == 'model':
                    model_name = element.attrib['value'] # Dict
                    dict_ = dict(id = id_ ,model = model_name )
                    node_models.append(dict_)
    return node_models

def getRoot(file):
    tree = ET.parse(file)
    root = tree.getroot()
    return root

def getElementDict(elements):
    node_id = elements.attrib['id']
    x_pos, y_pos, z_pos, node_type = 0,0,0,None

    for element in elements:
        if element.tag == 'xPos':
            x_pos = float(element.attrib['value'])
        if element.tag == 'yPos':
            y_pos = float(element.attrib['value'])
        if element.tag == 'zPos':
            z_pos = float(element.attrib['value'])
        if element.tag == 'nodeType':
            node_type = int(element.attrib['value'])

    dict_ = dict(node_id= node_id, node_type= node_type, x= x_pos, y= y_pos, z= z_pos)
    return dict_

def getAllNodeType(root):
    """" List of Dict {node_id, model_name} """
    pe_info = []
    router_info = []
    for nodes in root.findall('nodes'):
        for elements in nodes:
            element_dict = getElementDict(elements)
            if element_dict['node_type'] == 0:
                router_info.append(element_dict)
            elif element_dict['node_type'] == 1:
                pe_info.append(element_dict)
    return router_info,pe_info

def getCommInfo(ports):
    source = ports[0][0].attrib['value']
    destination = ports[1][0].attrib['value']
    return dict(src=source, dest=destination)

def getConnectInfo(root):
    connections = []
    for node_type in root.findall('connections'):
        for connection in node_type:
            for ports in connection:
                connections.append(getCommInfo(ports))
    return connections

def addNodeToGraph(nodes, abrev, G):
    for node in nodes:
        node_num = abrev + str(node['node_id'])
        G.add_node(node_num)

def addConnectionToGraph(G, connections):
    for connection in connections: 
        src = connection['src']
        dest = connection['dest']
        G.add_edge(src, dest)

def getNodeAttributes(G):
    node_id = 0
    if node_id in G.nodes:
        attributes = G.nodes[node_id]
        print(f"Attributes of Node {node_id}:")
        for key, value in attributes.items():
            print(f"{key}: {value}")
    else:
        print(f"Node {node_id} not found in the graph.")


def visGraph(G):
    pos = nx.spring_layout(G)  # Layout algorithm (you can choose others)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10, font_color='black', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7)
    plt.show()

def visMultiDiGraph(net):
# Plot with pyvis
    # net = Network(notebook=False,
    #     directed = True,
    #     # select_menu = True, # Show part 1 in the plot (optional)
    #     # filter_menu = True, # Show part 2 in the plot (optional)
    # )
    # net.show_buttons() # Show part 3 in the plot (optional)
    # net.from_nx(G) # Create directly from nx graph
    net.show_buttons(filter_=['physics'])
    net.show('test.html', notebook=False) 

def reMapNodes(G):
    return {node: f'R_{node}' if int(node) <= 15 else f'PE_{int(node)-16}' for node in G.nodes()}

def getAdjList(G, toPrint=False):
    adj_list = nx.to_dict_of_lists(G)
    if toPrint:
        for node, neighbors in adj_list.items():
            print(f"Node {node}: Neighbors {neighbors}")
    return adj_list
