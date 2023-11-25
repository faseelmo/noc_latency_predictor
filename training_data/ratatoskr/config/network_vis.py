import utils
import networkx as nx

file = 'network.xml'
root = utils.getRoot(file)

"""Extract info from .XML"""
type_of_nodes = utils.findNodelModel(root)
router_info, pe_info= utils.getAllNodeType(root)
"""
    router_info -> List of dictionaties {
       'node_id'   : value, # Nodes 
       'node_type' : value, # Router or P.E
       'x'         : value, # x_pos
       'y'         : value, # y_pos
       'z'         : value, # z_pos
    }
"""
connections = utils.getConnectInfo(root)


G = nx.Graph()

utils.addNodeToGraph(router_info, '', G)
utils.addNodeToGraph(pe_info, '', G)
utils.addConnectionToGraph(G, connections)

num_routers = len(router_info)
node_mapping = utils.reMapNodes(G)

# G = nx.relabel_nodes(G, node_mapping)
adj_list = utils.getAdjList(G,toPrint=True)

utils.visGraph(G)
