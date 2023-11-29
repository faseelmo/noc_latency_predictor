from utils.XML_Generator import XMLGenerator 
import random
import shutil
import os 
import copy
import matplotlib.pyplot as plt
import networkx as nx

class MapGenerator:
    def __init__(self, num_of_tasks, network, file_location):
        self.num_of_tasks = num_of_tasks
        self.network = network 
        self.map = self.doMap()
        self.createMapFile(file_location)
    
    def getPEfromNetwork(self):
        network_file_path = 'ratatoskr/config/misc/networks/' 
        if self.network == "4":
            list_of_pe = list(range(16, 32, 1))
            network_file_name = '4x4_Mesh.xml' 
        elif self.network == "2":
            list_of_pe = list(range(4, 8, 1))
            network_file_name = '2x2_Mesh.xml' 
        else: 
            print("[Warning] Not a Valid Network Choice")

        network_config_path = 'ratatoskr/config/'
        shutil.copy(network_file_path + network_file_name, network_config_path + network_file_name)
        os.rename(network_config_path + network_file_name, network_config_path + 'network.xml') # This deletes the original file

        return list_of_pe

    def doMap(self):
        list_of_pe = self.getPEfromNetwork()
        map_ = [] # List of Tuples (task, corresponding_p.e)
        for i in range(self.num_of_tasks):
            choosen_pe = random.choice(list_of_pe)
            list_of_pe.remove(choosen_pe)
            map_.append((i, choosen_pe))
        return map_

    def createMapFile(self, file_location):
        Map = XMLGenerator('map', file_location)
        map_root = Map.initRoot()

        for task, pe in self.map:
            bind_root = Map.addChild(map_root, 'bind' )
            Map.addChild(bind_root, 'task', ['value'], [str(task)])
            Map.addChild(bind_root, 'node', ['value'], [str(pe)])

        Map.writeFile(map_root)

    def getRenameDict(self, pos):
        """Return a dictionary that can be used to rename the nodes in the original graph
            Also returns a dictionary with positons of nodes in the new graph
        """
        rename = {}
        final_task_id = self.num_of_tasks - 1
        new_pos = copy.deepcopy(pos)

        for task, node in self.map:
            if task == 0:
                new_pos[node] = new_pos.pop('Start')
                rename['Start'] = node
            elif task == final_task_id:
                new_pos[node] = new_pos.pop('Exit')
                rename['Exit'] = node
            else: 
                new_pos[node] = new_pos.pop(task)
                rename[task] = node

        return rename, new_pos

    def doGraphRemapping(self, g1, renaming_dict):
        return nx.relabel_nodes(g1, renaming_dict)

    def plotTaskAndMap(self, task_graph, task_positon, map_graph, map_position, lat_list=[]):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        nx.draw(task_graph, arrows=True,with_labels=True, pos=task_positon, ax=axes[0])
        axes[0].set_title('Task Graph')

        nx.draw(map_graph, arrows=True,with_labels=True, pos=map_position, ax=axes[1])
        axes[1].set_title('Task Mapped to Network Node graph')
        
        if lat_list:
            print("Printing Results on Graph")
            text = 'Avg Flit Lat = ' + str(lat_list[0]) + ', Avg Packet Latency = ' + str(lat_list[1]) + ', Avg Network Lat = ' + str(lat_list[2])
            fig.text(0.2, 0.05, text, fontsize=12, color='red')

        plt.savefig("Graphs.png", format="PNG")



            




