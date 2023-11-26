from utils.XML_Generator import XMLGenerator 
import random
import shutil
import os 

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




