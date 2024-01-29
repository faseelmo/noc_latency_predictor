from .XML_Generator import XMLGenerator 
import random
import shutil
import os 

class MapGenerator:
    def __init__(self, dag, network, file_location):
        """
            network="4" for 4x4 mesh, network="3" for 3x3 mesh

            self.map = {task_1: mapped_pe_1, .. }
                'Start' -> 0
                'Exit' -> len(self.map) - 1
        """
        num_of_tasks = len(dag.graph.nodes)
        list_of_pe = self.getPEfromNetwork(num_of_tasks, network)

        self.map = self.doMap(num_of_tasks, list_of_pe)
        self.createMapFile(self.map, file_location)
        
    def doMap(self, num_of_tasks, list_of_pe):
        """
            Return a dict "task_ corresponding_p.e"
        """
        map_ = {} 
        for i in range(num_of_tasks):
            choosen_pe = random.choice(list_of_pe)
            list_of_pe.remove(choosen_pe)
            map_[i] = choosen_pe
        return map_
    
    def getPEfromNetwork(self, num_of_tasks, network):
        """
        Gets Lists of PEs from the network (str) 
        Also copies appropriate network file from misc/ to config/ 
        """
        network_file_path = 'ratatoskr/config/misc/networks/' 
        if network == "4":
            list_of_pe = list(range(16, 32, 1))
            network_file_name = '4x4_Mesh.xml' 
        elif network == "2":
            list_of_pe = list(range(4, 8, 1))
            network_file_name = '2x2_Mesh.xml' 
        elif network == "3":
            list_of_pe = list(range(9, 18, 1))
            network_file_name = '3x3_Mesh.xml' 
        else: 
            print("[Warning] Not a Valid Network Choice")
            list_of_pe = []

        assert len(list_of_pe) >= num_of_tasks, "Number of Tasks greater than available PEs in Network" 

        network_config_path = 'ratatoskr/config/'
        shutil.copy(network_file_path + network_file_name, network_config_path + network_file_name)
        os.rename(network_config_path + network_file_name, network_config_path + 'network.xml') # This deletes the original file

        return list_of_pe

    @staticmethod
    def createMapFile(map,  file_location):
        Map = XMLGenerator('map', file_location)
        map_root = Map.initRoot()
        for task in map:
            pe = map[task]
            bind_root = Map.addChild(map_root, 'bind' )
            Map.addChild(bind_root, 'task', ['value'], [str(task)])
            Map.addChild(bind_root, 'node', ['value'], [str(pe)])
        Map.writeFile(map_root)
            
def test(): 
    from .DAG_Generator import DAG
    dag = DAG(nodes=4, max_out=2, demand_range=(1,100))
    from .TASK_Generator import TaskGenerator
    task = TaskGenerator(dag, 'data.xml')
    map = MapGenerator(dag, network="4", file_location="test_map.xml")

test()