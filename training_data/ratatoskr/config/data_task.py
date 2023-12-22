import utils
import subprocess
from xml.dom import minidom
import xml.etree.ElementTree as ET

root = utils.getRoot('data.xml')
all_tasks = root.findall('.//tasks/task')
last_task_num = len(all_tasks) - 1

DEMAND = 100
DURATION = -1
DELAY = 100

def changeGenerateCount(task):
    generate = task.findall('generates/possibility/destinations/destination')
    for destinations in generate:
        dest_address = int(destinations.find('task').attrib['value'])
        if dest_address != 0 and dest_address != last_task_num:
            destinations.find('count').attrib['min'] = str(DEMAND)
            destinations.find('count').attrib['max'] = str(DEMAND)

            destinations.find('delay').attrib['min'] = str(DELAY)
            destinations.find('delay').attrib['max'] = str(DELAY)

def matchRequireToGenerate(task):
    for requirement in task.findall('requires/requirement'):
        source_address = int(requirement.find('source').attrib['value'])
        if source_address != 0 and source_address != last_task_num:
            requirement.find('count').attrib['min'] = str(DEMAND)
            requirement.find('count').attrib['max'] = str(DEMAND)

def setDuration(task):
    task.find('duration').attrib['min'] = str(DURATION)
    task.find('duration').attrib['max'] = str(DURATION)

def getMapOfLastTask():
    map_root = utils.getRoot('map.xml')
    all_binds = map_root.findall('bind')
    last_map_num = None
    for bind in all_binds:
        task = int(bind.find('task').attrib['value'])
        node = int(bind.find('node').attrib['value'])
        if task == int(last_task_num):
            last_map_num = node
            break
    return last_map_num

for task in all_tasks:
    current_task_id = int(task.attrib['id'])
    if (current_task_id != 0) and (current_task_id != last_task_num):
        changeGenerateCount(task)
        matchRequireToGenerate(task)
        setDuration(task)

tree = ET.ElementTree(root)
tree.write('data.xml')

last_map_num = getMapOfLastTask()
assert last_map_num is not None, "Map of last task not found"

command = "cd .. && ./sim"
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

processing_time_string = 'Node' + str(last_map_num) 
processing_time = None
simulation_time = None
for line in process.stdout:
    # print(line)
    if processing_time_string in line and 'Receive Flit' in line: 
        processing_time = line.split()[0][:-3]
    if 'Execution time' in line: 
                    start_index = line.index(':')
                    simulation_time = line[start_index+2:]

assert processing_time is not None, "Simulation Not Complete"
print(f"Total Processing Time is {processing_time}ns")
print(f"Simulation Time: {simulation_time}")