### Docker  
Ratatoskr simulator depends on SystemC. If your machine dosent have systemC, you can either install it, or run the simulation on a docker image. To build and run the docker image, execute the following commands. 

```
source download_sysc.sh
docker build --no-cache -t systemc .
docker run -ti --rm systemc
```
Once inside the docker container, you can clone this repo and then run the simulation.  
When the simulation is done, you can copy the data from the docker container to the host machine using 
```
docker cp systemc:<container-path> <host-path>
```

### Training Data Generation 
Generate a random task from a Directed Acyclic Graph, map those random tasks to network nodes, and then calculate network latency on Ratatoskr.   

Navigate to the [training data](training_data) directory. 
```
cd training_data/
```
1. If you want to generate a data.xml file with 5 tasks and perform the simulation once, run the [generate](training_data/generate.py) python script with the following arguments. 
```
python3 generate.py --tasks 5
```
> The results can be seen in _training_data/ratatoskr/results_.

2. If you want to generate multiple random possible combination of DAGs for a given number of tasks (say 4) and then do 50 random mapping per tasks, run the python script with the following arguments. Here demand for each task is random. 
```
python3 generate.py --gen --tasksNum 4 --mapsPerTask 50
```
> The results can be seen in _training_data/data_.

3. Demand and PE Delay 
Current demand parameters are set to  
    - lowDemend = [1,100]  
    - mediumDemend = [100,300]  
    - highDemend = [300,500]  
    If needed these values can be modied in [DAG Genearator](training_data/utils/DAG_Generator.py)  

For now delay values of each processing element is also set using the above demand parameters.  
Note: The term 'duration' and 'delay' are used interchangeably in the code.

