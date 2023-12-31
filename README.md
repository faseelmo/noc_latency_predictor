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
