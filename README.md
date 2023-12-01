### Training Data Generation 
Generate a random task from a Directed Acyclic Graph, map tasks to network nodes, and calculate network latency using these commands.  

```
cd training_data/
python3 generate.py --tasks 4 --runSim True
```
The code above creates 4 tasks (excluding the starting and ending node) and then runs the ratatoskr simulator on the created config files. 