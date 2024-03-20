### Training Data Generation 

#### Non-Isomorphic Graphs 
The file `training_data/data/non_iso_graphs.zip` contains a directory with around 800
non-isomorphic graphs having 9 nodes (including 'Start' and 'Exit') each. To unzip, run the 
following command in the `data` directory 
```
unzip non_iso_graphs.zip
```
If you don't want to use these graphs, or want graphs with different number of 
nodes, you can do that while running the `training_data/generate.py` script. 
Look into this file for more information about its arguments. 

#### Docker  
Ratatoskr simulator depends on SystemC. If your machine doesn't have SystemC, you can either install it, or run the simulation on a docker image. To build and run the docker image, execute the following commands. 

```
cd docker
sudo python3 build.py
sudo ./start.sh
sudo ./attach.sh
```

To detach from the container press `Ctrl + P` followed by `Ctrl + Q`