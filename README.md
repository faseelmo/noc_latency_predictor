### Docker  
Ratatoskr simulator depends on SystemC. If your machine dosent have systemC, you can either install it, or run the simulation on a docker image. To build and run the docker image, execute the following commands. 

```
cd docker
sudo python3 build.py
sudo ./start.sh
sudo ./attach.sh
```

To detach from the container press `Ctrl + P` followed by `Ctrl + Q`


### Training Data Generation 
The file `training_data/data/non_iso_graphs.zip` contains a directory with around 800
non-isomorphic graphs with 9 nodes (including 'Start' and 'Exit'). To unzip, run the 
following command in the `data` directory 
```
unzip non_iso_graphs.zip
```
