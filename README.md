### Framework for Generating Network Application Dataset 

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

### GNN based on-chip Network Latency Analysis
#### Results
The results of each training iteration can be found in the directory `gcn/results`.
The table below summarizes the results and the source of their training data.  
Each of these folders contains a jupyter notebook called `analysis.ipynb` with the results. 

<table>
<tr>
    <th>Folder</th>
    <th>Training Data</th>
    <th>Data Description</th>
</tr>
<tr>
    <td>application_latency_model</td>
    <td>task_7</td>
    <td>Generated using data-generation framework. Each graph has 7 <br>
        nodes (excluding 'Start' and 'Exit' node).
    </td>
</tr>
<tr>
    <td>map_latency_model</td>
    <td>unique_graphs_with_links</td>
    <td>Extract all the unique (non-isomorphic) graphs from task_7. <br>
        For each task, we create 200 maps with links between nodes <br>
        characterizing communication delay. (see analysis.ipynb).
    </td>
</tr>
<tr>
    <td>task_from_graph</td>
    <td>task_from_graph</td>
    <td>From the unique graph, we create 100 mappings per graph <br>
        and find their latencies using the data-generation pipeline <br>
        for a more non-biased dataset.
    </td>
</tr>
</table>




