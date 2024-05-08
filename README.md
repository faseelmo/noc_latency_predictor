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
#### Dataset 
The table below summarizes the results and the source of their training data.  
<table>
<tr>
    <th>Folder</th>
    <th>Training Data</th>
    <th>Data Description</th>
</tr>
<tr>
    <td>application_latency_model</td>
    <td><i>task_7</i></td>
    <td>Generated using data-generation framework.  <br>
        Each graph has 7 nodes (excluding 'Start' and 'Exit' node).
    </td>
</tr>
<tr>
    <td>map_latency_model</td>
    <td><i>unique_graphs_with_links</i></td>
    <td>Extract all the unique (non-isomorphic) graphs from task_7. <br>
        For each task, we create 200 maps with links between nodes characterizing communication delay (see analysis.ipynb)
    </td>
</tr>
<tr>
    <td>task_from_graph</td>
    <td><i>task_from_graph</i></td>
    <td>From the unique graph, we create 100 mappings per graph 
        and find their latencies using the data-generation pipeline 
        for a more non-biased dataset.
    </td>
</tr>
</table>

#### Input Node Feature
All the training data is converted to the object torch_geometric.data.Data. 
The following table summarizes the input feature for each dataset. 
<table>
<tr>
    <th>Training Data</th>
    <th>Input Feature</th>
</tr>
<tr>
    <td><i>task_7</i></td>
    <td>Delay of processing element (PE). Fixed to 10 here.</td>
</tr>
<tr>
    <td><i>unique_graphs_with_links</i></td>
    <td>Each node (PE or communication link) has its corresponding delay plus its correspoding type (PE or communication link) <br>
    'Start' and 'Exit' node considered to be of type PE. 
    </td>
</tr>
<tr>
    <td><i>task_from_graph</i></td>
    <td>Each node has its type as one-hot encoding. Types can be 'Start' node, 'Exit' node, or 'PE' node. </td>
</tr>
</table>


#### Results
The directory `gcn/results` contains the results of each training iteration. Inside each of these folders, you can find a Jupyter notebook named `analysis.ipynb`, with the corresponding results.


##### 1. DAG on Mesh Network  
1. V1 (7.5.24)  
    Total Parameters = 1,228,801
    3 GraphConv Layers **512->512->512**  
    4 Layer MLP **512->256->128->64->1**

2. V2 (8.5.24)
    Total Parameters = 5,443,585
    3 GraphConv Layers **2048->1024->512**  
    4 Layer MLP **512->256->128->64->1**



| Test idx | V1 Tau | V2 Tau | 
| - | ------------- | ------ | 
| 0	|   0.133201    |
| 1	|   0.117391    |
| 2	|   0.122078    |
| 3	|   0.151247    |
| 4	|   0.125000    |
| 5	|   0.171571    |
| 6	|   0.272187    |
| 7	|   0.151343    |
| 8	|   0.181536    |
| 9	|   0.164637    |
| **mean**| 0.1590  |



