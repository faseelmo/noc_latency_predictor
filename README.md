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
    Tensor Graph Type = undirected  
    Model = 250th Epoch   
    3 GraphConv Layers **512->512->512**   
    4 Layer MLP **512->256->128->64->1**  
    Total Parameters = 1,228,801  

2. V2 (8.5.24)  
    Tensor Graph Type = undirected  
    Model = 100th Epoch 
    3 GraphConv Layers **2048->1024->512**   
    4 Layer MLP **512->256->128->64->1**  
    Total Parameters = 5,443,585  

3. V3 (9.5.24)  
    Tensor Graph Type = directed  
    Model = 300th Epoch  
    Model same as **V1**  

4. V4 (9.5.24)
    Tensor Graph Type = undirected  
    Model = 600 (Model wasnt converging)   
    6 GraphConv with Residual   
    4 Layer MLP   
    Total Parameters = 2,803,201

5. V5 (9.5.24)  
    Same as **V1** but implemented Residuals after every GraphConv  
    Model = 100th Epoch 

5. V6 (9.5.24)  
    Same as **V1** but implemented Residuals after every GraphConv  
    + added an extra conv layer with Residuals 
    Model = 250th Epoch 






| Test idx | V1 Tau | V2 Tau    | V3 Tau    |  V4 Tau    |  V5 Tau    |  V5 Tau    |  
| - | ------------- | --------- | --------- |  --------- |  --------- |  --------- |
| 0	|   0.160858    | 0.105544  | 0.058364  |  0.076260  |  0.167365  |  0.107984  |
| 1	|   0.139070    | 0.101848  | 0.065854  |  0.073216  |  0.217603  |  0.125163  |
| 2	|   0.156667    | 0.169282  | 0.092780  |  0.151784  |  0.219334  |  0.160737  |
| 3	|   0.183540    | 0.264069  | 0.156152  |  0.169233  |  0.247718  |  0.233410  |
| 4	|   0.277471    | 0.116483  | 0.250640  |  0.145869  |  0.333264  |  0.195273  |
| 5	|   0.103764    | 0.124723  | 0.040889  |  0.104175  |  0.123490  |  0.149790  |
| 6	|   0.269341    | 0.318128  | 0.129487  |  0.203886  |  0.315282  |  0.259178  |
| 7	|   0.101575    | 0.208453  | 0.138289  |  0.124419  |  0.183570  |  0.217428  |
| 8	|   0.212645    | 0.154111  | 0.013303  |  0.067743  |  0.179489  |  0.125868  |
| 9	|   0.216671    | 0.213012  | 0.132929  |  0.203256  |  0.182524  |  0.253257  |
| **mean**| 0.1821  | 0.1775    | 0.10786   |  0.13198   |  0.2169    |  0.1828    |
