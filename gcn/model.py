import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

torch.manual_seed(1)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=512, num_node_features=32):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

if __name__ == "__main__":
    
    batch_size = 10

    # Loading the Model
    device = torch.device('cpu')
    model = GCN().to(device)
    learn_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of Learnable parameters: {learn_model_parameters}, Total Param: {total_params}")
    

    # Loading the dataset
    from .dataset import load_data
    data_loader, _ = load_data('training_data/data/training_data_old', batch_size=batch_size)
    for i, data in enumerate(data_loader):
        print(f"\nBatch {i} has {len(data)} data points")
        print(f"Data is {data}")
        # print(f"Data batch is {data.batch}")
        output = model(data.x, data.edge_index, data.batch).squeeze(1)
        print(f"Shape of output is {output.shape} and target is {data.y.shape}")
        loss = F.mse_loss(output, data.y)
        print(f"Output is {output}")


        if i == 1: break
    

    # data_iter = iter(data_loader)
    # first_batch = next(data_iter)
    # print(f"First Batch size is {len(first_batch)}")
    # print(f"Number of Batches is {len(data_loader)}\n")



    # print(first_batch[0])
    # print(first_batch)
    # data = first_batch.to(device)
    # print(f"Data is {data}")
    # x = data.x
    # edge_index = data.edge_index
    # print(f"Data is {data}")
    # print(f"Data.x is {x}")
    # print(f"Data.edge_index is {edge_index}")
    # output = model(x, edge_index)
    # print(f"Shape of output is {output.shape}")
    # print(f"Shape of target is {data.y.shape}")
    # print(f"\nOutput of the model is {output}")
    # loss = F.mse_loss(output, data.y)
    # print(f"Loss is {loss}")
    
    """
    Testing Conv for Batches
    """

    # embedding_dim = 16
    # conv_model = ConvBlock(embedding_dim,2)
    
    # # For single data
    # first_data = first_batch[0]
    # first_data_edge_index = first_data.edge_index
    # embedding = nn.Embedding(9, embedding_dim)
    # nn.init.normal_(embedding.weight, std=0.1)
    
    # print(f"\nSingle Data is {first_data}")
    # print(f"Embedding shape {embedding.weight.shape}")
    # conv_model(embedding.weight ,first_data_edge_index)

    # # For Batch
    # first_batch_edge_index = first_batch.edge_index
    # embedding = nn.Embedding(18, embedding_dim)
    # nn.init.normal_(embedding.weight, std=0.1)
    
    # print(f"\nBatch Data is {first_batch}")
    # print(f"Embedding shape {embedding.weight.shape}")
    # conv_model(embedding.weight, first_batch_edge_index)


    # print(model)

