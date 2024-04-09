import torch 
import torch.nn as nn 
import torch.optim as optim 

from .model import GCN
from .dataset import load_data

import os 
import time
import pickle
import sys
import math
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Training Information """
EPOCHS = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
WEIGHT_DECAY = 0
BATCH_SIZE = 128

"""Dataset Information """
# DATA_DIR = 'training_data/data/task_from_graph_tensor'
INPUT_FEATURES = 3                                             #Node Level Features
# NUM_NODES = 32

"""Load Model"""
LOAD_MODEL = False
MODEL_PATH = "gcn/results/"

torch.manual_seed(1)

SAVE_RESULTS = "gcn/results/task_from_graph/upto_q2"

if not os.path.exists(SAVE_RESULTS):
    os.makedirs(SAVE_RESULTS)
    print(f"Folder '{SAVE_RESULTS}' created.")
else:
    print(f"Folder '{SAVE_RESULTS}' already exists.")

# Copy model.py to the results folder
shutil.copy2('gcn/model.py', f'{SAVE_RESULTS}/model.py')

def train_fn(train_loader, model, optimizer, loss_fn): 
    loop = tqdm(train_loader, leave=True)

    mean_loss = []
    for batch_idx, data in enumerate(loop):
        data = data.to(DEVICE)
        output = model(data.x, data.edge_index, data.batch).squeeze(1)
        loss = loss_fn(output, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

    train_loss = math.sqrt(sum(mean_loss)/len(mean_loss) )
    return train_loss

def validation_fn(test_loader, model, loss_fn, epoch):
    mean_loss = []
    for data in test_loader:
        data = data.to(DEVICE)
        x = data.x
        edge_index = data.edge_index
        output = model(x, edge_index, data.batch).squeeze(1)
        loss = loss_fn(output, data.y)
        mean_loss.append(loss.item()) 
    
    validation_set_loss = math.sqrt(sum(mean_loss)/len(mean_loss))
    print(f"[{epoch+1}/{EPOCHS}] Validation Loss is {math.sqrt(validation_set_loss)}")
    return validation_set_loss

def plot_and_save_loss(train_loss, valid_loss):
    epochs = range(1, len(train_loss) + 1)
    
    plt.yscale('log')
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, valid_loss, label='Validation Loss')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.savefig(f'{SAVE_RESULTS}/validation_plot_log.png')
    plt.clf()

    loss_dict = {
        'train_loss' : train_loss,
        'valid_loss' : valid_loss,
    }
    with open(f'{SAVE_RESULTS}/loss_dict.pkl', 'wb') as file:
        pickle.dump(loss_dict, file)

def main():
    start_time = time.time()
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
        print(f"Data Directory is {DATA_DIR}")
        # sys.exit(0)

    train_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE)
    

    model = GCN(num_node_features=INPUT_FEATURES).to(DEVICE)
    # model = GCN()
    # model = nn.DataParallel(model).to(DEVICE)

    learning_rate = 0.001 #5e-4

    if LOAD_MODEL:
        model_state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(model_state_dict)
        print(f"\nPre-Trained Wieghts Loaded\n")

    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.MSELoss().to(DEVICE)
    # loss_fn = nn.L1Loss().to(DEVICE)

    valid_loss_list = []
    train_loss_list = []
    for epoch in range(EPOCHS): 
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        valid_loss = validation_fn(test_loader, model, loss_fn, epoch)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        plot_and_save_loss(train_loss_list, valid_loss_list)

        if (epoch+1) == 50: 
            learning_rate = 0.0005
            print(f"Learning Rate Changed to {learning_rate}")

        if (epoch+1) == 150: 
            learning_rate = 0.0001
            print(f"Learning Rate Changed to {learning_rate}")

        if (epoch+1) == 250: 
            learning_rate = 0.00005
            print(f"Learning Rate Changed to {learning_rate}")

        if (epoch+1) == 350: 
            learning_rate = 0.00001
            print(f"Learning Rate Changed to {learning_rate}")

        if (epoch+1) % 50 == 0 or (epoch+1) == 1:
            torch.save(model, f'{SAVE_RESULTS}/LatNet_{epoch+1}.pth')
            torch.save(model.state_dict(), f'{SAVE_RESULTS}/LatNet_{epoch+1}_state_dict.pth')
            end_time = time.time()
            time_elapsed = (end_time - start_time) / 60
            print(f"\nTraining Time: {time_elapsed} minutes\n")

    torch.save(model.state_dict(), f'{SAVE_RESULTS}/LatNet_state_dict.pth')
    torch.save(model, f'{SAVE_RESULTS}/LatNet_final.pth')
    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60
    print(f"\nFinal Training Time: {time_elapsed} minutes\n")
        
if __name__ == "__main__":
    main()
