import torch 
import torch.nn as nn 
import torch.optim as optim 

from model import LatNet
from dataset import load_data

import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Training Information """
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0
BATCH_SIZE = 128

"""Dataset Information """
DATA_DIR = '../training_data/data/task_7'
INPUT_FEATURES = 4                                             #Node Level Features
NUM_NODES = 9

"""Load Model"""
LOAD_MODEL = False
MODEL_PATH = "results/"

torch.manual_seed(1)

def train_fn(train_loader, model, optimizer, loss_fn): 
    loop = tqdm(train_loader, leave=True)

    mean_loss = []
    for batch_idx, data in enumerate(loop):
        output = model(data.to(DEVICE)).to(DEVICE)
        loss = loss_fn(output.view(-1), data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

    train_loss = sum(mean_loss)/len(mean_loss) 
    return train_loss


def validation_fn(test_loader, model, loss_fn, epoch):
    mean_loss = []
    for data in test_loader:
        output = model(data.to(DEVICE)).to(DEVICE)
        loss = loss_fn(output.view(-1), data.y)
        mean_loss.append(loss.item()) 
    
    validation_set_loss = sum(mean_loss)/len(mean_loss)
    print(f"[{epoch}] Validation MAE is {validation_set_loss}")
    return validation_set_loss

def plot_and_save_loss(train_loss, valid_loss):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, valid_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('validation_plot.png')
    plt.clf()

    loss_dict = {
        'train_loss' : train_loss,
        'valid_loss' : valid_loss,
    }
    with open('loss_dict.pkl', 'wb') as file:
        pickle.dump(loss_dict, file)

def main():
    start_time = time.time()
    train_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE)

    model = LatNet(INPUT_FEATURES, NUM_NODES).to(DEVICE)

    if LOAD_MODEL:
        model_state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(model_state_dict)
        print(f"\nPre-Trained Wieghts Loaded\n")

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.L1Loss()

    valid_loss_list = []
    train_loss_list = []
    for epoch in range(EPOCHS): 
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        valid_loss = validation_fn(test_loader, model, loss_fn, epoch)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        plot_and_save_loss(train_loss_list, valid_loss_list)

    torch.save(model.state_dict(), 'LatNet.pth')
    end_time = time.time()
    
    time_elapsed = (end_time - start_time) / 60
    print(f"\nTraining Time: {time_elapsed} minutes\n")
        
if __name__ == "__main__":
    main()
