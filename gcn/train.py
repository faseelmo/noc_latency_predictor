import torch 
import torch.nn as nn 
import torch.optim as optim 

from .model import LatNet
from .dataset import load_data

import os 
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Training Information """
EPOCHS = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0
BATCH_SIZE = 1

"""Dataset Information """
DATA_DIR = 'training_data/data/training_data'
# INPUT_FEATURES = 4                                             #Node Level Features
NUM_NODES = 9

"""Load Model"""
LOAD_MODEL = False
MODEL_PATH = "gcn/results/"

torch.manual_seed(1)

SAVE_RESULTS = "gcn/new_model/"

if not os.path.exists(SAVE_RESULTS):
    os.makedirs(SAVE_RESULTS)
    print(f"Folder '{SAVE_RESULTS}' created.")
else:
    print(f"Folder '{SAVE_RESULTS}' already exists.")

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

def test_fn(test_loader, model, data_set, num_examples=10):
    pred_list = []
    label_list = []
    error_list = []
    for idx, data in enumerate(test_loader):
        data = data.to(DEVICE)

        output = model(data).to(DEVICE).item()
        ground_truth = data.y.item()
        error = abs(ground_truth-output)

        pred_list.append(output)
        label_list.append(ground_truth)
        error_list.append(error)

    plot_test(pred_list, label_list, name="test_results")
    
def plot_test(pred_list, label_list, error_list):
    fig, ax = plt.subplots()
    ax.scatter(range(len(pred_list)), pred_list, label='Prediction', color='blue', marker='o')
    ax.scatter(range(len(label_list)), label_list, label='Label', color='red', marker='^')
    ax.set_xlabel('Index')
    ax.set_ylabel('Latency')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_ylim(500, 17500)

    ax2 = ax.twinx()
    ax2.bar(range(len(error_list)), error_list, alpha=0.5, color='green', label='Error')
    ax2.set_ylabel('Error')
    ax2.legend(loc='upper right')
    plt.show()

    plt.savefig(f'{SAVE_RESULTS}/validation_plot_log.png')
    plt.clf()

def validation_fn(test_loader, model, loss_fn, epoch):
    mean_loss = []
    for data in test_loader:
        output = model(data.to(DEVICE)).to(DEVICE)
        loss = loss_fn(output.view(-1), data.y)
        mean_loss.append(loss.item()) 
    
    validation_set_loss = sum(mean_loss)/len(mean_loss)
    print(f"[{epoch+1}/{EPOCHS}] Validation MAE is {validation_set_loss}")
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
    train_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE)

    hidden_size = 16
    model = LatNet(NUM_NODES, hidden_size=hidden_size).to(DEVICE)

    print(f"Learning Rate is {LEARNING_RATE}")


    if LOAD_MODEL:
        model_state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(model_state_dict)
        print(f"\nPre-Trained Wieghts Loaded\n")

    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    valid_loss_list = []
    train_loss_list = []
    for epoch in range(EPOCHS): 
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        valid_loss = validation_fn(test_loader, model, loss_fn, epoch)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        plot_and_save_loss(train_loss_list, valid_loss_list)

        if (epoch+1) % 100 == 0:
            torch.save(model, f'{SAVE_RESULTS}/LatNet_{epoch+1}.pth')
            end_time = time.time()
            time_elapsed = (end_time - start_time) / 60
            print(f"\nTraining Time: {time_elapsed} minutes\n")

    torch.save(model.state_dict(), f'{SAVE_RESULTS}/LatNet_state_dict.pth')
    torch.save(model, 'gcn/current_model/LatNet_final.pth')
    end_time = time.time()
    time_elapsed = (end_time - start_time) / 60
    print(f"\nFinal Training Time: {time_elapsed} minutes\n")
        
if __name__ == "__main__":
    main()
