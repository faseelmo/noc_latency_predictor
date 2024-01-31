import torch 
import torch.nn as nn 

from .model import LatNet
from .dataset import load_data

import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Training Information """
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

"""Dataset Information """
DATA_DIR = 'training_data/data/task_7_trained'
INPUT_FEATURES = 4 #Node Level Features
NUM_NODES = 9

"""Load Model"""
MODEL_PATH = "gcn/results/epoch_500_L1/LatNet_200.pth"

torch.manual_seed(1)

def test_fn(test_loader, model, num_examples=10):

    for idx, data in enumerate(test_loader):
        output = model(data).item()
        ground_truth = data.y.item()
        error = abs(ground_truth-output)
        print(f"Predicton: {output}, Actual: {ground_truth}, Error: {error}")
        if ((idx + 1) ==  num_examples):
            break
    
        
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
    train_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE)

    model_state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model = LatNet(in_features=4, num_nodes=9).to("cpu")
    model.load_state_dict(model_state_dict)

    # valid_loss_list = []
    valid_loss = test_fn(test_loader, model, num_examples=100)


        
if __name__ == "__main__":
    main()
