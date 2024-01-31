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
DATA_DIR = 'gcn/data/1585'

INPUT_FEATURES = 4 #Node Level Features
NUM_NODES = 9

"""Load Model"""
MODEL_PATH = "gcn/results/epoch_1000_L1/LatNet.pth"

torch.manual_seed(1)

def test_fn(test_loader, model, num_examples=10):

    pred_list = []
    label_list = []
    error_list = []

    for idx, data in enumerate(test_loader):
        output = model(data).item()
        ground_truth = data.y.item()
        error = abs(ground_truth-output)

        print(f"Predicton: {output}, Actual: {ground_truth}, Error: {error}")

        pred_list.append(output)
        label_list.append(ground_truth)
        error_list.append(error)

        # if ((idx + 1) ==  num_examples):
        #     break
    return pred_list, label_list, error_list
    
def plot_scatter(pred_list, label_list):
    fig, ax = plt.subplots()
    ax.scatter(range(len(pred_list)), pred_list, label='Prediction', color='blue', marker='o')
    ax.scatter(range(len(label_list)), label_list, label='Label', color='red', marker='^')

    ax.set_xlabel('Index')
    ax.set_ylabel('Latency')
    ax.legend()
    ax.grid(True)
    plt.show()

def main():
    train_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE)

    model_state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model = LatNet(in_features=4, num_nodes=9).to("cpu")
    model.load_state_dict(model_state_dict)

    pred_list, label_list, error_list = test_fn(test_loader, model, num_examples=1000)
    plot_scatter(pred_list, label_list)
        
if __name__ == "__main__":
    main()
