import torch 
import torch.nn as nn 
from .dataset import load_data
import matplotlib.pyplot as plt

"""Training Information """
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

"""Dataset Information """
DATA_DIR = 'training_data/data/training_data'

INPUT_FEATURES = 4 #Node Level Features
NUM_NODES = 9

"""Load Model"""
# 1100 -> Best Model yet
MODEL_PATH = "gcn/new_model/LatNet_1150.pth"

torch.manual_seed(1)

def test_fn(test_loader, model, num_examples=10):

    pred_list = []
    label_list = []
    error_list = []

    for idx, data in enumerate(test_loader):
        output = model(data).item()
        # print(data)
        ground_truth = data.y.item()
        error = abs(ground_truth-output)

        print(f"Predicton: {output}, Actual: {ground_truth}, Error: {error}")

        pred_list.append(output)
        label_list.append(ground_truth)
        error_list.append(error)

        if ((idx + 1) ==  num_examples):
            break


    return pred_list, label_list, error_list
    
def plot_test(pred_list, label_list, error_list):
    fig, ax = plt.subplots()
    ax.scatter(range(len(pred_list)), pred_list, label='Prediction', color='blue', marker='o')
    ax.scatter(range(len(label_list)), label_list, label='Label', color='red', marker='^')
    ax.set_xlabel('Index')
    ax.set_ylabel('Latency')
    ax.legend(loc='upper left')
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.bar(range(len(error_list)), error_list, alpha=0.5, color='green', label='Error')
    ax2.set_ylabel('Error')
    ax2.legend(loc='upper right')
    # ax2.set_ylim(0, 500)
    plt.show()


def main():
    _, test_loader = load_data(DATA_DIR, BATCH_SIZE)
    print(len(test_loader))

    model = torch.load(MODEL_PATH).to(torch.device('cpu'))
    # model = LatNet(in_features=4, num_nodes=9).to("cpu")
    # model.load_state_dict(model_state_dict)

    pred_list, label_list, error_list = test_fn(test_loader, model, num_examples=100)
    print(f"\nAverage Test Error is {sum(error_list)/len(error_list)}")
    plot_test(pred_list, label_list, error_list)
        
if __name__ == "__main__":
    main()