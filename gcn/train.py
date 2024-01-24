import torch 
import torch.nn as nn 
import torch.optim as optim 

from model import LatNet
from dataset import load_data

from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

"""Training Information """
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0
BATCH_SIZE = 128

"""Dataset Information """
DATA_DIR = '../training_data/data/task_7'
INPUT_FEATURES = 4                                             #Node Level Features
NUM_NODES = 9

torch.manual_seed(1)

def train_fn(train_loader, model, optimizer, loss_fn): 
    loop = tqdm(train_loader, leave=True)

    for batch_idx, data in enumerate(loop):
        output = model(data.to(DEVICE)).to(DEVICE)
        loss = loss_fn(output.view(-1), data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def validation_fn(test_loader, model, loss_fn, epoch):
    mean_loss = []
    for data in test_loader:
        output = model(data.to(DEVICE)).to(DEVICE)
        loss = loss_fn(output.view(-1), data.y)
        mean_loss.append(loss.item()) 
    
    validation_set_loss = sum(mean_loss)/len(mean_loss)
    print(f"[{epoch}] Validation MAE is {validation_set_loss}")
    return validation_set_loss

def plot_and_save_loss(epoch_loss_list):
    epochs, losses = zip(*epoch_loss_list)

    plt.plot(epochs, losses, label='Loss')
    plt.title('Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('validation_plot.png')

    with open('validation_loss.pkl', 'wb') as file:
        pickle.dump(epoch_loss_list, file)

def main():
    train_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE)

    model = LatNet(INPUT_FEATURES, NUM_NODES).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.L1Loss()

    validation_loss = []
    for epoch in range(EPOCHS): 
        train_fn(train_loader, model, optimizer, loss_fn)
        loss = validation_fn(test_loader, model, loss_fn, epoch)
        validation_loss.append((epoch, loss))

    torch.save(model.state_dict(), 'LatNet.pth')
    plot_and_save_loss(validation_loss)
        
if __name__ == "__main__":
    main()