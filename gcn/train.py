import torch 
import torch.nn as nn 
import torch.optim as optim 

from model import LatNet
from dataset import load_data

from tqdm import tqdm

"""Training Information """
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
BATCH_SIZE = 100

"""Dataset Information """
DATA_DIR = '../training_data/data/task_7'
INPUT_FEATURES = 2                                             #Node Level Features
NUM_NODES = 9

torch.manual_seed(1)

def train_fn(train_loader, model, optimizer, loss_fn): 
    loop = tqdm(train_loader, leave=True)

    for batch_idx, data in enumerate(loop):
        output = model(data)
        loss = loss_fn(output.view(-1), data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def validation_fn(test_loader, model, loss_fn, epoch):
    mean_loss = []
    for data in test_loader:
        output = model(data)
        loss = loss_fn(output.view(-1), data.y)
        mean_loss.append(loss.item()) 
    
    print(f"[{epoch}] Validation MSE is {sum(mean_loss)/len(mean_loss)}")


def main():
    train_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE)

    model = LatNet(INPUT_FEATURES, NUM_NODES, BATCH_SIZE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    for epoch in range(EPOCHS): 
        train_fn(train_loader, model, optimizer, loss_fn)
        validation_fn(test_loader, model, loss_fn, epoch)
        
if __name__ == "__main__":
    main()