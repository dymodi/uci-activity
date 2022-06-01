import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# Read Data
df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")
# display(df_train.head(5))

# Visualize data
# display(df_train.head(5))
# display(df_train.describe())
# df_train.hist(bins=20, figsize=(30,30))
# plt.show()
# display(df_test.head(5))
# display(df_test.describe())

# Pytorch Data Loader
class DatasetFromPandas(Dataset):
    def __init__(self, df):
        ## Label processing, standing or not
        df['label_standing'] = df.apply(lambda row: 1 if row[562] == 'WALKING_UPSTAIRS' else 0, axis=1)

        x = df.iloc[:, 0:500].values
        y = df['label_standing'].values

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


my_train = DatasetFromPandas(df_train)
my_test = DatasetFromPandas(df_test)
batch_size = 40

print("my_train.y", my_train.y)
print("my_test.y", my_test.y)

# Create data loaders.
train_dataloader = DataLoader(my_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(my_test, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)  # N: batch size, C: channel, H: height, W: width
    print("Shape of y: ", y.shape, y.dtype)
    break

# Creating Models
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
# Simple nn with only linear and relu layer
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()    ## Flattens a contiguous range of dims into a tensor. For use with Sequential.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(500, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# Optimizing the Model Parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation (three standard steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # print(pred)


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")


