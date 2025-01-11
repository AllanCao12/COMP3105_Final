# A4codes.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

def plotImg(x):
    img = x.reshape((84, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    return

def preparedata(file_path):
    data = pd.read_csv(file_path) # Read data
    labels = data.iloc[:, 0].values.astype(int) # Getting labels from first column of each row
    features = data.iloc[:, 1:].values / 255.0  # Normalize pixel values to [0, 1]

    # Splitting the images into 3
    topImages = features[:, :784]
    middleImages = features[:, 784:1568]
    bottomImages = features[:, 1568:]
    return topImages, middleImages, bottomImages, labels

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 channel since it's grayscaled, 32 kernels = 32 outputs, kernel size is 3x3, same padding. 


        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32 channels coming from conv1, 32 kernels = 32 outputs, kernel size is 3x3, same padding
        self.pool = nn.MaxPool2d(2, 2)  # (kernel_size, stride)

        # Dropout
        self.dropout= nn.Dropout(0.25)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 output classes for digits 0-9

    def forward(self, x):
        # Convolutional Layer 1
        x = F.relu(self.conv1(x)) # Since padding=1, same padding, the output shape is the same, but the number of parameters is now 32 (28,28,32)
        x = self.pool(x)          # Since stride=2 and kernel=2, (28-2)/2+1 = 14. Number of parameters doesn't change (14,14,32)
        x = self.dropout(x)  

        # Convolutional Layer 2
        x = F.relu(self.conv2(x)) # Since padding=1, same padding, but, in examples I saw online, double # of parameters, so 64. (14,14,64)
        x = self.pool(x) # Since stride=2 and kernel=2, (14-2)/2+1 = 7. Number of parameters doesn't change (7,7,64)
        x = self.dropout(x)

        # Flatten the tensor for fully connected layers
        x = x.flatten(1)

        # Fully Connected Layer 1
        x = F.relu(self.fc1(x)) # FC layer 1
        x = self.dropout(x)

        # Output Layer
        x = self.fc2(x)  # Getting the numbers, so shrink output to 10 (0-9)

        return x

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()

        self.digitIdentify = Net()

    def forward(self, topImg, middleImg, botImg):

        top = self.digitIdentify(topImg)
        middle = self.digitIdentify(middleImg)
        bottom = self.digitIdentify(botImg)

        # Predict the digit for the top image to determine parity
        topPred = top.argmax(dim=1) # Gets biggest index
        odd = (topPred % 2 == 1)

        # Convert to probabilities
        middleMax = F.log_softmax(middle, dim=1)  # Shape: (batch_size, 10) #nll wants it to be log
        bottomMax = F.log_softmax(bottom, dim=1)

        res = torch.where(
            odd.unsqueeze(1),  # Condition tensor
            middleMax,               # If condition is True
            bottomMax                # If condition is False
        )

        return res

def train(model, optimizer, train_dataloader):
    model.train()
    for topImg, middleImg, botImg, target in train_dataloader:
        topImg, middleImg, botImg, target = topImg.to(device), middleImg.to(device), botImg.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(topImg, middleImg, botImg)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def validate(model, test_dataloader):
    model.eval()
    m = len(test_dataloader.dataset)
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for topImg, middleImg, botImg, target in test_dataloader:
            topImg, middleImg, botImg, target = topImg.to(device), middleImg.to(device), botImg.to(device), target.to(device)
            output = model(topImg, middleImg, botImg)
            loss = F.nll_loss(output, target, reduction='sum')
            testLoss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / m
    testLoss /= m
    print(f'Test average loss: {testLoss:.4f}, accuracy: {accuracy:.3f}')
    return accuracy

def learn(x,y):

    topTrain = x[0]
    midTrain = x[1]
    botTrain = x[2]
    
    topTrain = torch.tensor(topTrain.reshape(-1, 1, 28, 28), dtype=torch.float32)
    midTrain = torch.tensor(midTrain.reshape(-1, 1, 28, 28), dtype=torch.float32)
    botTrain = torch.tensor(botTrain.reshape(-1, 1, 28, 28), dtype=torch.float32)
    labelsTrain = torch.tensor(y, dtype=torch.long)

    train_dataset = TensorDataset(topTrain, midTrain, botTrain, labelsTrain)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    topVal, midVal, botVal, labelsVal = preparedata('A4val.csv')

    topVal = torch.tensor(topVal.reshape(-1, 1, 28, 28), dtype=torch.float32)
    midVal = torch.tensor(midVal.reshape(-1, 1, 28, 28), dtype=torch.float32)
    botVal = torch.tensor(botVal.reshape(-1, 1, 28, 28), dtype=torch.float32)
    labelsVal = torch.tensor(labelsVal, dtype=torch.long)

    val_dataset = TensorDataset(topVal, midVal, botVal, labelsVal)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = Net2().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.Adagrad(model.parameters())
    n_epochs = 100
    startTime = time.time()
    bestAcc = 0.0
    for epoch in range(n_epochs):
        train(model, optimizer, train_dataloader)
        accuracy = validate(model, val_dataloader)
        print(f"Epoch {epoch+1}/{n_epochs}, Val Acc: {accuracy:.4f}")
        if accuracy > bestAcc:
            bestAcc = accuracy

    elapsedTime = time.time() - startTime
    print(f"Training Time: {elapsedTime:.3f} seconds")
    print(f"Best Validation Accuracy: {bestAcc:.4f}")

    return model

def classify(Xtest, model):

    model.eval()
    with torch.no_grad():
        # Split Xtest into three sub-images
        topImages = Xtest[:, :784]
        midImages = Xtest[:, 784:1568]
        bottomImages = Xtest[:, 1568:]

        topImages = torch.tensor(topImages.reshape(-1, 1, 28, 28), dtype=torch.float32).to(device) # Reshapes all the arrays from a shape of (n,784) to (n,1,28,28), where n is the number of data points
        midImages = torch.tensor(midImages.reshape(-1, 1, 28, 28), dtype=torch.float32).to(device)
        bottomImages = torch.tensor(bottomImages.reshape(-1, 1, 28, 28), dtype=torch.float32).to(device)

        outputs = model(topImages, midImages, bottomImages) # Calls Net2's forward function
        predictions = outputs.argmax(dim=1).cpu().numpy() # Converts the output to a numpy array and numpy is only compatible with cpu tensors
    return predictions

if __name__ == "__main__":
    print(torch.__version__)
    topTrain, midTrain, botTrain, labelsTrain = preparedata('A4train.csv')

    x = [topTrain, midTrain, botTrain]
    y = labelsTrain

    trainedModel = learn(x,y)

    top, mid, bot, y = preparedata('A4val.csv')
    X = np.concatenate((top, mid, bot), axis=1)
    yhat = classify(X, trainedModel)
    print("Predictions:", yhat)
