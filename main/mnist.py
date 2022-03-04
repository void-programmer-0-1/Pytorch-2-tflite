import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

epochs = 30
batch_size = 100
lr = 0.001

MEAN = 0.1307
STANDARD_DEVIATION = 0.3081

train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomAffine(
                               degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                               shear=(-30, 30, -30, 30)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2) 
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


model = NeuralNetwork()
model.train()

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(epochs):

    for batch_id,(image,label) in enumerate(train_loader):

        prediction = model(image)
        loss = loss_fn(prediction,label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if ((batch_id + 1) % 100 == 0):
                print(f"EPOCH [{epoch + 1}/{epochs}] STEP [{batch_id + 1}/{len(train_loader)}] LOSS {loss.item():.4f}")
    

torch.save(model.state_dict(),"../weights/mnist.pt")

model.eval()

with torch.no_grad():

    n_correct = 0
    n_sample = 0

    for image,label in test_loader:

        output = model(image)
        _,predictions = torch.max(output,1)
        n_sample += label.shape[0]
        n_correct += (predictions == label).sum().item()

    acc = 100 * n_correct / n_sample
    print(f'accuracy = {acc}')

