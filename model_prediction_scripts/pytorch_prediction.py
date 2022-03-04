
import torch
from model import NeuralNetwork
import torchvision.transforms as transforms
from torchvision import datasets

model = NeuralNetwork()
model.load_state_dict(torch.load("../weights/mnist.pt"))
model.eval()

loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                ])),batch_size=100, shuffle=True)

inputs_batch, labels_batch = next(iter(loader))

index = 10

image = inputs_batch[index]
label = labels_batch[index]

prediction = torch.argmax(model(image[None,...]),axis=1)
print(prediction)
print(label)

