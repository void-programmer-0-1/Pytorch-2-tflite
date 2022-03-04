
import onnx
from onnx_tf.backend import prepare
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision import datasets

loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                ])),batch_size=100, shuffle=True)

inputs_batch, labels_batch = next(iter(loader))

index = 10

image = inputs_batch[index].cpu().detach().numpy()
label = labels_batch[index]

onnx_model = onnx.load("../weights/mnist.onnx")  
tf_rep = prepare(onnx_model) 

prediction = np.array(tf_rep.run(image[None,...]))[0].argmax(axis=1)
print(prediction)
print(label)