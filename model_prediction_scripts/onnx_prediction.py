
import onnxruntime
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


session = onnxruntime.InferenceSession("../weights/mnist.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = np.array(session.run(None, {input_name: image[None,...]}))

img_index  = result.argmax()
print(img_index)
print(label)