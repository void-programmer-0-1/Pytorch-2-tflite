# https://www.tensorflow.org/lite/guide/inference

import numpy as np
import tensorflow as tf

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


interpreter = tf.lite.Interpreter(model_path="../weights/mnist.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
img = image.reshape(input_shape)

interpreter.set_tensor(input_details[0]["index"],img)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"])
output_data = np.argmax(output_data,axis=1)
print(output_data)
print(label)