
import torch
from model import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load("../weights/mnist.pt"))
model.eval()
model_input = torch.zeros(1,1,28,28)
torch.onnx.export(model, model_input, '../weights/mnist.onnx',
                                    export_params=True,
                                    verbose=True,
                                    opset_version=13,
                                    input_names=["input"],
                                    output_names=["output"])
