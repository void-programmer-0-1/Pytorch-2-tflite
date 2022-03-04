
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("../weights/mnist.onnx")  
tf_rep = prepare(onnx_model)                                                        # prepare tf representation
tf_rep.export_graph("../weights/mnist/")                                   # export the model (SavedModel format)



# https://stackoverflow.com/questions/58834684/how-could-i-convert-onnx-model-to-tensorflow-saved-model
# https://github.com/onnx/onnx-tensorflow

