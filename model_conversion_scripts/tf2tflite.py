
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("../weights/mnist/")
tflite_model = converter.convert()

with open("../weights/mnist.tflite","wb") as f:
    f.write(tflite_model)

