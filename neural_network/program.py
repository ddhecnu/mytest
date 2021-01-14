# import neuralnetwork as nn
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
with np.load('mnist.npz') as data:
    print(data.files)
exit()

# layer_sizes = (300,150,100,50,10)
layer_sizes = (3,5,10)
x = np.ones((layer_sizes[0],1))

# print(x)

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)

# print(prediction)