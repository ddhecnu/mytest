import numpy as np
# import torch

class NeuralNetwork:
    def __init__(self, layer_sizes):
        # layer_sizes = (2,3,5,2)
        w_shapes = [(a,b) for a,b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in w_shapes]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]
        print(w_shapes)
        # for w in weights:
        #     print(w,'\n')
    
    def predict(self, a):
        for w,b in zip(self.weights, self.biases):
            z = np.matmul(w,a) + b
            print(z[0])
            a = self.activation(np.matmul(w,a) + b)
        return a 

    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))
