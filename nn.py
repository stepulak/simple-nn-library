import numpy as np
import math
import itertools
import random
from datetime import datetime

class ActivationFunction:
    """Constructor.

    Args:
        f: activation function
        df: activation function derivative
        textinfo: info about function
    """
    def __init__(self, f, df, textinfo):
        self.f = f
        self.df = df
        self.textinfo = textinfo

    """Print info about activation function."""
    def __str__(self):
        return self.textinfo

    """Create sigmoid activation function.

    Returns:
        ActivationFunction: sigmoid activation function.
    """
    @staticmethod
    def create_sigmoid():
        return ActivationFunction(lambda x: 1.0 / (1.0 + math.exp(-x)), lambda y: y * (1 - y), "sigmoid")

    """Create identity activation function.

    Returns:
        ActivationFunction: identity activation function.
    """
    @staticmethod
    def create_identity():
        return ActivationFunction(lambda x: x, lambda y: 1, "identity")


class Layer:

    """Constructor.

    Args:
        num_nodes: number of nodes in this layer
        num_nodes_next_layer: number of nodes in next layer
        activation_func_creator: creator of activation function
        bias_creator: creator of single bias
        weight_creator: creator of single weight
    """
    def __init__(self, num_nodes, num_nodes_next_layer,
                 activation_func_creator, bias_creator, weight_creator):
        self.ac_funcs = np.array([activation_func_creator()
                                  for _ in range(num_nodes)])
        self.biases = np.array([[bias_creator() for _ in range(num_nodes)]])
        self.weights = np.array([[weight_creator() for _ in range(
            num_nodes_next_layer)] for _ in range(num_nodes)])

    """Number of nodes in the layer"""
    def __len__(self):
        return self.biases.shape[1]

    """Layer info"""
    def __str__(self):
        return "\nLayer info:\nNodes: " + str(len(self)) + "\n" + \
            "Biases:\n" + str(self.biases) + "\n" + \
            "Weights:\n" + str(self.weights) + "\n" + \
            "Activation functions:\n" + \
            "".join([str(f)+", " for f in self.ac_funcs]) + "\n"


class NeuralNetwork:
    def __init__(self, layers, ac_funcs):
        assert(len(layers) == len(ac_funcs))
        assert(len(layers) > 1)

        self.layers = [Layer(layers[0], layers[1], ac_funcs[0],
                             lambda: 0.0, self.create_weight)]

        for i in range(1, len(layers)):
            next_l = layers[i+1] if i+1 < len(layers) else 1
            layer = Layer(layers[i], next_l, ac_funcs[i],
                          self.create_bias, self.create_weight)
            self.layers.append(layer)

    def predict(self, input):
        return self.feedforward(input)[-1]

    def feedforward(self, input):
        assert(len(input) == len(self.layers[0]))

        output = np.array([input])
        outputs = []

        for i in range(len(self.layers)):
            layer = self.layers[i]
            output = output + layer.biases
            output = np.array([y.f(x) for x, y in zip(output[0], layer.ac_funcs)])
            outputs.append(output)

            if i + 1 < len(self.layers):
                output = output * layer.weights

        return outputs

    def train(self, input, target, learning_rate=0.1):
        outputs = self.feedforward(input)
        error = np.array(target) - outputs[-1]

        # reverse iteration
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer_o = outputs[i]
            next_layer_o = outputs[i + 1]
            next_layer_dfunc = [x.df for x in next_layer.ac_funcs]

            # Count gradient
            error = np.vectorize(lambda x: x * learning_rate)(error)
            gradient = [f(x) for f, x in zip(next_layer_dfunc, next_layer_o)]
            gradient = np.multiply(error, np.array(gradient))[0]
            
            # Count deltas
            weight_deltas = np.array([gradient.transpose() * layer_o]).transpose()
            bias_deltas = gradient.transpose() # just the gradient

            # Count new error
            error = (layer.weights * error.transpose()).transpose()
            
            layer.weights += weight_deltas
            next_layer.biases += bias_deltas

    def __str__(self):
        return "".join([str(x) for x in self.layers])

    @staticmethod
    def create_bias():
        return random.random()

    @staticmethod
    def create_weight():
        return 2.0 * random.random() - 1.0

random.seed(42) #datetime.now())
nn = NeuralNetwork([2, 3, 4, 1], [ActivationFunction.create_identity, ActivationFunction.create_sigmoid,
                               ActivationFunction.create_sigmoid, ActivationFunction.create_sigmoid])
print(nn.train([1, 0], [1]))
print(nn)
