import numpy as np
import math
import itertools
import random

class ActivationFunction:
    """Constructor.

    Args:
        f: activation function
        df: activation function derivative
    """
    def __init__(self, f, df):
        self.f = f
        self.df = df

    """Create sigmoid activation function.

    Returns:
        ActivationFunction: sigmoid activation function.
    """
    @staticmethod
    def create_sigmoid():
        return ActivationFunction(lambda x: 1.0 / (1.0 + math.exp(-x)), lambda y: y * (1 - y))

    """Create identity activation function.

    Returns:
        ActivationFunction: identity activation function.
    """
    @staticmethod
    def create_identity():
        return ActivationFunction(lambda x: x, lambda y: 1)


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
        self.ac_funcs = [activation_func_creator() for _ in range(num_nodes)]
        self.biases = np.array([bias_creator() for _ in range(num_nodes)])
        self.weights = np.array([[weight_creator() for _ in range(
            num_nodes_next_layer)] for _ in range(num_nodes)])

    """Number of nodes in the layer"""
    def __len__(self):
        return self.biases.shape[1]

    """Layer info"""
    def __str__(self):
        return ""


class NeuralNetwork:

    def __init__(self, layers, ac_funcs):
        assert(len(layers) == len(ac_funcs))
        assert(len(layers) > 1)

        self.layers = [Layer(layers[0], layers[1], ac_funcs[0],
                             lambda: 0.0, self.create_weight)]

        for i in range(1, len(layers)):
            next_l = layers[i+1] if i < len(layers) else 1
            self.layers.append(
                Layer(layers[i], next_l, ac_funcs[i], self.create_bias, self.create_weight))

    def predict(self, input):
        return self.feedforward(input)[-1]
    
    def feedforward(self, input):
        pass

    def train(self, input, target, learning_rate=0.1):
        pass
    
    @staticmethod
    def create_bias():
        return random.random()

    @staticmethod
    def create_weight():
        return 2.0 * random.random() - 1.0
            
