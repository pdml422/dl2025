import random
from math import exp

random.seed(42)


class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.output = 0.0

    def activate(self, inputs):
        linear_sum = sum(w * i for w, i in zip(self.weight, inputs)) + self.bias
        # self.output = self.sigmoid(linear_sum)
        self.output = 1 if self.sigmoid(linear_sum) > 0.5 else 0
        return self.output

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + exp(-x))


class Layer:
    def __init__(self, weight_list, bias_list):
        self.neurons = [Neuron(w, b) for w, b in zip(weight_list, bias_list)]

    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]


class Network:
    def __init__(self, weights, biases):
        self.layers = [Layer(w_layer, b_layer) for w_layer, b_layer in zip(weights, biases)]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


if __name__ == '__main__':
    # with open("nn.txt", "r") as f:
    #     lines = f.read().splitlines()
    #     num_layers = int(lines[0])
    #     num_neurons = [int(i) for i in lines[1:]]

    num_layers = 3
    num_neurons = [2, 2, 1]

    weights = [
        [[-1.0, -1.0], [1.0, 1.0]],
        [[1.0, 1.0]]
    ]

    biases = [
        [1.5, -0.5],
        [-1.5]
    ]

    inputs = [1.0, 0.0]

    # weights = []
    # biases = []
    # for i in range(1, len(num_neurons)):
    #     prev_layer_size = num_neurons[i - 1]
    #     current_layer_size = num_neurons[i]
    #
    #     weight_layer = [[random.uniform(-1, 1) for _ in range(prev_layer_size)] for _ in range(current_layer_size)]
    #     bias_layer = [random.uniform(-1, 1) for _ in range(current_layer_size)]
    #
    #     weights.append(weight_layer)
    #     biases.append(bias_layer)
    #
    # inputs = [random.uniform(-1, 1) for _ in range(num_neurons[0])]

    nn = Network(weights, biases)
    output = nn.forward(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {output}")
