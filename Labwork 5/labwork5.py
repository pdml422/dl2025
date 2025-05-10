import random
from math import exp, log
import matplotlib.pyplot as plt

random.seed(42)

# --- Activation and Loss Functions ---
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def loss(y_true, y_pred):
    return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))


# --- Neuron, Layer, and Network Classes ---
class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.output = 0.0
        self.input = []
        self.z = 0.0
        self.grad_w = [0.0] * len(weight)
        self.grad_b = 0.0

    def activate(self, inputs):
        self.input = inputs
        self.z = sum(w * i for w, i in zip(self.weight, inputs)) + self.bias
        self.output = sigmoid(self.z)
        return self.output


class Layer:
    def __init__(self, weight_list, bias_list):
        self.neurons = [Neuron(w, b) for w, b in zip(weight_list, bias_list)]

    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]


class Network:
    def __init__(self, num_neurons):
        self.layers = []
        for i in range(1, len(num_neurons)):
            weight_layer = [[random.uniform(-1, 1) for _ in range(num_neurons[i - 1])]
                            for _ in range(num_neurons[i])]
            bias_layer = [random.uniform(-1, 1) for _ in range(num_neurons[i])]
            self.layers.append(Layer(weight_layer, bias_layer))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, expected):
        last_layer = self.layers[-1]
        for i, neuron in enumerate(last_layer.neurons):
            y_pred = neuron.output
            y_true = expected[i]
            dL_dz = (y_pred - y_true) * y_pred * (1 - y_pred)
            for j in range(len(neuron.weight)):
                neuron.grad_w[j] = dL_dz * neuron.input[j]
            neuron.grad_b = dL_dz

        for l in reversed(range(len(self.layers) - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            for i, neuron in enumerate(layer.neurons):
                downstream = sum(n.weight[i] * n.grad_b for n in next_layer.neurons)
                dL_dz = downstream * neuron.output * (1 - neuron.output)
                for j in range(len(neuron.weight)):
                    neuron.grad_w[j] = dL_dz * neuron.input[j]
                neuron.grad_b = dL_dz

    def update_weights(self, lr):
        for layer in self.layers:
            for neuron in layer.neurons:
                for j in range(len(neuron.weight)):
                    neuron.weight[j] -= lr * neuron.grad_w[j]
                neuron.bias -= lr * neuron.grad_b


def grad_desc(net, data, lr, epochs):
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in data:
            output = net.forward(x)
            total_loss += loss(y[0], output[0])
            net.backward(y)
            net.update_weights(lr)
        losses.append(total_loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    return losses


if __name__ == '__main__':
    xor_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    nn = Network([2, 2, 1])
    losses = grad_desc(nn, xor_data, lr=0.1, epochs=10000)

    for x, y in xor_data:
        output = nn.forward(x)
        print(f"Input: {x}, Expected: {y[0]}, Predicted: {round(output[0], 4)}")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.show()
