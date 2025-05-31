import random
from math import exp, log

import mnist

random.seed(42)
mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


class Convolution:
    def __init__(self, num_filters, filter_size, activation=None):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.filters = [
            [
                [random.uniform(-1, 1) / 9 for _ in range(self.filter_size)] for _ in range(self.filter_size)
            ]
            for _ in range(num_filters)
        ]

    def iterate_regions(self, image):
        h = len(image)
        w = len(image[0])

        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                region = [
                    [image[i + x][j + y] for y in range(self.filter_size)] for x in range(self.filter_size)
                ]
                yield region, i, j

    @staticmethod
    def relu(x):
        return max(0, x)

    def forward(self, input):
        self.last_input = input

        h = len(input)
        w = len(input[0])
        output = [
            [
                [0.0 for _ in range(self.num_filters)] for _ in range(w - self.filter_size + 1)
            ] for _ in range(h - self.filter_size + 1)
        ]

        for region, i, j in self.iterate_regions(input):
            for f in range(self.num_filters):
                s = 0.0
                for x in range(self.filter_size):
                    for y in range(self.filter_size):
                        s += region[x][y] * self.filters[f][x][y]
                if self.activation == 'relu':
                    s = self.relu(s)
                output[i][j][f] = s
        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = [
            [
                [0.0 for _ in range(self.filter_size)] for _ in range(self.filter_size)
            ] for _ in range(self.num_filters)
        ]

        for region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):

                s = 0.0
                for x in range(self.filter_size):
                    for y in range(self.filter_size):
                        s += region[x][y] * self.filters[f][x][y]
                grad = d_L_d_out[i][j][f]

                if self.activation == 'relu':
                    grad = grad if s > 0 else 0

                for x in range(self.filter_size):
                    for y in range(self.filter_size):
                        d_L_d_filters[f][x][y] += grad * region[x][y]

        for f in range(self.num_filters):
            for x in range(self.filter_size):
                for y in range(self.filter_size):
                    self.filters[f][x][y] -= learn_rate * d_L_d_filters[f][x][y]

        return None


class MaxPool:
    def __init__(self, pool_size):
        self.size = pool_size

    def iterate_regions(self, image):
        h = len(image)
        w = len(image[0])

        new_h = h // self.size
        new_w = w // self.size
        for i in range(new_h):
            for j in range(new_w):
                region = [
                    [image[i * self.size + x][j * self.size + y] for y in range(self.size)] for x in range(self.size)
                ]
                yield region, i, j

    def forward(self, input):
        self.last_input = input
        h = len(input)
        w = len(input[0])
        num_filters = len(input[0][0])

        output = [
            [
                [0.0 for _ in range(num_filters)] for _ in range(w // self.size)
            ] for _ in range(h // self.size)
        ]

        for region, i, j in self.iterate_regions(input):
            for f in range(num_filters):
                max_val = region[0][0][f]
                for x in range(self.size):
                    for y in range(self.size):
                        if region[x][y][f] > max_val:
                            max_val = region[x][y][f]
                output[i][j][f] = max_val
        return output

    def backprop(self, d_L_d_out):
        h = len(self.last_input)
        w = len(self.last_input[0])
        num_filters = len(self.last_input[0][0])

        d_L_d_input = [
            [
                [0.0 for _ in range(num_filters)] for _ in range(w)
            ] for _ in range(h)
        ]

        for region, i, j in self.iterate_regions(self.last_input):
            for f in range(num_filters):
                max_val = region[0][0][f]
                max_x, max_y = 0, 0
                for x in range(self.size):
                    for y in range(self.size):
                        if region[x][y][f] > max_val:
                            max_val = region[x][y][f]
                            max_x, max_y = x, y
                d_L_d_input[i * self.size + max_x][j * self.size + max_y][f] = d_L_d_out[i][j][f]
        return d_L_d_input


class Dense:
    def __init__(self, input_len, nodes):
        self.input_len = input_len
        self.nodes = nodes
        self.weights = [
            [random.uniform(-1, 1) / input_len for _ in range(nodes)] for _ in range(input_len)
        ]
        self.biases = [0.0 for _ in range(nodes)]

    def forward(self, input):
        self.last_input_shape = (len(input), len(input[0]), len(input[0][0]))
        flatten = []

        for i in range(len(input)):
            for j in range(len(input[0])):
                for k in range(len(input[0][0])):
                    flatten.append(input[i][j][k])
        self.last_input = flatten

        totals = [0.0 for _ in range(self.nodes)]
        for i in range(self.nodes):
            for j in range(self.input_len):
                totals[i] += flatten[j] * self.weights[j][i]
            totals[i] += self.biases[i]
        self.last_totals = totals

        exp_totals = [exp(x) for x in totals]
        S = sum(exp_totals)
        out = [x / S for x in exp_totals]

        return out

    def backprop(self, dL_dout, learn_rate):
        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue

            # e^totals
            t_exp = [exp(x) for x in self.last_totals]

            # Sum of all e^totals
            S = sum(t_exp)

            # Gradients of out[i] against totals
            dout_dt = [-t_exp[i] * t_exp[j] / (S ** 2) for j in range(self.nodes)]
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            dt_dw = self.last_input
            dt_db = 1
            dt_dinputs = [self.weights[j][i] for j in range(self.input_len)]

            # Gradients of loss against totals
            dL_dt = [gradient * dout_dt[j] for j in range(self.nodes)]

            # Gradients of loss against weights/biases/input
            dL_dw = [[dt_dw[j] * dL_dt[k] for k in range(self.nodes)] for j in range(self.input_len)]
            dL_db = [dL_dt[k] * dt_db for k in range(self.nodes)]
            dL_dinputs = [sum(self.weights[j][k] * dL_dt[k] for k in range(self.nodes)) for j in
                          range(self.input_len)]

            for j in range(self.input_len):
                for k in range(self.nodes):
                    self.weights[j][k] -= learn_rate * dL_dw[j][k]
            for k in range(self.nodes):
                self.biases[k] -= learn_rate * dL_db[k]

            # Reshape d_L_d_inputs to last_input_shape
            idx = 0
            out = []
            for i1 in range(self.last_input_shape[0]):
                mat = []
                for i2 in range(self.last_input_shape[1]):
                    row = []
                    for i3 in range(self.last_input_shape[2]):
                        row.append(dL_dinputs[idx])
                        idx += 1
                    mat.append(row)
                out.append(mat)
            return out
        return None


class CNN:
    def __init__(self, num_filters, filter_size, activation, pool_size):
        self.conv = Convolution(num_filters, filter_size, activation)
        self.pool = MaxPool(pool_size)

        input_width = (28 - filter_size + 1) // pool_size
        input_height = (28 - filter_size + 1) // pool_size
        input_depth = num_filters
        input_len = input_width * input_height * input_depth

        self.softmax = Dense(input_len, 10)

    def forward(self, image, label):
        # Normalize image to [-0.5, 0.5]
        norm_img = [[(pixel / 255.0) - 0.5 for pixel in row] for row in image]
        out = self.conv.forward(norm_img)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        loss = -log(out[label])
        acc = 1 if out.index(max(out)) == label else 0
        return out, loss, acc

    def train(self, im, label, lr=0.005):
        out, loss, acc = self.forward(im, label)

        gradient = [0.0 for _ in range(10)]
        gradient[label] = -1 / out[label]
        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        self.conv.backprop(gradient, lr)

        return loss, acc


with open("config.txt") as f:
    conv = f.readline()
    pool = f.readline()
num_filters = int(conv.split()[0])
filter_size = int(conv.split()[1])
activation = conv.split()[2]
pool_size = int(pool.split()[0])

print(f'Convolution: {num_filters} filters of size {filter_size} using {activation}, Pooling: size {pool_size}')

cnn = CNN(num_filters, filter_size, activation, pool_size)
# Main training/testing loop
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
for epoch in range(3):
    print(f'--- Epoch {epoch + 1} ---')
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(f'[Step {i + 1}] Past 100 steps: Average Loss {loss / 100:.3f} | Accuracy: {num_correct}%')
            loss = 0
            num_correct = 0
        l, acc = cnn.train(im, label)
        loss += l
        num_correct += acc
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = cnn.forward(im, label)
    loss += l
    num_correct += acc
num_tests = len(test_images)
print(f'Test Loss: {loss / num_tests:.4f}')
print(f'Test Accuracy: {num_correct / num_tests:.4f}')
