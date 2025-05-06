from math import log, exp
import matplotlib.pyplot as plt

file_path = "loan2.csv"

w0 = 0
w1 = 1
w2 = 2
x1 = []
x2 = []
y = []
l = []

with open(file_path, "r") as file:
    next(file)
    for line in file:
        x1.append(float(line.strip().split(",")[0]))
        x2.append(float(line.strip().split(",")[1]))
        y.append(int(line.strip().split(",")[2]))


def j(w0, w1, w2, x1, x2, y):
    N = len(x1)
    return -(1 / N) * sum(
        yi * (w1 * xi1 + w2 * xi2 + w0) - log(1 + exp(w1 * xi1 + w2 * xi2 + w0)) for xi1, xi2, yi in zip(x1, x2, y))


def sigmoid(x):
    return 1 / (1 + exp(-x))


def df0(w0, w1, w2, x1, x2, y):
    return 1 - y - sigmoid(-(w1 * x1 + w2 * x2 + w0))


def df1(w0, w1, w2, x1, x2, y):
    return -y * x1 + x1 * (1 - sigmoid(-(w1 * x1 + w2 * x2 + w0)))


def df2(w0, w1, w2, x1, x2, y):
    return -y * x2 + x2 * (1 - sigmoid(-(w1 * x1 + w2 * x2 + w0)))


def grad_desc(w0, w1, w2, x1, x2, y, lr, times):
    for i in range(times):
        N = len(x1)
        w0 = w0 - lr * sum(df0(w0, w1, w2, xi1, xi2, yi) for xi1, xi2, yi in zip(x1, x2, y)) / N
        w1 = w1 - lr * sum(df1(w0, w1, w2, xi1, xi2, yi) for xi1, xi2, yi in zip(x1, x2, y)) / N
        w2 = w2 - lr * sum(df2(w0, w1, w2, xi1, xi2, yi) for xi1, xi2, yi in zip(x1, x2, y)) / N

        loss = j(w0, w1, w2, x1, x2, y)
        l.append(loss)

        print(f"w0: {round(w0, 2)}, w1: {round(w1, 2)}, w2: {round(w2, 2)} loss: {round(loss, 2)}")
    return w0, w1, w2

w0, w1, w2 = grad_desc(w0, w1, w2, x1, x2, y, 0.01, 500000)

