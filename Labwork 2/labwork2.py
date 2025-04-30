import matplotlib.pyplot as plt

file_path = "lr.csv"

w0 = 1
w1 = 1
x = []
y = []
l = []

with open(file_path, "r") as file:
    for line in file:
        x.append(float(line.strip().split(",")[0]))
        y.append(float(line.strip().split(",")[1]))


def f(w0, w1, xi, yi):
    return (1 / 2) * (w1 * xi + w0 - yi) ** 2


def df0(w0, w1, xi, yi):
    return w1 * xi + w0 - yi


def df1(w0, w1, xi, yi):
    return xi * (w1 * xi + w0 - yi)


def j(w0, w1, xi, yi):
    N = len(x)
    return (1 / N) * sum(f(w0, w1, xi, yi) for xi, yi in zip(x, y))


def grad_desc(w0, w1, xi, yi, lr, times):
    for i in range(times):
        N = len(x)
        w0 = w0 - lr * sum(df0(w0, w1, xi, yi) for xi, yi in zip(x, y)) / N
        w1 = w1 - lr * sum(df1(w0, w1, xi, yi) for xi, yi in zip(x, y)) / N
        loss = j(w0, w1, xi, yi)
        l.append(loss)

        print(f"w0: {round(w0, 2)}, w1: {round(w1, 2)}, loss: {round(loss, 2)}")
    return w0, w1


w0, w1 = grad_desc(w0, w1, x, y, 0.001, 200)

# Plotting the data points
plt.scatter(x, y, color='blue', label='Data Points')
# Plotting the linear regression line
print(f"w0: {w0}, w1: {w1}")
plt.plot(x, [w1 * xi + w0 for xi in x], color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

plt.plot(range(len(l)), l)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.show()