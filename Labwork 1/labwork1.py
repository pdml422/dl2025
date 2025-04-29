def f(x):
    return x * x


def f_(x):
    return 2 * x


def grad_desc(x, l, time):
    for i in range(time):
        x = x - l * f_(x)
        print(f"{round(x, 2)} - {round(f(x), 2)}")


l = 0.99
x = 10
time = 10

print("x - f(x)")
grad_desc(x, l, time)
