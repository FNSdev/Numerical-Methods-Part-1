import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math


x, y = sp.symbols('x y')

expr = x ** 2 * sp.sin(x)
func = sp.lambdify(x, expr)
a = 0
b = 1


def get_rough_steps_count(a, b, p, error):
    h = math.pow(error, 1 / p)
    return (b - a) / h


def get_steps_count(func, a, b, method, p, error):
    n = 4
    while True:
        delta = (method(func, a, b, 2 * n) - method(func, a, b, n)) / (2 ** p - 1)
        if abs(delta) <= error:
            break
        n *= 2
    return n


def trapezoid_method(func, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(n):
        sum += (func(a + h * i) + func(a + h * (i + 1))) / 2
    return h * sum


trapezoid_method_rough_n = get_rough_steps_count(a, b, 2, 0.0001)
print(f'Trapezoid method rough n = {trapezoid_method_rough_n}')
trapezoid_method_n = get_steps_count(func, a, b, trapezoid_method, 2, 0.001)
trapezoid_int_h = trapezoid_method(func, a, b, trapezoid_method_n)
print(f'Trapezoid method: {trapezoid_int_h}, n = {trapezoid_method_n}')
trapezoid_int_2h = trapezoid_method(func, a, b, trapezoid_method_n // 2)
print(f'Trapezoid method: {trapezoid_int_2h}, n = {trapezoid_method_n // 2}')


def simpson_method(func, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(0, n - 1, 2):
        sum += func(a + h * i) + 4 * func(a + h * (i + 1)) + func(a + h * (i + 2))
    return h * sum / 3


simpson_method_rough_n = get_rough_steps_count(a, b, 4, 0.0001)
print(f'Simpson method rough n = {simpson_method_rough_n}')
simpson_method_n = get_steps_count(func, a, b, simpson_method, 4, 0.001)
simpson_int_h = simpson_method(func, a, b, simpson_method_n)
print(f'Simpson method: {simpson_int_h}, n = {simpson_method_n}')
simpson_int_2h = simpson_method(func, a, b, simpson_method_n // 2)
print(f'Simpson method: {simpson_int_2h}, n = {simpson_method_n // 2}')


def newton_leibniz_method(expr, a, b):
    F = sp.integrate(expr, x)
    return (F.subs(x, b) - F.subs(x, a)).evalf()


newton_leibniz_int = newton_leibniz_method(expr, a, b)
print(f'Newton-Leibniz method: {newton_leibniz_int}')


expr = (3 * y - 20 * x ** 2 * y ** 3 - 12 * y ** 3) / (2 * x)
func = sp.lambdify((x, y), expr)
a = 1
b = 5
y_x0 = 0.25
x0 = 1


def get_ode_steps_count(func, a, b, method, x0, y_x0, p, error):
    n = 4
    while True:
        _, y1 = method(func, a, b, x0, y_x0, 2 * n)
        _, y2 = method(func, a, b, x0, y_x0, n)
        stop = True
        for i in range(len(y2)):
            if abs(y1[2 * i] - y2[i]) / (2 ** p - 1) > error:
                stop = False
        if stop:
            break
        n *= 2
        
    return n


def runge_kutta_method(func, a, b, x0, y_x0, n):
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = y_x0

    for i in range(n):
        F1 = func(x[i], y[i])
        F2 = func(x[i] + h / 2, y[i] + h * F1 / 2)
        F3 = func(x[i] + h / 2, y[i] + h * F2 / 2)
        F4 = func(x[i] + h, y[i] + h * F3)
        y[i + 1] = y[i] + (h / 6) * (F1 + 2 * F2 + 2 * F3 + F4)
        x[i + 1] = x[i] + h

    return x, y 


runge_n = get_ode_steps_count(func, a, b, runge_kutta_method, x0, y_x0, 4, 0.0001)
runge_x, runge_y = runge_kutta_method(func, a, b, x0, y_x0, runge_n)
runge_x_2, runge_y_2 = runge_kutta_method(func, a, b, x0, y_x0, runge_n // 2)
print(f'Method Runge-Kutta, n = {runge_n}')

for i in range(len(runge_x)):
    if i % 2 == 0:
        print(f'x = {runge_x[i]}, y_h = {runge_y[i]}, y_2h = {runge_y_2[i // 2]}, delta = {abs(runge_y[i] - runge_y_2[i // 2] )}')
    else:
        print(f'x = {runge_x[i]}, y_h = {runge_y[i]}')

plt.plot(runge_x, runge_y, label='h')
plt.plot(runge_x_2, runge_y_2, label='2h')
plt.legend()
plt.show()


def adams_method(func, a, b, x0, y_x0, n):
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = y_x0
    x[1] = x[0] + h
    y[1] = y[0] + h * func(x[0], y[0])

    for i in range(1, n):
        predictor = y[i] + h / 2 * (3 * func(x[i], y[i]) - func(x[i - 1], y[i - 1]))
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h / 2 * (func(x[i], y[i]) + func(x[i + 1], predictor))

    return x, y 

adams_n = get_ode_steps_count(func, a, b, adams_method, x0, y_x0, 4, 0.0001)
adams_x, adams_y = adams_method(func, a, b, x0, y_x0, adams_n)
adams_x_2, adams_y_2 = adams_method(func, a, b, x0, y_x0, adams_n // 2)
print(f'Adams Method, n = {adams_n}')

for i in range(len(adams_x)):
    if i % 2 == 0:
        print(f'x = {adams_x[i]}, y_h = {adams_y[i]}, y_2h = {adams_y_2[i // 2]}, delta = {abs(adams_y[i] - adams_y_2[i // 2] )}')
    else:
        print(f'x = {adams_x[i]}, y_h = {adams_y[i]}')

plt.plot(adams_x, adams_y, label='h')
plt.plot(adams_x_2, adams_y_2, label='2h')
plt.legend()
plt.show()


def euler_method(func, a, b, x0, y_x0, n):
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = y_x0

    for i in range(n):
       x[i + 1] = x[i] + h
       y[i + 1] = y[i] + h * func(x[i], y[i])

    return x, y 

euler_n = get_ode_steps_count(func, a, b, euler_method, x0, y_x0, 4, 0.0001)
euler_x, euler_y = euler_method(func, a, b, x0, y_x0, euler_n)
euler_x_2, euler_y_2 = euler_method(func, a, b, x0, y_x0, euler_n // 2)
print(f'Euler Method, n = {euler_n}')

for i in range(len(euler_x)):
    if i % 2 == 0:
        print(f'x = {euler_x[i]}, y_h = {euler_y[i]}, y_2h = {euler_y_2[i // 2]}, delta = {abs(euler_y[i] - euler_y_2[i // 2] )}')
    else:
        print(f'x = {euler_x[i]}, y_h = {euler_y[i]}')

plt.plot(euler_x, euler_y, label='h')
plt.plot(euler_x_2, euler_y_2, label='2h')
plt.legend()
plt.show()


f = sp.Function('f')
solution = sp.dsolve(sp.Eq(2 * x * sp.diff(f(x), x) - 3 * f(x) + 20 * x ** 2 * f(x) ** 3 + 12 * f(x) **3), f(x))
print(f'Solution: {solution}')

C1 = sp.solve(sp.sqrt(2) / 2 * sp.sqrt(1 / (x + 4)) - 0.25, x)
print(f'C1 = {C1}')

solution = sp.lambdify(x, sp.sqrt((2 * x ** 3) / (C1[0] + 2 * x ** 5 + 2 * x ** 3)) / 2)
space = np.linspace(a, b)
plt.plot(space, solution(space))
plt.show()


plt.plot(runge_x, runge_y, label='runge')
plt.plot(adams_x, adams_y, label='adams')
plt.plot(euler_x, euler_y, label='euler')
plt.plot(space, solution(space), label='solution')
plt.legend()
plt.show()
