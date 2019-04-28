import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import copy

x_values = (0.218, 0.562, 1.492, 2.119, 2.948)
y_values = (0.511, 0.982, 2.411, 3.115, 4.561)


def lagrange(x_values, y_values):
    x = sp.symbols('x')
    p = 1
    L = 0
    for x_i in x_values:
        p *= x - x_i 
    for x_i, y_i in zip(x_values, y_values):
        L += y_i * (p / (x - x_i)) / (p / (x - x_i)).subs(x, x_i)

    return sp.lambdify(x, L, 'numpy')

lagrange_function = lagrange(x_values, y_values)
for x in x_values:
    print(f'x = {x}, lagrange y = {lagrange_function(x)}')

print(f'L(x1 + x2) = {lagrange_function(x_values[0] + x_values[1])}\n')

space = np.linspace(0, 3)
plt.plot(space, lagrange_function(space), label='Lagrange')
plt.scatter(x_values, y_values)
plt.show()

def finite_differences(y_values):
    cnt = len(y_values)
    d = []
    d.append(y_values)

    for i in range(cnt - 1):
        d.append(['undefined' for val in range(cnt)])
        for j in range(1, cnt  - i):
            d[i + 1][j - 1] = round(d[i][j] - d[i][j - 1], 4)
    
    return d

f_diff = finite_differences(y_values)
for elem, index in zip(f_diff, range(len(f_diff))):
    print(f'finite differences, k = {index}: {elem}')
print()

def divided_differences(x_values, y_values):
    def divided_difference(x_values, y_values):
        x = sp.symbols('x')
        p = 1
        d = 0

        for x_i in x_values:
            p *= x - x_i 
        for x_i, y_i in zip(x_values, y_values):
            d += y_i / (p / (x - x_i)).subs(x, x_i)

        return d

    cnt = len(y_values)
    d = []
    d.append(y_values)

    for i in range(1, cnt):
        d.append(['undefined' for val in range(cnt)])
        for j in range(cnt - i):
            d[i][j] = round(divided_difference(x_values[j:j+i+1], y_values[j:j+i+1]), 4)

    return d 


d_diff = divided_differences(x_values, y_values)
for elem, index in zip(d_diff, range(len(f_diff))):
    print(f'divided differences, k = {index}: {elem}')
print()

def newton_polinomial(d_diff, x_values):
    x = sp.symbols('x')
    p = 1
    n = len(x_values)
    N = 0
    for i in range(n):
        N += d_diff[i][0] * p
        p *= (x - x_values[i])
        
    return sp.lambdify(x, N, "numpy")

newton_function = newton_polinomial(d_diff, x_values)
print(f'N(x1 + x2) = {newton_function(x_values[0] + x_values[1])}\n')

space = np.linspace(0, 3)
plt.plot(space, newton_function(space), label='Newton')
plt.scatter(x_values, y_values)
plt.show()


def linear_interpolation(x_values, y_values):
    cnt = len(x_values)
    a = np.empty(cnt - 1)
    b = np.empty(cnt - 1)

    for i in range(cnt - 1):
        A = np.array([[x_values[i], 1], [x_values[i + 1], 1]])
        B = np.array(y_values[i:i + 2])

        a[i], b[i] = np.linalg.solve(A, B)

        space = np.linspace(x_values[i], x_values[i + 1])
        plt.plot(space, a[i] * space + b[i])
    
    plt.scatter(x_values, y_values)
    plt.show()

    return a, b


lin_coefficients = linear_interpolation(x_values, y_values)


def quadratic_interpolation(x_values, y_values):
    cnt = len(x_values)
    a = np.empty(cnt - 1)
    b = np.empty(cnt - 1)
    c = np.empty(cnt - 1)

    for i in range(cnt - 2):
        A = []
        for j in range(3):
            A.append([x_values[i + j] ** 2, x_values[i + j], 1])

        a[i], b[i], c[i] = np.linalg.solve(np.array(A), np.array(y_values[i:i+3]))
        

    for i in range(0, cnt - 2, 2):
        #space = np.linspace(x_values[i], x_values[i + 1] if i < cnt - 3 else x_values[i + 2])
        space = np.linspace(x_values[i], x_values[i + 2])
        plt.plot(space, a[i] * space ** 2 + b[i] * space + c[i])

    
    plt.scatter(x_values, y_values)
    plt.show()

    return a, b, c

quad_coefficients = quadratic_interpolation(x_values, y_values)


def cubic_interpolation(x_values, y_values):
    cnt  = len(x_values)
    
    h_i = np.array([x_values[i] - x_values[i - 1] for i in range (1, cnt)])
    l_i = np.array([(y_values[i] - y_values[i - 1]) / h_i[i - 1] for i in range(1, cnt)])
    delta_i = np.empty(cnt - 2, float)
    lambda_i = np.empty(cnt - 2, float)
    
    delta_i[0] =  -0.5 * h_i[1] / (h_i[0] + h_i[1])
    lambda_i[0] = 1.5 * (l_i[1] - l_i[0]) / (h_i[0] + h_i[1])
    
    for i in range(1, cnt - 2):
        delta_i[i] = - h_i[i + 1] / (2 * h_i[i] + 2 * h_i[i + 1] + h_i[i] * delta_i[i - 1])
        lambda_i[i] = (2 * l_i[i + 1] - 3 * l_i[i] - h_i[i] * lambda_i[i - 1]) / \
                      (2 * h_i[i] + 2 * h_i[i + 1] + h_i[i] * delta_i[i - 1])
        
    a = np.array(copy.copy(y_values)[1:])
    b = np.empty(cnt - 1)
    c = np.empty(cnt - 1)
    d = np.empty(cnt - 1)
    c[cnt - 2] = 0
    
    for i in range(cnt - 3, -1, -1):
        c[i] = delta_i[i] * c[i + 1] + lambda_i[i]
    for i in range(cnt - 2, -1, -1):
        b[i] = l_i[i] + 2 / 3 * c[i] * h_i[i] + 1 / 3 * h_i[i] * c[i - 1]
        d[i] = (c[i] - c[i - 1]) / (3 * h_i[i])
    
    for i in range(cnt - 1):
        space = np.linspace(x_values[i], x_values[i + 1])
        plt.plot(space, a[i] + (b[i] + (c[i] + d[i] * (space - x_values[i + 1])) * (space - x_values[i + 1])) * (space - x_values[i + 1]))
    plt.scatter(x_values, y_values)
    plt.show()

    return a, b, c, d


cubic_coefficients = cubic_interpolation(x_values, y_values)


def linear_spline_function(x_value, x_points, a, b):
    interval_index = len(x_points) - 2
    for i in range(1, len(x_points)):
        if x_value < x_points[i]:
            interval_index = i - 1
            break
            
    return a[interval_index] * x_value + b[interval_index]

def quadratic_spline_function(x_value, x_points, a, b, c):
    interval_index = len(x_points) - 3
    for i in range(1, len(x_points) - 1):
        if x_value < x_points[i]:
            interval_index = i - 1
            break
            
    return c[interval_index] + (b[interval_index] + a[interval_index] * x_value) * x_value

def cubic_spline_function(x_value, x_points, a, b, c, d):
    ii = len(x_points) - 2 # ii == interval_index
    for i in range(1, len(x_points)):
        if x_value < x_points[i]:
            ii = i - 1
            break
            
    return a[ii] + (b[ii] + (c[ii] + d[ii] * (x_value - x_points[ii + 1])) * (x_value - x_points[ii + 1])) * (x_value - x_points[ii + 1])


space = np.linspace(0, 3)
plt.plot(space, lagrange_function(space), label='Lagrange')
plt.plot(space, newton_function(space), label='Newton')
plt.plot(space, [linear_spline_function(x_value, x_values, *lin_coefficients) for x_value in space], label='linear')
plt.plot(space, [quadratic_spline_function(x_value, x_values, *quad_coefficients) for x_value in space], label='quadratic')
plt.plot(space, [cubic_spline_function(x_value, x_values, *cubic_coefficients) for x_value in space], label='cubic')
plt.scatter(x_values, y_values)
plt.legend()
plt.show()