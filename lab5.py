import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

DEBUG = True

error = 0.01
x, y = sp.symbols('x y')
expression = sp.cos(sp.cos(x / 2)) * sp.ln(x - 1)
function = sp.lambdify(x, expression, 'numpy')


range = np.linspace(1.1, 4)
plt.plot(range, function(range))
plt.plot(range, np.zeros_like(range))
plt.show()

a = 1.5
b = 3

def check_range(function, a , b):
    return function(a) * function(b) < 0


print(f'Is range correct? : {check_range(function, a, b)}')


def chord_method(function, a, b, expression, error):
    temp_a = a
    temp_b = b
    x = sp.symbols('x')

    if function(a) * sp.diff(expression, x, x).subs(x, a) > 0:
        temp_a, temp_b = temp_b, temp_a
        if DEBUG:
            print('swap a & b')

    ans = temp_a + error
    temp_x = temp_a  
    ans = ans - (function(ans) * (ans - temp_a)) / (function(ans) - function(temp_a)) 
   
    while abs(ans - temp_x) > error:
        temp_x = ans
        ans = ans - (function(ans) * (ans - temp_a)) / (function(ans) - function(temp_a)) 
        if DEBUG:
            print('chord method iteration')

    return ans


print(f'chord method answer: {chord_method(function, a, b, expression, error)}')


def targent_method(function, a, b, expression, error):
    temp_a = a
    temp_b = b
    x = sp.symbols('x')

    if function(a) * sp.diff(expression, x, x).subs(x, a) > 0:
        temp_a, temp_b = temp_b, temp_a
        if DEBUG:
            print('swap a & b')

    function_diff = sp.lambdify(x, sp.diff(expression, x), 'numpy')

    temp_x = temp_a  
    ans = temp_x - function(temp_x) / function_diff(temp_x)
       
    while abs(ans - temp_x) > error:
        temp_x = ans
        ans = temp_x - function(temp_x) / function_diff(temp_x)
        if DEBUG:
            print('targent method iteration')

    return ans


print(f'targent method answer: {targent_method(function, a, b, expression, error)}')


#expression_1 = sp.tan(x*y + 1) - x ** 2
#expression_2 = 3 * x**2 + 2 * y ** 2 - 1

#function_1 = sp.lambdify(x, expression_1, 'numpy')
#function_2 = sp.lambdify(x, expression_2, 'numpy')

#a_x = -1
#b_x = 1
#a_y = -1
#b_y = 1

#plot_1 = sp.plotting.plot_implicit(sp.Eq(expression_1, 0), (x, a_x, b_x), (y, a_y, b_y))
#plot_2 = sp.plotting.plot_implicit(sp.Eq(expression_2, 0), (x, a_x, b_x), (y, a_y, b_y))
#plot_1.extend(plot_2)

#plot_1.show()


expression_1 = sp.sin(sp.sin(2 * x + y)) - 1.6 * x
expression_2 = x ** 2 + 2 * y ** 2 - 1

function_1 = sp.lambdify((x, y), expression_1, 'numpy')
function_2 = sp.lambdify((x, y), expression_2, 'numpy')

a_x = -1
b_x = 1
a_y = -1
b_y = 1

plot_1 = sp.plotting.plot_implicit(sp.Eq(expression_1, 0), (x, a_x, b_x), (y, a_y, b_y))
plot_2 = sp.plotting.plot_implicit(sp.Eq(expression_2, 0), (x, a_x, b_x), (y, a_y, b_y))
plot_1.extend(plot_2)

plot_1.show()

x_0 = 0.3
y_0 = 0.3


def jacobian(expression_1, expression_2, x_0, y_0):
    x, y = sp.symbols('x y')
    j = np.empty((2, 2), dtype=float)
    j[0][0] = sp.diff(expression_1, x).subs([(x, x_0), (y, y_0)]).evalf()
    j[0][1] = sp.diff(expression_1, y).subs([(x, x_0), (y, y_0)]).evalf()
    j[1][0] = sp.diff(expression_2, x).subs([(x, x_0), (y, y_0)]).evalf()
    j[1][1] = sp.diff(expression_2, y).subs([(x, x_0), (y, y_0)]).evalf()

    return j


modified_expression_1 = sp.sin(sp.sin(2 * x + y)) / 1.6
modified_expression_2 = sp.sqrt((1 - x ** 2) / 2)


def simple_iterations_method(expression_1, expression_2, x_0, y_0, error):
    x, y = sp.symbols('x y')

    j = jacobian(expression_1, expression_2, x_0, y_0)
    if np.linalg.norm(j) >= 1:
        raise ValueError("Jacobian norm is not less than 1")

    function_x = sp.lambdify((x, y), expression_1, 'numpy')
    function_y = sp.lambdify((x, y), expression_2, 'numpy')

    while True:
        temp_x = function_x(x_0, y_0)
        temp_y = function_y(x_0, y_0)

        if DEBUG:
            print('simple iterations method iteration')

        if max([abs(x_0 - temp_x), abs(y_0 - temp_y)]) <= error:
            break
        x_0, y_0 = temp_x, temp_y

    return temp_x, temp_y

ans_x, ans_y = simple_iterations_method(modified_expression_1, modified_expression_2, x_0, y_0, error)
print(f'simple iterations method answer: x = {ans_x}, y = {ans_y}')


def newton_method(expression_1, expression_2, function_1, function_2, x_0, y_0, error):
    prev_x = x_0
    prev_y = y_0

    while True:
        if DEBUG:
            print('newton method iteration')
        
        inv_j = np.linalg.inv(jacobian(expression_1, expression_2, prev_x, prev_y))

        if not np.linalg.det(inv_j):
            raise ValueError('Jacobian equals to zero')

        next_x = prev_x - inv_j[0].dot(np.array([function_1(prev_x, prev_y), function_2(prev_x, prev_y)]))
        next_y = prev_y - inv_j[1].dot(np.array([function_1(prev_x, prev_y), function_2(prev_x, prev_y)]))

        if max(abs(next_x - prev_x), abs(next_y - prev_y)) <= error:
            break

        prev_x = next_x
        prev_y = next_y

    return next_x, next_y


ans_x, ans_y = newton_method(expression_1, expression_2, function_1, function_2, x_0, y_0, error)
print(f'newton method answer: x = {ans_x}, y = {ans_y}')


def modified_newton_method(expression_1, expression_2, function_1, function_2, x_0, y_0, error):
    prev_x = x_0
    prev_y = y_0

    inv_j = np.linalg.inv(jacobian(expression_1, expression_2, x_0, y_0))

    if not np.linalg.det(inv_j):
            raise ValueError('Jacobian equals to zero')

    while True:
        if DEBUG:
            print('newton method iteration')
      
        next_x = prev_x - inv_j[0].dot(np.array([function_1(prev_x, prev_y), function_2(prev_x, prev_y)]))
        next_y = prev_y - inv_j[1].dot(np.array([function_1(prev_x, prev_y), function_2(prev_x, prev_y)]))

        if max(abs(next_x - prev_x), abs(next_y - prev_y)) <= error:
            break

        prev_x = next_x
        prev_y = next_y

    return next_x, next_y


ans_x, ans_y = modified_newton_method(expression_1, expression_2, function_1, function_2, x_0, y_0, error)
print(f'modified newton method answer: x = {ans_x}, y = {ans_y}')