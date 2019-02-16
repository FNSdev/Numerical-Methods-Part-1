import numpy as np
import math


def diagonalize(A : np.array, b : np.array = None):
    rows_count = A.shape[0]
    j = 0
    if b is not None:
        for n in range(1, rows_count):
            index = n - 1
            for i in range(n, rows_count):
                k = A[i][j] / A[index][j]
                A[i] = A[i] - k * A[index]
                b[i] = b[i] - k * b[index]
            j += 1
    else:
        for n in range(1, rows_count):
            index = n - 1
            for i in range(n, rows_count):
                k = A[i][j] / A[index][j]
                A[i] = A[i] - k * A[index]
            j += 1 
    

def gauss(A : np.array, b: np.array):
    if np.linalg.det(A) == 0:
        raise ValueError('There is no solution') 
    temp_A = np.array(A, copy=True)
    temp_b = np.array(b, copy=True)
    diagonalize(temp_A, temp_b)
    count = temp_A.shape[0]
    ans = np.ones(count)
    for i in range(count - 1, -1, -1):
        temp_A[i] = temp_A[i] * ans
        for j in range(count):
            if i == j:
                continue
            temp_b[i] -= temp_A[i][j]    
        ans[i] = temp_b[i] / temp_A[i][i]
    print(f'answer : {ans}')


def matrix_norm_sum_by_i(arr : np.array):
    max_sum = 0
    for i in range(arr.shape[0]):
        temp_sum = 0
        for j in range(arr.shape[1]):
            temp_sum += abs(arr[i][j])
        if temp_sum > max_sum:
            max_sum = temp_sum
    return max_sum 


def matrix_norm_sum_by_j(arr : np.array):
    max_sum = 0
    for i in range(arr.shape[1]):
        temp_sum = 0
        for j in range(arr.shape[0]):
            temp_sum += abs(arr[i][j])
        if temp_sum > max_sum:
            max_sum = temp_sum
    return max_sum 


def matrix_norm_by_sqr(arr : np.array):
    sum = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sum += arr[i][j]**2
    return sum


def vector_norm(vector : np.array):
    if len(vector.shape) > 1:
        raise ValueError(f'{vector} is not a vector!')
    return max([abs(i) for i in vector])


def det(arr : np.array):
    temp_arr = np.array(arr, copy=True)
    rows = temp_arr.shape[0]
    diagonalize(temp_arr)
    det = 1
    for i in range(rows):
        det *= arr[i][i]
    return det


def absolute_error(arr : np.array, abs_b = 0.001):
    return abs_b * matrix_norm_sum_by_i(np.linalg.inv(arr))


def relative_error(A : np.array, b : np.array, abs_b = 0.001):
    rel_b = abs_b / vector_norm(b)
    return rel_b * matrix_norm_sum_by_i(A) * matrix_norm_sum_by_i(np.linalg.inv(A))


def is_simply_iterable(B : np.array):
    seidel_iterable, norm = is_seidel_iterable(B)
    if seidel_iterable is True:
        return True, norm
    elif matrix_norm_by_sqr(B) < 1:
        return True, matrix_norm_by_sqr
    return False, None


def is_seidel_iterable(B : np.array):
    if matrix_norm_sum_by_i(B) < 1:
        return True, matrix_norm_sum_by_i
    elif matrix_norm_sum_by_j(B) < 1:
        return True, matrix_norm_sum_by_j
    return False, None


def simple_iter_data_transform(A : np.array, b : np.array):
    temp_A = np.array(A, copy=True)
    temp_b = np.array(b, copy=True)

    rows = len(temp_A)

    for i in range(rows):
        temp_b[i] = temp_b[i] / temp_A[i][i]
        temp_A[i] = temp_A[i] / temp_A[i][i]
   
    E = np.diag(np.ones(rows))
    B = E - temp_A

    return (B, temp_b)

#TODO check value of k
def simple_iter(A : np.array, b : np.array, error = 0.01):
    B, c = simple_iter_data_transform(A, b)

    simply_iterable, matrix_norm = is_simply_iterable(B)
    if simply_iterable is False:
        raise ValueError('Сonvergence condition is not satisfied')

    x0 = np.array(c, copy=True)
    x1 = np.dot(B, x0) + c

    B_norm = matrix_norm(B)
    k = math.ceil(math.log(error * (1 - B_norm) / vector_norm(x1 - x0)) / math.log(B_norm))

    x = x1

    for i in range(k - 1):
        x = np.dot(B, x) + c

    return x


def seidel(A : np.array, b : np.array, error = 0.01):
    B, c = simple_iter_data_transform(A, b)
    seidel_iterable, matrix_norm = is_seidel_iterable(B)

    if seidel_iterable is False:
        raise ValueError('Сonvergence condition is not satisfied')

    n = len(B)
    x_k = np.copy(c)

    convergense = False
    while not convergense:
        x_k_new = np.copy(x_k)
        for i in range(0, n):
            s1 = sum(B[i][j] * x_k_new[j] for j in range(i))
            s2 = sum(B[i][j] * x_k[j] for j in range(i, n))
            x_k_new[i] = s1 + s2 + c[i]
        convergense = math.sqrt(sum((x_k_new[i] - x_k[i]) ** 2 for i in range(n))) <= error
        x_k = x_k_new

    return x_k


def is_matrix_symmetrical(A : np.array, tol = 1e-08):
    return not False in (np.abs(A - A.T) < tol)


def make_system_symmetrical(A : np.array, b : np.array):
    new_A = np.dot(A.T, A)
    new_b = np.dot(A.T, b)

    return new_A, new_b 


def sqrt_method(A : np.array, b : np.array):
    temp_A = np.copy(A)
    temp_b = np.copy(b)

    if is_matrix_symmetrical(A) is False:
        temp_A, temp_b = make_system_symmetrical(temp_A, temp_b)

    n = len(A)
    S = np.zeros((n ,n))

    S[0][0] = math.sqrt(temp_A[0][0])

    for j in range(1, n):
        S[0][j] = temp_A[0][j] / S[0][0]

    for i in range(1 , n):
        s = sum(S[p][i] ** 2 for p in range(i))
        S[i][i] = math.sqrt(temp_A[i][i] - s)
        for j in range(i + 1, n):
            s = sum(S[p][i] * S[p][j] for p in range(i))
            S[i][j] = (temp_A[i][j] - s) / S[i][i]

    S_T = S.T

    y = np.zeros(n)
    y[0] = temp_b[0] / S_T[0][0]
    for k in range(1, n):
        s = sum(S_T[k][s] * y[s] for s in range(k))
        y[k] = (temp_b[k] - s) / S_T[k][k]

    x = np.zeros(n)
    x[n - 1] = y[n - 1] / S[n - 1][n - 1]
    for k in range(n - 2, -1, -1):
        s = sum(S[k][p] * x[p] for p in range(k + 1, n))
        x[k] = (y[k] - s) / S[k][k]

    return x 


def sqrt_inv_matrix(A : np.array):
    temp_A = np.copy(A)

    n = len(A)
    S = np.zeros((n ,n))

    S[0][0] = math.sqrt(temp_A[0][0])

    for j in range(1, n):
        S[0][j] = temp_A[0][j] / S[0][0]

    for i in range(1 , n):
        s = sum(S[p][i] ** 2 for p in range(i))
        S[i][i] = math.sqrt(temp_A[i][i] - s)
        for j in range(i + 1, n):
            s = sum(S[p][i] * S[p][j] for p in range(i))
            S[i][j] = (temp_A[i][j] - s) / S[i][i]

    S_T = S.T

    E = np.diag(np.ones(n))
    inv_A = np.zeros((n, n))

    for i in range(n):
        y = np.zeros(n)
        y[0] = E[i][0] / S_T[0][0]
        for k in range(1, n):
            s = sum(S_T[k][s] * y[s] for s in range(k))
            y[k] = (E[i][k] - s) / S_T[k][k]

        x = np.zeros(n)
        x[n - 1] = y[n - 1] / S[n - 1][n - 1]
        for k in range(n - 2, -1, -1):
            s = sum(S[k][p] * x[p] for p in range(k + 1, n))
            x[k] = (y[k] - s) / S[k][k]  

        inv_A[i] = x
    
    return inv_A

