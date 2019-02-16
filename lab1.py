import numpy as np

from utility_functions import * 


def main():
    A = np.array([[3.241, 0.197, 0.643, 0.236],
                  [0.257, 3.853, 0.342, 0.427],
                  [0.324, 0.317, 2.793, 0.238],
                  [0.438, 0.326, 0.483, 4.229]], dtype='float')
    b = np.array([0.454, 0.371, 0.465, 0.822], dtype='float')
    gauss(A, b)
    print(f'inv A : {np.linalg.inv(A)}')
    print(f'absolute error = {absolute_error(A)}')
    print(f'relative error <= {relative_error(A, b)}')


main()
