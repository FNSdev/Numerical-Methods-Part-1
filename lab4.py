import numpy as np
from utility_functions import eigen_values_and_vectors


def main():
    A = np.array([[3.241, 0.197, 0.643, 0.236],
                    [0.257, 3.853, 0.342, 0.427],
                    [0.324, 0.317, 2.793, 0.238],
                    [0.438, 0.326, 0.483, 4.229]], dtype='float')

    A = np.dot(A.T, A)

    w, v = np.linalg.eig(A)
    values, vectors = eigen_values_and_vectors(A, debug=True)
    print(f'eigen values: {values}')
    print(f'numpy eigen values: {w}')
    print('eigen vectors:')
    print(vectors)
    print('numpy eigen vectors:')
    print(v)


if __name__ == "__main__":
    main()