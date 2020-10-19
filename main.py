import logging
import sys
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def means(x, y):
    means_vector = []
    for c in range(3):
        x1_mean = sum(x[y == c, 0]) / len(x[y == c, 0])
        x2_mean = sum(x[y == c, 1]) / len(x[y == c, 1])
        x3_mean = sum(x[y == c, 2]) / len(x[y == c, 2])
        x4_mean = sum(x[y == c, 3]) / len(x[y == c, 3])
        means_vector.append([x1_mean, x2_mean, x3_mean, x4_mean])

    return means_vector


def matrix_empty(rows, cols):
    """
    Cria uma matriz vazia
        :param rows: número de linhas da matriz
        :param cols: número de colunas da matriz

        :return: matriz preenchida com 0.0
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def matrix_subtraction(A, B):
    """
    Subtracts matrix B from matrix A and returns difference
        :param A: The first matrix
        :param B: The second matrix

        :return: Matrix difference
    """
    # Section 1: Ensure dimensions are valid for matrix subtraction
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Create a new matrix for the matrix difference
    C = matrix_empty(rowsA, colsB)

    # Section 3: Perform element by element subtraction
    for i in range(rowsA):
        for j in range(colsB):
            C[i][j] = A[i][j] - B[i][j]

    return C


def matrix_transpose(m):
    """
    Retorna a matriz transposta
        :param m: matriz a ser transposta

        :return: resultado da matriz de entrada transposta
    """
    if not isinstance(m[0], list):
        m = [m]

    rows = len(m)
    cols = len(m[0])

    mt = matrix_empty(cols, rows)

    for i in range(rows):
        for j in range(cols):
            mt[j][i] = m[i][j]

    return mt


def matrix_addition(A, B):
    """
    Adds two matrices and returns the sum
        :param A: The first matrix
        :param B: The second matrix

        :return: Matrix sum
    """
    # Section 1: Ensure dimensions are valid for matrix addition
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Create a new matrix for the matrix sum
    C = matrix_empty(rowsA, colsB)

    # Section 3: Perform element by element sum
    for i in range(rowsA):
        for j in range(colsB):
            C[i][j] = A[i][j] + B[i][j]

    return C


def vector_to_matrix(vector):
    matriz = []
    for item in vector:
        matriz.append([item])

    return matriz


def matrix_multiply(a, b):
    """
    Retorna o produto da multiplicação da matriz a com b
        :param a: primeira matriz
        :param b: segunda matriz

        :return: matriz resultante
    """
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])
    if cols_a != rows_b:
        raise ArithmeticError('O número de colunas da matriz a deve ser igual ao número de linhas da matriz b.')

    result_matrix = matrix_empty(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for ii in range(cols_a):
                total += a[i][ii] * b[ii][j]
            result_matrix[i][j] = total

    return result_matrix


def scatter_within(x, y, mean):
    sw = matrix_empty(4, 4)
    for cl, mv in zip(range(3), mean):
        class_sw = matrix_empty(4, 4)
        mv = vector_to_matrix(mv)
        for row in x[y == cl]:
            row = list(row)
            row = vector_to_matrix(row)
            row_minus_mv = matrix_subtraction(row, mv)
            row_minus_mv_t = matrix_transpose(row_minus_mv)
            multiply = matrix_multiply(row_minus_mv, row_minus_mv_t)
            class_sw = matrix_addition(class_sw, multiply)

        sw = matrix_addition(sw, class_sw)

    return sw


def plot_data():
    logging.debug("*******PLOT*******")
    plt.scatter(iris_data[:, 0], iris_data[:, 1], iris_data[:, 2], iris_data[:, 3])
    plt.show()


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
iris_dataset = datasets.load_iris()
iris_data = iris_dataset.data
iris_target = iris_dataset.target

means_per_class = means(iris_data, iris_target)
s_w = scatter_within(iris_data, iris_target, means_per_class)

plot_data()
