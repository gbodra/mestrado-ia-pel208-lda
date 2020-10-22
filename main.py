import logging
import sys
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def means(x, y):
    means_vector = []
    for c in range(3):
        x1_y_mean = sum(x[y == c, 0]) / len(x[y == c, 0])
        x2_y_mean = sum(x[y == c, 1]) / len(x[y == c, 1])
        x3_y_mean = sum(x[y == c, 2]) / len(x[y == c, 2])
        x4_y_mean = sum(x[y == c, 3]) / len(x[y == c, 3])
        means_vector.append([x1_y_mean, x2_y_mean, x3_y_mean, x4_y_mean])

    x1_mean = sum(x[:, 0]) / len(x[:, 0])
    x2_mean = sum(x[:, 1]) / len(x[:, 1])
    x3_mean = sum(x[:, 2]) / len(x[:, 2])
    x4_mean = sum(x[:, 3]) / len(x[:, 3])
    overall_mean = [x1_mean, x2_mean, x3_mean, x4_mean]

    return means_vector, overall_mean


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


def matrix_minor(m, i, j):
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def matrix_determinant(m):
    # caso especial para matriz 2x2
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * matrix_determinant(matrix_minor(m, 0, c))
    return determinant


def matrix_inverse(m):
    determinant = matrix_determinant(m)

    # caso especial para matriz 2x2
    if len(m) == 2:
        return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                [-1 * m[1][0] / determinant, m[0][0] / determinant]]

    # calcular matriz de cofatores
    cofactors = []

    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = matrix_minor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * matrix_determinant(minor))
        cofactors.append(cofactorRow)

    cofactors = matrix_transpose(cofactors)

    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors


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


def scatter_between(x, y, means_vector, overall_mean):
    sb = matrix_empty(4, 4)
    overall_mean = matrix_transpose(overall_mean)

    for i, mean_vec in enumerate(means_vector):
        n = len(x[y == i, :])
        mean_vec = vector_to_matrix(mean_vec)
        mean_vec_minus_overall_mean = matrix_subtraction(mean_vec, overall_mean)
        mean_vec_minus_overall_mean_t = matrix_transpose(mean_vec_minus_overall_mean)
        multiply = matrix_multiply(mean_vec_minus_overall_mean, mean_vec_minus_overall_mean_t)
        sb_n = [[item * n for item in item_list] for item_list in multiply]
        sb = matrix_addition(sb, sb_n)

    return sb


def sw_sb(sw, sb):
    sw_i = matrix_inverse(sw)
    sw_i_sb = matrix_multiply(sw_i, sb)
    eig_vals, eig_vecs = np.linalg.eig(sw_i_sb)

    return eig_vals, eig_vecs


def sort_eig(eig_vals, eig_vecs):
    # STEP 4 - Selecting linear discriminants for the new feature subspace

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues

    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])

    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i, j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i + 1, (j[0] / eigv_sum).real))

    # STEP 4.2. Choosing k eigenvectors with the largest eigenvalues
    W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
    print('Matrix W:\n', W.real)

    # STEP 5 Transforming the samples onto the new subspace
    X_lda = iris_data.dot(W)
    assert X_lda.shape == (150, 2), "The matrix is not 150x2 dimensional."

    return X_lda


def plot_step_lda():
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(0, 3), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X_lda[:, 0].real[iris_target == label],
                    y=X_lda[:, 1].real[iris_target == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()


def plot_data():
    logging.debug("*******PLOT*******")
    plt.scatter(iris_data[:, 0], iris_data[:, 1], iris_data[:, 2], iris_data[:, 3])
    plt.show()


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
iris_dataset = datasets.load_iris()
iris_data = iris_dataset.data
iris_target = iris_dataset.target
label_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

means_per_class, grand_mean = means(iris_data, iris_target)
s_w = scatter_within(iris_data, iris_target, means_per_class)
s_b = scatter_between(iris_data, iris_target, means_per_class, grand_mean)
v, w = sw_sb(s_w, s_b)
X_lda = sort_eig(v, w)

plot_step_lda()

# plot_data()
