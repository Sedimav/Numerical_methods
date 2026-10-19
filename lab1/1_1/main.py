import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.matrix_utils import (copy_matrix, format_number, identity, matmul, matvec, print_matrix, print_vector, transpose, validate_square_matrix)
from common.runner import run_program


def validate_rhs(a, b):
    if len(b) != len(a) or not all(math.isfinite(value) for value in b):
        raise ValueError("Размер или значения вектора правой части неверны")


def lu_decomposition_with_pivoting(a):
    validate_square_matrix(a, "A")
    n = len(a)
    u = copy_matrix(a)
    l = identity(n)
    p = identity(n)
    swaps = 0

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(u[i][k]))
        if u[pivot_row][k] == 0:
            raise ValueError("Матрица вырождена: найден нулевой опорный элемент")

        if pivot_row != k:
            u[k], u[pivot_row] = u[pivot_row], u[k]
            p[k], p[pivot_row] = p[pivot_row], p[k]
            for j in range(k):
                l[k][j], l[pivot_row][j] = l[pivot_row][j], l[k][j]
            swaps += 1

        for i in range(k + 1, n):
            factor = u[i][k] / u[k][k]
            l[i][k] = factor
            for j in range(k, n):
                u[i][j] -= factor * u[k][j]

    return p, l, u, swaps


def forward_substitution(l, b):
    n = len(l)
    y = [0.0 for _ in range(n)]
    for i in range(n):
        y[i] = b[i] - sum(l[i][j] * y[j] for j in range(i))
        if l[i][i] == 0:
            raise ValueError("Нулевой элемент на диагонали L")
        y[i] /= l[i][i]
    return y


def backward_substitution(u, y):
    n = len(u)
    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        value = y[i] - sum(u[i][j] * x[j] for j in range(i + 1, n))
        if u[i][i] == 0:
            raise ValueError("Нулевой элемент на диагонали U")
        x[i] = value / u[i][i]
    return x


def solve_with_lu(p, l, u, b):
    validate_rhs(l,b)
    pb = matvec(p, b)
    y = forward_substitution(l, pb)
    return backward_substitution(u, y)


def inverse_with_lu(p, l, u):
    n = len(l)
    columns = []
    for col in range(n):
        e = [1.0 if i == col else 0.0 for i in range(n)]
        columns.append(solve_with_lu(p, l, u, e))
    return transpose(columns)


def determinant_from_lu(u, swaps):
    det = -1.0 if swaps % 2 else 1.0
    for i in range(len(u)):
        det *= u[i][i]
    return det


def run_lu(data):
    print("\n=== 1.1. LU-разложение ===")
    a = [[float(value) for value in row] for row in data["A"]]
    b = [float(value) for value in data["b"]]
    p, l, u, swaps = lu_decomposition_with_pivoting(a)
    x = solve_with_lu(p, l, u, b)
    inverse = inverse_with_lu(p, l, u)
    determinant = determinant_from_lu(u, swaps)

    print_matrix("Матрица L:", l)
    print_matrix("Матрица U:", u)
    print_matrix("L * U:", matmul(l, u))
    print_vector("Решение Ax = b:", x)
    print_matrix("Обратная матрица A^(-1):", inverse)
    print(f"Определитель det(A): {format_number(determinant)}")
    print_matrix("Проверка A * A^(-1):", matmul(a, inverse))
    print_matrix("Проверка L * U:", matmul(l, u))
    print_matrix("Проверка P * A:", matmul(p, a))


def run(data, out):
    print(f"Вариант {data['variant']}")
    run_lu(data)


if __name__ == "__main__":
    run_program(run, 'ЛР 1.1', __file__)
