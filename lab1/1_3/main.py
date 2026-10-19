import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.matrix_utils import (copy_matrix, identity, matvec, print_vector, transpose, validate_square_matrix, vector_norm_inf)
from common.runner import run_program


def validate_precision(epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Точность должна быть конечной и положительной")


def output_digits(epsilon):
    validate_precision(epsilon)
    return max(6,2-math.floor(math.log10(epsilon)))


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


def iteration_error_bound(a, b, x, inverse):
    residual = [abs(value-right) for value,right in zip(matvec(a,x),b)]
    return max(sum(abs(value)*r for value,r in zip(row,residual)) for row in inverse)


def simple_iterations(a, b, epsilon, max_iterations=10000, inverse=None):
    validate_square_matrix(a, "A")
    validate_rhs(a,b)
    validate_precision(epsilon)
    if inverse is None:
        p,l,u,_ = lu_decomposition_with_pivoting(a)
        inverse = inverse_with_lu(p,l,u)
    n = len(a)
    x_old = [0.0 for _ in range(n)]

    for iteration in range(1, max_iterations + 1):
        x_new = [0.0 for _ in range(n)]
        for i in range(n):
            if a[i][i] == 0:
                raise ValueError("Нулевой диагональный элемент в методе простых итераций")
            s = sum(a[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / a[i][i]

        if not all(math.isfinite(value) for value in x_new):
            raise ValueError("Метод простых итераций расходится")
        if iteration_error_bound(a,b,x_new,inverse) <= epsilon:
            return x_new, iteration
        x_old = x_new

    raise ValueError("Метод простых итераций не сошелся за заданное число итераций")


def seidel_method(a, b, epsilon, max_iterations=10000, inverse=None):
    validate_square_matrix(a, "A")
    validate_rhs(a,b)
    validate_precision(epsilon)
    if inverse is None:
        p,l,u,_ = lu_decomposition_with_pivoting(a)
        inverse = inverse_with_lu(p,l,u)
    n = len(a)
    x = [0.0 for _ in range(n)]

    for iteration in range(1, max_iterations + 1):
        x_old = x[:]
        for i in range(n):
            if a[i][i] == 0:
                raise ValueError("Нулевой диагональный элемент в методе Зейделя")
            left = sum(a[i][j] * x[j] for j in range(i))
            right = sum(a[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - left - right) / a[i][i]

        if not all(math.isfinite(value) for value in x):
            raise ValueError("Метод Зейделя расходится")
        if iteration_error_bound(a,b,x,inverse) <= epsilon:
            return x, iteration

    raise ValueError("Метод Зейделя не сошелся за заданное число итераций")


def run_iterations(data):
    print("\n=== 1.3. Метод простых итераций и метод Зейделя ===")
    a = [[float(value) for value in row] for row in data["A"]]
    b = [float(value) for value in data["b"]]
    epsilon = float(data["epsilon"])
    digits = output_digits(epsilon)
    p, l, u, _ = lu_decomposition_with_pivoting(a)
    reference = solve_with_lu(p, l, u, b)
    inverse = inverse_with_lu(p,l,u)
    simple_x, simple_count = simple_iterations(a, b, epsilon,inverse=inverse)
    seidel_x, seidel_count = seidel_method(a, b, epsilon,inverse=inverse)

    print(f"Заданная точность epsilon: {epsilon}")
    print_vector("Эталонное решение через LU:", reference,digits)
    print_vector("Метод простых итераций:", simple_x,digits)
    print(f"Количество итераций метода простых итераций: {simple_count}")
    print_vector("Метод Зейделя:", seidel_x,digits)
    print(f"Количество итераций метода Зейделя: {seidel_count}")
    print(f"Ошибка простых итераций относительно эталона: {vector_norm_inf([v-r for v,r in zip(simple_x,reference)]):.6e}")
    print(f"Ошибка Зейделя относительно эталона: {vector_norm_inf([v-r for v,r in zip(seidel_x,reference)]):.6e}")
    if seidel_count < simple_count:
        print("Сравнение: метод Зейделя сошелся быстрее.")
    elif seidel_count > simple_count:
        print("Сравнение: метод простых итераций сошелся быстрее.")
    else:
        print("Сравнение: оба метода сошлись за одинаковое число итераций.")


def run(data, out):
    print(f"Вариант {data['variant']}")
    run_iterations(data)


if __name__ == "__main__":
    run_program(run, 'ЛР 1.3', __file__)
