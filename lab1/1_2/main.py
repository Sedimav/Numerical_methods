import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.matrix_utils import (format_number, print_vector)
from common.runner import run_program


def thomas_algorithm(lower, diagonal, upper, rhs):
    n = len(diagonal)
    if not n or len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("Размеры диагоналей трехдиагональной системы не согласованы")
    if not all(math.isfinite(value) for values in (lower,diagonal,upper,rhs) for value in values):
        raise ValueError("Данные прогонки должны быть конечными числами")

    alpha = [0.0 for _ in range(n)]
    beta = [0.0 for _ in range(n)]

    if diagonal[0] == 0:
        raise ValueError("Нулевой первый знаменатель в методе прогонки")
    alpha[0] = -upper[0] / diagonal[0] if n > 1 else 0.0
    beta[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        denominator = diagonal[i] + lower[i - 1] * alpha[i - 1]
        if denominator == 0:
            raise ValueError("Нулевой знаменатель в методе прогонки")
        alpha[i] = -upper[i] / denominator if i < n - 1 else 0.0
        beta[i] = (rhs[i] - lower[i - 1] * beta[i - 1]) / denominator

    x = [0.0 for _ in range(n)]
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return alpha, beta, x


def run_thomas(data):
    print("\n=== 1.2. Метод прогонки ===")
    lower = [float(value) for value in data["lower"]]
    diagonal = [float(value) for value in data["diagonal"]]
    upper = [float(value) for value in data["upper"]]
    rhs = [float(value) for value in data["rhs"]]
    alpha, beta, x = thomas_algorithm(lower, diagonal, upper, rhs)

    print("Прогоночные коэффициенты:")
    for i, (a_i, b_i) in enumerate(zip(alpha, beta), start=1):
        print(f"i={i}: alpha={format_number(a_i)}, beta={format_number(b_i)}")
    print_vector("Итоговое решение:", x)


def run(data, out):
    print(f"Вариант {data['variant']}")
    run_thomas(data)


if __name__ == "__main__":
    run_program(run, 'ЛР 1.2', __file__)
