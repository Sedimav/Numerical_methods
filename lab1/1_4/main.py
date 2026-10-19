import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.matrix_utils import (copy_matrix, identity, matmul, matrix_offdiag_norm, print_matrix, print_vector, transpose, validate_square_matrix)
from common.runner import run_program


def validate_precision(epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Точность должна быть конечной и положительной")


def output_digits(epsilon):
    validate_precision(epsilon)
    return max(6,2-math.floor(math.log10(epsilon)))


def jacobi_rotation_method(a, epsilon, max_iterations=10000, history=None):
    validate_square_matrix(a, "A")
    validate_precision(epsilon)
    n = len(a)
    if any(not math.isclose(a[i][j],a[j][i],rel_tol=1e-12,abs_tol=1e-12)
           for i in range(n) for j in range(i)):
        raise ValueError("Метод вращений требует симметричную матрицу")
    work = copy_matrix(a)
    vectors = identity(n)

    for iteration in range(max_iterations + 1):
        error = matrix_offdiag_norm(work)
        if history is not None:
            history.append((iteration,error))
        if error <= epsilon:
            eigenvalues = [work[i][i] for i in range(n)]
            return eigenvalues, vectors, iteration
        if iteration == max_iterations:
            break

        p, q = 0, 1
        max_value = abs(work[p][q])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(work[i][j]) > max_value:
                    max_value = abs(work[i][j])
                    p, q = i, j

        if work[p][p] == work[q][q]:
            angle = math.pi / 4
        else:
            angle = 0.5 * math.atan2(2 * work[p][q], work[p][p] - work[q][q])

        c = math.cos(angle)
        s = math.sin(angle)
        rotation = identity(n)
        rotation[p][p] = c
        rotation[q][q] = c
        rotation[p][q] = -s
        rotation[q][p] = s

        work = matmul(transpose(rotation), matmul(work, rotation))
        vectors = matmul(vectors, rotation)

    raise ValueError("Метод вращений Якоби не сошелся за заданное число итераций")


def run_jacobi(data):
    print("\n=== 1.4. Метод вращений Якоби ===")
    a = [[float(value) for value in row] for row in data["A"]]
    epsilon = float(data["epsilon"])
    digits = output_digits(epsilon)
    history = []
    eigenvalues, vectors, iterations = jacobi_rotation_method(a, epsilon,history=history)
    lambda_matrix = [[0.0 for _ in eigenvalues] for _ in eigenvalues]
    for i, value in enumerate(eigenvalues):
        lambda_matrix[i][i] = value

    print(f"Заданная точность epsilon: {epsilon}")
    print(f"Количество вращений: {iterations}")
    print("Вращение | норма внедиагональных элементов (оценка погрешности)")
    for count,error in history:
        print(f"{count:8d} | {error:.6e}")
    print_vector("Собственные значения:", eigenvalues,digits)
    for i, vector in enumerate(transpose(vectors), start=1):
        print_vector(f"Собственный вектор h{i}:", vector,digits)
    print_matrix("Матрица собственных векторов V:", vectors,digits)
    print_matrix("Проверка A * V:", matmul(a, vectors),digits)
    print_matrix("Проверка V * Lambda:", matmul(vectors, lambda_matrix),digits)


def run(data, out):
    print(f"Вариант {data['variant']}")
    run_jacobi(data)


if __name__ == "__main__":
    run_program(run, 'ЛР 1.4', __file__)
