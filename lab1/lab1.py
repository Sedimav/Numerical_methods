import argparse
import cmath
import json
import math
from pathlib import Path

from matrix_utils import (
    copy_matrix,
    format_number,
    identity,
    matmul,
    matrix_offdiag_norm,
    matvec,
    print_matrix,
    print_vector,
    transpose,
    validate_square_matrix,
    vector_norm_inf,
)


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


def determinant_from_lu(u, swaps):
    det = -1.0 if swaps % 2 else 1.0
    for i in range(len(u)):
        det *= u[i][i]
    return det


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


def qr_decomposition(a):
    validate_square_matrix(a, "A")
    n = len(a)
    q = identity(n)
    r = copy_matrix(a)

    for k in range(n - 1):
        x = [r[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(abs(value)**2 for value in x))
        if norm_x == 0:
            continue

        sign = x[0]/abs(x[0]) if x[0] != 0 else 1.0
        v = x[:]
        v[0] += sign * norm_x
        norm_v = math.sqrt(sum(abs(value)**2 for value in v))
        if norm_v == 0:
            continue
        v = [value / norm_v for value in v]

        for j in range(k, n):
            projection = sum(v[i].conjugate() * r[k + i][j] for i in range(len(v)))
            for i in range(len(v)):
                r[k + i][j] -= 2 * v[i] * projection

        for i in range(n):
            projection = sum(q[i][k + j] * v[j] for j in range(len(v)))
            for j in range(len(v)):
                q[i][k + j] -= 2 * projection * v[j].conjugate()
        for i in range(k+1,n):
            r[i][k] = 0.0

    return q, r


def conjugate_transpose(a):
    return [[value.conjugate() for value in row] for row in transpose(a)]


def hessenberg(a):
    """Unitary similarity: zero the entries below the first subdiagonal."""
    work = copy_matrix(a)
    n = len(a)
    for k in range(n-2):
        v = [work[i][k] for i in range(k+1,n)]
        length = math.sqrt(sum(abs(value)**2 for value in v))
        if length == 0:
            continue
        v[0] += (v[0]/abs(v[0]) if v[0] != 0 else 1.0)*length
        length = math.sqrt(sum(abs(value)**2 for value in v))
        v = [value/length for value in v]
        rotation = identity(n)
        for i in range(len(v)):
            for j in range(len(v)):
                rotation[k+1+i][k+1+j] -= 2*v[i]*v[j].conjugate()
        work = matmul(rotation,matmul(work,rotation))
        for i in range(k+2,n):
            work[i][k] = 0.0
    return work


def roots_2x2(a, b, c, d):
    middle = (a+d)/2
    difference = cmath.sqrt(((a-d)/2)**2+b*c)
    return middle+difference, middle-difference


def qr_algorithm_eigenvalues(a, epsilon, max_iterations=10000):
    validate_square_matrix(a, "A")
    validate_precision(epsilon)
    work = hessenberg(a)
    active, iterations, stalled = len(a), 0, 0
    tolerance = min(epsilon/(10*len(a)),1e-12)
    while active > 2:
        if abs(work[active-1][active-2]) <= tolerance:
            work[active-1][active-2] = 0.0
            active -= 1
            stalled = 0
            continue
        if iterations >= max_iterations:
            raise ValueError("QR-алгоритм не сошелся за заданное число итераций")
        roots = roots_2x2(work[active-2][active-2],work[active-2][active-1],
                          work[active-1][active-2],work[active-1][active-1])
        shift = min(roots,key=lambda value:abs(value-work[active-1][active-1]))
        if stalled and stalled % 30 == 0:
            shift = work[active-1][active-1]+abs(work[active-1][active-2])*(.75+.25j)
        block = [[work[i][j]-(shift if i == j else 0) for j in range(active)] for i in range(active)]
        q,_ = qr_decomposition(block)
        embedded = identity(len(a))
        for i in range(active):
            for j in range(active):
                embedded[i][j] = q[i][j]
        work = matmul(conjugate_transpose(embedded),matmul(work,embedded))
        for i in range(len(a)):
            for j in range(i-1):
                work[i][j] = 0.0
        iterations += 1
        stalled += 1
    return extract_eigenvalues_from_quasi_triangular(work,0.0),work,iterations


def extract_eigenvalues_from_quasi_triangular(a, epsilon):
    eigenvalues = []
    i = 0
    n = len(a)
    while i < n:
        if i < n - 1 and abs(a[i + 1][i]) > epsilon:
            b11 = a[i][i]
            b12 = a[i][i + 1]
            b21 = a[i + 1][i]
            b22 = a[i + 1][i + 1]
            eigenvalues.extend(roots_2x2(b11,b12,b21,b22))
            i += 2
        else:
            eigenvalues.append(complex(a[i][i]))
            i += 1
    return eigenvalues


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


def run_qr(data):
    print("\n=== 1.5. QR-разложение и QR-алгоритм ===")
    a = [[float(value) for value in row] for row in data["A"]]
    epsilon = float(data["epsilon"])
    digits = output_digits(epsilon)
    q, r = qr_decomposition(a)
    eigenvalues, quasi_triangular, iterations = qr_algorithm_eigenvalues(a, epsilon)

    print_matrix("Исходная матрица A:", a)
    print_matrix("Матрица Q:", q,digits)
    print_matrix("Матрица R:", r,digits)
    print_matrix("Q * R:", matmul(q, r),digits)
    print(f"Заданная точность epsilon: {epsilon}")
    print(f"Количество QR-итераций: {iterations}")
    print_matrix("Почти треугольная форма после QR-итераций:", quasi_triangular,digits)
    print_vector("Найденные собственные значения:", eigenvalues,digits)

    complex_example = [[float(value) for value in row] for row in data["complex_example"]]
    complex_eigenvalues, _, complex_iterations = qr_algorithm_eigenvalues(
        complex_example,
        epsilon,
    )
    print_matrix("Собственный пример матрицы с комплексной парой:", complex_example)
    print(f"Количество QR-итераций для примера: {complex_iterations}")
    print_vector("Комплексно-сопряженные собственные значения:", complex_eigenvalues,digits)


def main():
    base = Path(__file__).resolve().parent
    default_input = base / "data/lab1.json"
    if not default_input.exists():
        default_input = base.parent / "data/lab1.json"
    parser = argparse.ArgumentParser(description="Лабораторная работа № 1")
    parser.add_argument(
        "input",
        nargs="?",
        default=default_input,
        help="JSON-файл с входными данными",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if "variant" in data:
        print(f"Лабораторная работа 1. Вариант {data['variant']}")

    run_lu(data["lu"])
    run_thomas(data["thomas"])
    run_iterations(data["iterations"])
    run_jacobi(data["jacobi"])
    run_qr(data["qr"])


if __name__ == "__main__":
    main()
