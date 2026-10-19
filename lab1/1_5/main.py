import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cmath
import math
from common.matrix_utils import (copy_matrix, identity, matmul, print_matrix, print_vector, transpose, validate_square_matrix)
from common.runner import run_program


def validate_precision(epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Точность должна быть конечной и положительной")


def output_digits(epsilon):
    validate_precision(epsilon)
    return max(6,2-math.floor(math.log10(epsilon)))


def qr_decomposition(a):
    validate_square_matrix(a, "A")
    n = len(a)
    q = identity(n)
    r = copy_matrix(a)

    for k in range(n - 1):
        x = [r[i][k] for i in range(k, n)]
        norm_x = math.hypot(*(abs(value) for value in x))
        if norm_x == 0:
            continue

        sign = x[0]/abs(x[0]) if x[0] != 0 else 1.0
        v = [value / norm_x for value in x]
        v[0] += sign
        norm_v = math.hypot(*(abs(value) for value in v))
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
    work = copy_matrix(a)
    n = len(a)
    for k in range(n-2):
        v = [work[i][k] for i in range(k+1,n)]
        length = math.hypot(*(abs(value) for value in v))
        if length == 0:
            continue
        sign = v[0]/abs(v[0]) if v[0] != 0 else 1.0
        v = [value/length for value in v]
        v[0] += sign
        length = math.hypot(*(abs(value) for value in v))
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
    scale = max(abs(a), abs(b), abs(c), abs(d))
    if scale == 0:
        return 0j, 0j
    a, b, c, d = a/scale, b/scale, c/scale, d/scale
    middle = (a+d)/2
    difference = cmath.sqrt(((a-d)/2)**2+b*c)
    first, second = middle+difference, middle-difference
    if abs(first) < abs(second):
        first, second = second, first
    if first != 0:
        second = (a*d-b*c)/first
    return first*scale, second*scale


def balance_matrix(a):
    work = copy_matrix(a)
    n = len(work)
    low, high = 0, n

    def exchange(i, j):
        work[i], work[j] = work[j], work[i]
        for row in work:
            row[i], row[j] = row[j], row[i]

    while low < high:
        row = next((i for i in range(low, high)
                    if all(i == j or work[i][j] == 0 for j in range(low, high))), None)
        if row is not None:
            high -= 1
            exchange(row, high)
            continue
        column = next((i for i in range(low, high)
                       if all(i == j or work[j][i] == 0 for j in range(low, high))), None)
        if column is not None:
            exchange(column, low)
            low += 1
            continue
        break
    for _ in range(100*n):
        changed = False
        for i in range(low, high):
            row = math.fsum(abs(work[i][j]) for j in range(low, high) if j != i)
            column = math.fsum(abs(work[j][i]) for j in range(low, high) if j != i)
            if row == 0 or column == 0:
                continue
            exponent = round((math.log2(row)-math.log2(column))/2)
            factor = math.ldexp(1.0, max(-500, min(500, exponent)))
            if row/factor+column*factor >= 0.95*(row+column):
                continue
            new_row = [work[i][j]/factor for j in range(n) if j != i]
            new_column = [work[j][i]*factor for j in range(n) if j != i]
            old_values = [work[i][j] for j in range(n) if j != i]
            old_values += [work[j][i] for j in range(n) if j != i]
            if any(not math.isfinite(abs(new)) or (old != 0 and new == 0)
                   for old, new in zip(old_values, new_row+new_column)):
                continue
            for j in range(n):
                if j != i:
                    work[i][j] /= factor
                    work[j][i] *= factor
            changed = True
        if not changed:
            return work
    raise ValueError("Не удалось завершить балансировку матрицы")


def negligible_subdiagonal(a, active, tolerance):
    i = active-1
    value = abs(a[i][i-1])
    if value == 0:
        return True
    diagonal = abs(a[i-1][i-1])+abs(a[i][i])
    if value > tolerance*diagonal:
        return False
    upper = abs(a[i-1][i])
    ab, ba = max(value, upper), min(value, upper)
    diagonal_difference = abs(a[i-1][i-1]-a[i][i])
    aa = max(abs(a[i][i]), diagonal_difference)
    bb = min(abs(a[i][i]), diagonal_difference)
    total = aa+ab
    return ba*(ab/total) <= max(sys.float_info.min, tolerance*bb*(aa/total))


def qr_algorithm_eigenvalues(a, epsilon, max_iterations=10000):
    validate_square_matrix(a, "A")
    validate_precision(epsilon)
    balanced = balance_matrix(a)
    scale = max(abs(value) for row in balanced for value in row)
    if scale == 0:
        return [0j for _ in a], balanced, 0
    work = hessenberg([[value/scale for value in row] for row in balanced])
    active, iterations, stalled = len(a), 0, 0
    tolerance = max(sys.float_info.epsilon, min(epsilon/scale/(10*len(a)),1e-12))
    while active > 2:
        if negligible_subdiagonal(work, active, tolerance):
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
    work = [[value*scale for value in row] for row in work]
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


def run(data, out):
    print(f"Вариант {data['variant']}")
    run_qr(data)


if __name__ == "__main__":
    run_program(run, 'ЛР 1.5', __file__)
