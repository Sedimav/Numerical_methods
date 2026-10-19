import math


def copy_matrix(a):
    return [row[:] for row in a]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a):
    rows = len(a)
    cols = len(a[0])
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matmul(a, b):
    rows = len(a)
    inner = len(b)
    cols = len(b[0])
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            for j in range(cols):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matvec(a, x):
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def vector_norm_inf(x):
    return max(abs(value) for value in x) if x else 0.0


def matrix_offdiag_norm(a):
    total = 0.0
    for i in range(len(a)):
        for j in range(len(a)):
            if i != j:
                total += a[i][j] * a[i][j]
    return math.sqrt(total)


def format_number(value, digits=6):
    if isinstance(value, complex):
        if value.imag == 0:
            return format_number(value.real,digits)
        sign = "+" if value.imag >= 0 else "-"
        return f"{format_number(value.real,digits)} {sign} {format_number(abs(value.imag),digits)}i"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def print_vector(name, vector, digits=6):
    print(name)
    print("[" + ", ".join(format_number(value,digits) for value in vector) + "]")


def print_matrix(name, matrix, digits=6):
    print(name)
    for row in matrix:
        print("[" + "  ".join(format_number(value,digits) for value in row) + "]")


def validate_square_matrix(a, name):
    if not a or any(len(row) != len(a) for row in a):
        raise ValueError(f"{name} должна быть непустой квадратной матрицей")
    if not all(math.isfinite(abs(value)) for row in a for value in row):
        raise ValueError(f"{name} содержит нечисловые или бесконечные значения")
