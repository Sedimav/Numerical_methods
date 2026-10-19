import ast
import json
import math
import operator
from pathlib import Path


FUNCTIONS = {name: getattr(math, name) for name in (
    "sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "exp", "log", "log10"
)}
FUNCTIONS["abs"] = abs
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


def expression(source, variables=("x",), parameters=None):
    tree = ast.parse(str(source), mode="eval").body
    constants = {"pi": math.pi, "e": math.e, **(parameters or {})}

    def evaluate(node, values):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            if node.id in constants:
                return constants[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.left, values), evaluate(node.right, values))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return (1 if isinstance(node.op, ast.UAdd) else -1) * evaluate(node.operand, values)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS and not node.keywords and len(node.args) == 1):
            return FUNCTIONS[node.func.id](evaluate(node.args[0], values))
        raise ValueError("Недопустимая формула: " + str(source))

    def function(*args):
        if len(args) != len(variables):
            raise ValueError("Неверное число аргументов формулы")
        result = float(evaluate(tree, dict(zip(variables, args))))
        if not math.isfinite(result):
            raise ValueError("Формула дала нечисловое или бесконечное значение")
        return result
    return function


def number(value):
    return expression(value, ())() if isinstance(value, str) else float(value)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def finite_vector(values, name="vector"):
    result = [number(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(name + ": требуется непустой набор конечных чисел")
    return result


def norm(values):
    return max(abs(value) for value in values)


def solve_linear(matrix, rhs):
    n = len(rhs)
    if not n or len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Несовместимые размерности СЛАУ")
    a = [finite_vector(row, "matrix") for row in matrix]
    b = finite_vector(rhs, "rhs")
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(a[i][k]))
        if a[pivot][k] == 0:
            raise ValueError("Вырожденная матрица")
        a[k], a[pivot] = a[pivot], a[k]
        b[k], b[pivot] = b[pivot], b[k]
        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            a[i][k] = 0.0
            for j in range(k + 1, n):
                a[i][j] -= factor * a[k][j]
            b[i] -= factor * b[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    if not all(math.isfinite(value) for value in x):
        raise ValueError("Не удалось получить конечное решение СЛАУ")
    return x


def solve_tridiagonal(lower, diagonal, upper, rhs):
    n = len(diagonal)
    if not n or len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("Неверные длины диагоналей")
    alpha, beta = [0.0] * n, [0.0] * n
    for i in range(n):
        denominator = diagonal[i] + (lower[i - 1] * alpha[i - 1] if i else 0.0)
        if denominator == 0:
            raise ValueError("Нулевой знаменатель прогонки")
        alpha[i] = -upper[i] / denominator if i < n - 1 else 0.0
        beta[i] = (rhs[i] - (lower[i - 1] * beta[i - 1] if i else 0.0)) / denominator
    result = beta[:]
    for i in range(n - 2, -1, -1):
        result[i] += alpha[i] * result[i + 1]
    return result


def plot_svg(path, title, series, xlabel="x", ylabel="y"):
    from matplotlib.figure import Figure

    series = [(item[0], item[1], item[2] if len(item) == 3 else {}) for item in series]
    points = [p for _, line, _ in series for p in line]
    if not points or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("Неверные данные графика")

    figure = Figure(figsize=(9, 6))
    axes = figure.subplots()
    for label, points, style in series:
        x, y = zip(*points)
        linestyle = "none" if style.get("scatter") else "--" if style.get("dash") else "-"
        marker = "o" if style.get("scatter") or len(points) < 30 else None
        curve, = axes.plot(x, y, label=label, linestyle=linestyle, marker=marker)
        if style.get("vertical"):
            for value in x:
                axes.axvline(value, color=curve.get_color(), linestyle=":")
    axes.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes.grid(True, alpha=0.3)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    figure.tight_layout()
    figure.savefig(path)


def sample(function, a, b, count=301):
    return [(a + (b-a)*i/(count-1), function(a + (b-a)*i/(count-1))) for i in range(count)]
