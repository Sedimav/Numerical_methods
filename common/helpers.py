import ast
import html
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
    """Read only arithmetic expressions; no eval, attributes or arbitrary calls."""
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
    """Gaussian elimination with partial pivoting, arbitrary square size."""
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
    """Series: (label, points[, style]); style supports scatter, dash and vertical."""
    series = [(item[0], item[1], item[2] if len(item) == 3 else {}) for item in series]
    points = [p for _, line, _ in series for p in line]
    if not points or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("Неверные данные графика")
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    dx, dy = xmax - xmin or 1.0, ymax - ymin or 1.0
    ymin, ymax = ymin - .06 * dy, ymax + .06 * dy
    def px(x):
        return 90 + 760 * (x - xmin) / dx
    def py(y):
        return 430 - 340 * (y - ymin) / (ymax - ymin)
    height = max(570, 520 + 22 * len(series))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 {height}">',
             f'<rect width="920" height="{height}" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="13" fill="#222">',
             '<text x="90" y="32" font-size="19">' + html.escape(title) + '</text>']
    for i in range(6):
        x, y = xmin + dx * i / 5, ymin + (ymax - ymin) * i / 5
        parts.append(f'<path d="M {px(x):.2f} 90 V 430 M 90 {py(y):.2f} H 850" stroke="#ddd"/>')
        parts.append(f'<text x="{px(x):.2f}" y="452" text-anchor="middle">{x:.5g}</text>')
        parts.append(f'<text x="80" y="{py(y)+4:.2f}" text-anchor="end">{y:.5g}</text>')
    if ymin <= 0 <= ymax:
        parts.append(f'<path d="M 90 {py(0):.2f} H 850" stroke="#888"/>')
    colors = ["#12766d", "#b52e49", "#435ec4", "#a26718", "#555"]
    for i, (label, line, style) in enumerate(series):
        color = colors[i % len(colors)]
        coords = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in line)
        dash = ' stroke-dasharray="8 5"' if style.get("dash") else ""
        if not style.get("scatter"):
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        if style.get("vertical"):
            parts.extend(f'<path d="M {px(x):.2f} 90 V 430" stroke="{color}" stroke-dasharray="3 5"/>' for x,_ in line)
        if style.get("scatter") or len(line) < 30:
            parts.extend(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="{color}" stroke="white"/>' for x,y in line)
        legend_y = 495 + 22*i
        if style.get("scatter"):
            parts.append(f'<circle cx="102" cy="{legend_y-4}" r="4" fill="{color}"/>')
        else:
            parts.append(f'<path d="M 90 {legend_y-4} H 114" stroke="{color}" stroke-width="2"{dash}/>')
        parts.append(f'<text x="125" y="{legend_y}" fill="{color}">{html.escape(label)}</text>')
    parts.extend([f'<text x="875" y="474" text-anchor="end">{html.escape(xlabel)}</text>',
                  f'<text x="15" y="65">{html.escape(ylabel)}</text>', '</g></svg>'])
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def sample(function, a, b, count=301):
    return [(a + (b-a)*i/(count-1), function(a + (b-a)*i/(count-1))) for i in range(count)]
