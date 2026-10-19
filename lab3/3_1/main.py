import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.helpers import (expression, finite_vector, number, plot_svg, sample)
from common.runner import run_program


def validate_table(x, y, minimum=2):
    if len(x) < minimum or len(x) != len(y) or any(b <= a for a,b in zip(x, x[1:])):
        raise ValueError("Нужны согласованные таблицы со строго возрастающими узлами")
    if not all(math.isfinite(v) for v in x + y):
        raise ValueError("Таблица должна содержать конечные значения")


def lagrange_value(x, y, point):
    if point in x:
        return y[x.index(point)]
    return math.fsum(y[i] * math.prod((point-x[j])/(x[i]-x[j])
                                    for j in range(len(x)) if j != i)
                     for i in range(len(x)))


def newton_value(x, differences, point):
    result = differences[-1][0]
    for k in range(len(x)-2, -1, -1):
        result = differences[k][0] + (point-x[k])*result
    return result


def lagrange_text(x, y):
    terms = []
    for i, value in enumerate(y):
        if value != 0:
            factors = [f"(x-({x[j]!r}))/({x[i]-x[j]!r})"
                       for j in range(len(x)) if j != i]
            terms.append(f"({value!r})*" + "*".join(factors))
    return " + ".join(terms) or "0"


def newton_text(x, differences):
    terms = []
    for k, row in enumerate(differences):
        if row[0] != 0:
            factors = "".join(f"*(x-({x[j]!r}))" for j in range(k))
            terms.append(f"({row[0]!r})" + factors)
    return " + ".join(terms) or "0"


def print_table(headers, rows):
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in text_rows)) for i,header in enumerate(headers)]
    def line(values):
        return " | ".join(value.ljust(width) for value,width in zip(values,widths))
    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(line(row))


def print_nodes(x, y, start=0):
    print_table(["i", "x_i", "y_i"],
                [[i, f"{xi:.12g}", f"{yi:.12g}"] for i,(xi,yi) in enumerate(zip(x,y),start=start)])


def divided_differences(x, y):
    validate_table(x, y)
    n = len(x)
    differences = [y[:]]
    for order in range(1, n):
        previous = differences[-1]
        differences.append([(previous[i+1] - previous[i]) / (x[i+order] - x[i])
                            for i in range(n-order)])
    return differences


def run(data, out):
    print(f"ЛР 3.1. Вариант {data['variant']}")
    interpolation = data
    function = expression(interpolation["function"])
    point = number(interpolation["point"])
    for index, nodes in enumerate(interpolation["node_sets"], start=1):
        x = finite_vector(nodes)
        y = [function(value) for value in x]
        table = divided_differences(x, y)
        lv = lagrange_value(x, y, point)
        nv = newton_value(x, table, point)
        exact = function(point)
        print(f"\n3.1. Набор {index}; x={x}; y={y}; X*={point:.10f}")
        print_nodes(x,y)
        print("L(x) = " + lagrange_text(x, y))
        print("N(x) = " + newton_text(x, table))
        print("Таблица разделенных разностей по порядкам:")
        for order, row in enumerate(table):
            print(order, row)
        print(f"L(X*)={lv:.10f}; N(X*)={nv:.10f}; f(X*)={exact:.10f}")
        print(f"Фактические ошибки: Лагранж {abs(lv-exact):.6e}; Ньютон {abs(nv-exact):.6e}")
        print(f"L(X*) - N(X*) = {lv-nv:.12e}; |L(X*) - N(X*)| = {abs(lv-nv):.12e}")
        if "derivative_bound" in interpolation:
            bound = number(interpolation["derivative_bound"])
            if bound < 0:
                raise ValueError("Оценка модуля производной должна быть неотрицательной")
            remainder = bound/math.factorial(len(x)) * abs(math.prod(point-xi for xi in x))
            print(f"Остаток при |f^({len(x)})| <= {bound}: |R| <= {remainder:.6e}")
        left, right = min(x[0],point), max(x[-1],point)
        plot_svg(out / f"interpolation_{index}.svg", f"3.1. Интерполяция, набор {index}",
                 [("Исходная функция f(x)", sample(function, left, right)),
                  ("Лагранж L(x)", sample(lambda t: lagrange_value(x,y,t), left, right)),
                  ("Ньютон N(x)", sample(lambda t: newton_value(x,table,t), left, right), {"dash":True}),
                  ("Узлы интерполяции", list(zip(x,y)), {"scatter":True}),
                  (f"x*={point:.6g}, f(x*)={exact:.6g}", [(point,exact)], {"scatter":True,"vertical":True})])


if __name__ == "__main__":
    run_program(run, 'ЛР 3.1', __file__)
