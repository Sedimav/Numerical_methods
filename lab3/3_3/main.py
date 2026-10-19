import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.helpers import (finite_vector, plot_svg, sample, solve_linear)
from common.runner import run_program


def validate_table(x, y, minimum=2):
    if len(x) < minimum or len(x) != len(y) or any(b <= a for a,b in zip(x, x[1:])):
        raise ValueError("Нужны согласованные таблицы со строго возрастающими узлами")
    if not all(math.isfinite(v) for v in x + y):
        raise ValueError("Таблица должна содержать конечные значения")


def polynomial_value(coefficients, x):
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * x + coefficient
    return result


def polynomial_text(coefficients, variable="x"):
    terms = []
    for power, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        factor = "" if power == 0 else "*" + variable + (f"^{power}" if power > 1 else "")
        term = f"{abs(coefficient):.12g}" + factor
        if not terms:
            terms.append(("-" if coefficient < 0 else "") + term)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(terms) or "0"


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


def least_squares(x, y, degree):
    validate_table(x, y)
    if type(degree) is not int or not 0 <= degree < len(x):
        raise ValueError("Степень МНК должна быть целой и меньше числа узлов")
    moments = [sum(value**k for value in x) for k in range(2*degree+1)]
    matrix = [[moments[i+j] for j in range(degree+1)] for i in range(degree+1)]
    rhs = [sum(yi*xi**k for xi,yi in zip(x,y)) for k in range(degree+1)]
    coefficients = solve_linear(matrix, rhs)
    sse = sum((polynomial_value(coefficients, xi)-yi)**2 for xi,yi in zip(x,y))
    return coefficients, sse, matrix, rhs


def run(data, out):
    print(f"ЛР 3.3. Вариант {data['variant']}")
    section = data
    x, y = finite_vector(section["x"]), finite_vector(section["y"])
    curves, rows = [("Исходные данные", list(zip(x,y)), {"scatter":True})], []
    print("\n3.3. МНК через нормальные системы")
    print_nodes(x,y)
    degrees = section["degrees"]
    if not {1,2,3}.issubset(degrees):
        raise ValueError("По требованиям ЛР 3 нужно включить степени 1, 2 и 3")
    for degree in degrees:
        coefficients, sse, matrix, rhs = least_squares(x,y,degree)
        print(f"Степень {degree}; нормальная система:")
        for row, right in zip(matrix,rhs):
            print(row, "|", right)
        print(f"Коэффициенты: {coefficients}; сумма квадратов ошибок: {sse:.10f}")
        mse = sse/len(x)
        print(f"P_{degree}(x) = {polynomial_text(coefficients)}")
        print(f"MSE{degree} = SSE/{len(x)} = {mse:.12g}")
        rows.append([degree, sse, mse, *coefficients])
        curves.append((f"МНК, степень {degree}", sample(lambda t: polynomial_value(coefficients,t),x[0],x[-1])))
    print("Сравнение ошибок по всем исходным точкам:")
    print_table(["Степень","SSE","MSE"],[[row[0],f"{row[1]:.12g}",f"{row[2]:.12g}"] for row in rows])
    best = min(rows,key=lambda row:row[2])
    print(f"Лучшее приближение исходных данных по MSE: P_{best[0]}(x), MSE={best[2]:.12g}")
    plot_svg(out / "least_squares.svg", "3.3. Приближение методом наименьших квадратов",curves)


if __name__ == "__main__":
    run_program(run, 'ЛР 3.3', __file__)
