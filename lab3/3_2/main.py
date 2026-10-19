import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bisect
import math
from common.helpers import (finite_vector, number, plot_svg, sample, solve_tridiagonal)
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


def natural_spline(x, y):
    validate_table(x, y)
    n = len(x)
    h = [b-a for a,b in zip(x, x[1:])]
    slopes = [(y[i+1]-y[i])/h[i] for i in range(n-1)]
    diagonal = [1.0] + [2*(h[i-1]+h[i]) for i in range(1, n-1)] + [1.0]
    rhs = [0.0] + [3*(slopes[i]-slopes[i-1]) for i in range(1, n-1)] + [0.0]
    c = solve_tridiagonal(h[:-1]+[0.0], diagonal, [0.0]+h[1:], rhs)
    segments = []
    for i in range(n-1):
        b = slopes[i] - h[i]*(2*c[i]+c[i+1])/3
        d = (c[i+1]-c[i])/(3*h[i])
        segments.append([y[i], b, c[i], d])
    return segments


def spline_interval(x, point):
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка находится вне области сплайна")
    return min(bisect.bisect_right(x, point)-1, len(x)-2)


def spline_value(x, segments, point):
    i = spline_interval(x, point)
    return polynomial_value(segments[i], point-x[i])


def run(data, out):
    print(f"ЛР 3.2. Вариант {data['variant']}")
    section = data
    x, y = finite_vector(section["x"]), finite_vector(section["y"])
    segments = natural_spline(x,y)
    point = number(section["point"])
    print("\n3.2. Натуральный кубический сплайн: S_i=a+b*t+c*t^2+d*t^3, t=x-x_i")
    print_nodes(x,y)
    for i, coefficients in enumerate(segments):
        print(f"Участок {i+1}, [{x[i]:g}, {x[i+1]:g}]: a,b,c,d = {coefficients}")
        print(f"S_{i+1}(x) = {polynomial_text(coefficients,'t')}, t=x-({x[i]:g})")
    value = spline_value(x, segments, point)
    interval = spline_interval(x,point)
    print(f"S({point})={value:.10f}; номер интервала: {interval+1} (нумерация с 1)")
    checks = [[i,xi,yi,spline_value(x,segments,xi),spline_value(x,segments,xi)-yi]
              for i,(xi,yi) in enumerate(zip(x,y))]
    print("Проверка прохождения через узлы:")
    print_table(["i","x_i","y_i","S(x_i)","S(x_i)-y_i"],
                [[i,f"{xi:.10g}",f"{yi:.10g}",f"{si:.10g}",f"{error:.3e}"] for i,xi,yi,si,error in checks])
    print(f"Максимальная ошибка в узлах: {max(abs(row[-1]) for row in checks):.3e}")
    print(f"S''(x0)={2*segments[0][2]:.3e}; S''(xn)="
          f"{2*segments[-1][2]+6*segments[-1][3]*(x[-1]-x[-2]):.3e}")
    curves = [(f"Участок S_{i+1}: [{x[i]:g}, {x[i+1]:g}]",
               sample(lambda t: polynomial_value(coefficients,t-x[i]),x[i],x[i+1]))
              for i,coefficients in enumerate(segments)]
    plot_svg(out / "spline.svg", "3.2. Кубический сплайн", curves + [
             ("Исходные узлы",list(zip(x,y)),{"scatter":True}),
             (f"x*={point:g}, S(x*)={value:.6g}",[(point,value)],{"scatter":True,"vertical":True})])


if __name__ == "__main__":
    run_program(run, 'ЛР 3.2', __file__)
