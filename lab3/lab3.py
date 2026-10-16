import argparse
import bisect
import contextlib
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.helpers import (expression, finite_vector, number, plot_svg, read_json,
                            sample, solve_linear, solve_tridiagonal)


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


def multiply_polynomials(a, b):
    result = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i+j] += ai * bj
    return result


def lagrange(x, y):
    validate_table(x, y)
    coefficients, weights = [0.0] * len(x), []
    for i in range(len(x)):
        basis, denominator = [1.0], 1.0
        for j in range(len(x)):
            if i != j:
                basis = multiply_polynomials(basis, [-x[j], 1.0])
                denominator *= x[i] - x[j]
        weight = y[i] / denominator
        weights.append(weight)
        for k, value in enumerate(basis):
            coefficients[k] += weight * value
    return coefficients, weights


def newton_polynomial(x, y):
    validate_table(x, y)
    n = len(x)
    differences = [y[:]]
    for order in range(1, n):
        previous = differences[-1]
        differences.append([(previous[i+1] - previous[i]) / (x[i+order] - x[i])
                            for i in range(n-order)])
    coefficients, basis = [0.0] * n, [1.0]
    for k in range(n):
        for j, value in enumerate(basis):
            coefficients[j] += differences[k][0] * value
        basis = multiply_polynomials(basis, [-x[k], 1.0])
    return coefficients, differences


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


def differentiate_table(x, y, point):
    validate_table(x, y, 3)
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка дифференцирования вне таблицы")
    # At an interior node use the interval on its left and a centered triple.
    i = min(max(bisect.bisect_left(x, point)-1, 0), len(x)-3)
    slope0 = (y[i+1]-y[i])/(x[i+1]-x[i])
    slope1 = (y[i+2]-y[i+1])/(x[i+2]-x[i+1])
    second_difference = (slope1-slope0)/(x[i+2]-x[i])
    first = slope0 + second_difference*(2*point-x[i]-x[i+1])
    return first, 2*second_difference, i, slope0, slope1


def integrate(function, a, b, step, method):
    if not all(math.isfinite(v) for v in (a,b,step)) or not a < b or step <= 0:
        raise ValueError("Нужны конечные a < b и h > 0")
    count = round((b-a)/step)
    if count < 1 or not math.isclose(count*step, b-a, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Шаг должен делить длину отрезка")
    if method == "left":
        return step * sum(function(a+i*step) for i in range(count))
    if method == "right":
        return step * sum(function(a+(i+1)*step) for i in range(count))
    if method == "midpoint":
        return step * sum(function(a+(i+.5)*step) for i in range(count))
    if method == "trapezoid":
        return step * ((function(a)+function(b))/2 + sum(function(a+i*step) for i in range(1,count)))
    if method == "simpson":
        if count % 2:
            raise ValueError("Для Симпсона число интервалов должно быть четным")
        return step/3 * (function(a)+function(b)
                        + sum((4 if i%2 else 2)*function(a+i*step) for i in range(1,count)))
    raise ValueError("Неизвестный метод интегрирования")


def integration_results(function, a, b, coarse_h, fine_h, exact=None):
    if not coarse_h > fine_h > 0:
        raise ValueError("Требуются шаги h1 > h2 > 0")
    ratio = coarse_h/fine_h
    if not math.isclose(ratio, round(ratio), rel_tol=1e-10):
        raise ValueError("Для сравнения сеток требуется целое отношение шагов")
    rows = []
    for method, order in (("left",1),("right",1),("midpoint",2),("trapezoid",2),("simpson",4)):
        coarse, fine = [integrate(function,a,b,h,method) for h in (coarse_h,fine_h)]
        correction = (fine-coarse)/(ratio**order-1)
        extrapolated = fine+correction
        rows.append({"method": method, "p": order, "I_h1": coarse, "I_h2": fine,
                     "I_RR": extrapolated, "RR_error_h2": abs(correction),
                     "exact": exact,
                     "abs_error_h1": abs(coarse-exact) if exact is not None else None,
                     "abs_error_h2": abs(fine-exact) if exact is not None else None,
                     "abs_error_RR": abs(extrapolated-exact) if exact is not None else None})
    return rows


def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    print(f"ЛР 3. Вариант {data['variant']}")
    print("Коэффициенты полиномов приводятся в порядке [a0, a1, ..., an].")
    interpolation = data["interpolation"]
    function = expression(interpolation["function"])
    point = number(interpolation["point"])
    for index, nodes in enumerate(interpolation["node_sets"], start=1):
        x = finite_vector(nodes)
        y = [function(value) for value in x]
        lc, weights = lagrange(x, y)
        nc, table = newton_polynomial(x, y)
        lv, nv, exact = polynomial_value(lc, point), polynomial_value(nc, point), function(point)
        print(f"\n3.1. Набор {index}; x={x}; y={y}; X*={point:.10f}")
        print_nodes(x,y)
        print("Лагранж: коэффициенты:", lc)
        print("L(x) = " + polynomial_text(lc))
        print("Веса y_i / произведение (x_i-x_j):", weights)
        print("Ньютон: коэффициенты:", nc)
        print("N(x) = " + polynomial_text(nc))
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
                  ("Лагранж L(x)", sample(lambda t: polynomial_value(lc,t), left, right)),
                  ("Ньютон N(x)", sample(lambda t: polynomial_value(nc,t), left, right), {"dash":True}),
                  ("Узлы интерполяции", list(zip(x,y)), {"scatter":True}),
                  (f"x*={point:.6g}, f(x*)={exact:.6g}", [(point,exact)], {"scatter":True,"vertical":True})])

    section = data["spline"]
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

    section = data["least_squares"]
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

    section = data["differentiation"]
    x, y = finite_vector(section["x"]), finite_vector(section["y"])
    point = number(section["point"])
    first, second, i, left_slope, right_slope = differentiate_table(x,y,point)
    print(f"\n3.4. Производные в X*={point}; выбранные узлы: {x[i:i+3]}")
    print("Исходная таблица:")
    print_nodes(x,y)
    print("Использованные узлы и значения:")
    print_nodes(x[i:i+3],y[i:i+3],start=i)
    print(f"Левый и правый секущие наклоны: {left_slope:.10f}, {right_slope:.10f}")
    print(f"f'(X*) ~ {first:.10f}; f''(X*) ~ {second:.10f}")
    print("Производные вычислены от локального квадратичного интерполянта.")

    section = data["integration"]
    f = expression(section["function"])
    a,b = finite_vector(section["interval"])
    steps = sorted(finite_vector(section["steps"]), reverse=True)
    if len(steps) != 2 or steps[0] <= steps[1]:
        raise ValueError("Требуются два различных положительных шага")
    coarse_h, fine_h = steps
    exact = None
    if "primitive" in section:
        primitive = expression(section["primitive"])
        exact = primitive(b)-primitive(a)
    print(f"\n3.5. Интеграл {section['function']} на [{a}, {b}]; h={steps}")
    if exact is not None:
        print(f"Аналитический эталон из первообразной: {exact:.12f}")
    rows = integration_results(f,a,b,coarse_h,fine_h,exact)
    names = {"left":"Левые прямоуг.","right":"Правые прямоуг.","midpoint":"Средние прямоуг.",
             "trapezoid":"Трапеции","simpson":"Симпсон"}
    print(f"h1={coarse_h:g}; h2={fine_h:g}")
    print_table(["Метод","I(h1)","I(h2)","p","I_RR","Оценка ошибки h2"],
                [[names[row["method"]],f'{row["I_h1"]:.12f}',f'{row["I_h2"]:.12f}',row["p"],
                  f'{row["I_RR"]:.12f}',f'{row["RR_error_h2"]:.6e}'] for row in rows])
    if exact is not None:
        print("Абсолютные ошибки относительно точного интеграла:")
        print_table(["Метод","Ошибка h1","Ошибка h2","Ошибка I_RR"],
                    [[names[row["method"]],*[f'{row[key]:.6e}' for key in
                      ("abs_error_h1","abs_error_h2","abs_error_RR")]] for row in rows])
        for key,label in (("abs_error_h1","h1"),("abs_error_h2","h2"),("abs_error_RR","после Рунге-Ромберга")):
            best = min(rows,key=lambda row:row[key])
            print(f'Наиболее точный метод ({label}): {names[best["method"]]}, ошибка {best[key]:.6e}')
    print("\nРезультаты сохранены в папке " + out.name)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="ЛР 3: приближение, производные и интеграл")
    parser.add_argument("input", nargs="?", type=Path, default=base / "data/lab3.json")
    parser.add_argument("--output", type=Path, default=base / "results")
    args = parser.parse_args()
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run(read_json(args.input),args.output)
        report = buffer.getvalue()
        (args.output / "report.txt").write_text(report, encoding="utf-8")
        print(report, end="")
    except (ValueError,KeyError,TypeError,SyntaxError,OSError,ZeroDivisionError,OverflowError) as error:
        parser.exit(1,"Ошибка: " + str(error) + "\n")


if __name__ == "__main__":
    main()
