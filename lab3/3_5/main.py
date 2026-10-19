import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.helpers import (expression, finite_vector)
from common.runner import run_program


def print_table(headers, rows):
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in text_rows)) for i,header in enumerate(headers)]
    def line(values):
        return " | ".join(value.ljust(width) for value,width in zip(values,widths))
    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(line(row))


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
    print(f"ЛР 3.5. Вариант {data['variant']}")
    section = data
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


if __name__ == "__main__":
    run_program(run, 'ЛР 3.5', __file__)
