import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bisect
import math
from common.helpers import (finite_vector, number)
from common.runner import run_program


def validate_table(x, y, minimum=2):
    if len(x) < minimum or len(x) != len(y) or any(b <= a for a,b in zip(x, x[1:])):
        raise ValueError("Нужны согласованные таблицы со строго возрастающими узлами")
    if not all(math.isfinite(v) for v in x + y):
        raise ValueError("Таблица должна содержать конечные значения")


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


def differentiate_table(x, y, point):
    validate_table(x, y, 3)
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка дифференцирования вне таблицы")
    i = min(max(bisect.bisect_left(x, point)-1, 0), len(x)-3)
    slope0 = (y[i+1]-y[i])/(x[i+1]-x[i])
    slope1 = (y[i+2]-y[i+1])/(x[i+2]-x[i+1])
    second_difference = (slope1-slope0)/(x[i+2]-x[i])
    first = slope0 + second_difference*(2*point-x[i]-x[i+1])
    return first, 2*second_difference, i, slope0, slope1


def run(data, out):
    print(f"ЛР 3.4. Вариант {data['variant']}")
    section = data
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


if __name__ == "__main__":
    run_program(run, 'ЛР 3.4', __file__)
