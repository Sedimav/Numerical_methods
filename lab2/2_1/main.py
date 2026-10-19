import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
from common.helpers import (expression, finite_vector, norm, number, plot_svg, sample, solve_linear)
from common.runner import run_program


def bisection(f, a, b, epsilon):
    fa, fb = f(a), f(b)
    if not a < b or fa * fb > 0 or epsilon <= 0:
        raise ValueError("Нужен отрезок смены знака и epsilon > 0")
    if fa == 0:
        return a
    if fb == 0:
        return b
    for _ in range(10000):
        middle = (a + b) / 2
        fm = f(middle)
        if fm == 0 or (b - a) / 2 <= epsilon:
            return middle
        if fa * fm < 0:
            b = middle
        else:
            a, fa = middle, fm
    raise ValueError("Деление пополам: исчерпан лимит итераций")


def fixed_point(phi, initial, box, q, epsilon, max_iterations):
    n = len(initial)
    if len(phi) != n or len(box) != n or not 0 <= q < 1 or epsilon <= 0:
        raise ValueError("Для итераций нужны согласованные размеры, 0 <= q < 1 и epsilon > 0")
    def inside(x):
        return all(lo <= value <= hi for value, (lo, hi) in zip(x, box))
    if not inside(initial):
        raise ValueError("Начальное приближение вне области сжатия")
    x, history = initial[:], [initial[:]]
    for _ in range(max_iterations):
        new = [function(*x) for function in phi]
        if not inside(new):
            raise ValueError("Итерация вышла из области сжатия")
        history.append(new[:])
        error_bound = q / (1 - q) * norm([v - u for v, u in zip(new, x)])
        x = new
        if error_bound <= epsilon:
            return x, history, error_bound
    raise ValueError("Простая итерация не сошлась за заданное число шагов")


def newton(functions, jacobian, initial, epsilon, max_iterations):
    n = len(initial)
    if len(functions) != n or len(jacobian) != n or any(len(row) != n for row in jacobian):
        raise ValueError("Размерность системы и матрицы Якоби должна совпадать")
    x, history = initial[:], [initial[:]]
    for _ in range(max_iterations):
        residual = [f(*x) for f in functions]
        if norm(residual) == 0:
            return x, history
        matrix = [[df(*x) for df in row] for row in jacobian]
        delta = solve_linear(matrix, [-value for value in residual])
        x = [value + step for value, step in zip(x, delta)]
        if not all(math.isfinite(value) for value in x):
            raise ValueError("Метод Ньютона дал бесконечное приближение")
        history.append(x[:])
        if norm(delta) <= epsilon and norm([f(*x) for f in functions]) <= epsilon:
            return x, history
    raise ValueError("Метод Ньютона не сошелся за заданное число шагов")


def report_history(name, histories, functions, reference, out):
    curves = []
    for method, history in histories:
        rows = []
        print("\n" + name + ": " + method)
        print(" k | приближение | невязка (inf) | ошибка относительно эталона (inf)")
        for k, x in enumerate(history):
            residual = norm([f(*x) for f in functions])
            error = norm([v - r for v, r in zip(x, reference)])
            rows.append([k, *x, residual, error])
            print(f"{k:2d} | " + ", ".join(f"{v:.10f}" for v in x) + f" | {residual:.3e} | {error:.3e}")
        curves.append((method, [(row[0], math.log10(max(row[-1], 1e-16))) for row in rows]))
        print(f"Итераций: {len(history)-1}; решение: {history[-1]}")
    plot_svg(out / (name + "_convergence.svg"), name + ": сходимость", curves, "k", "log10(error)")
    counts = [len(history) - 1 for _, history in histories]
    print(f"Сравнение: простая итерация {counts[0]} шагов, Ньютон {counts[1]} шагов.")


def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    limit = data.get("max_iterations", 10000)
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_iterations должен быть положительным целым")
    scalar = data
    eps = number(scalar["epsilon"])
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon должна быть конечной и положительной")
    f, df, phi = [expression(scalar[key]) for key in ("f", "df", "phi")]
    a, b = finite_vector(scalar["interval"])
    initial = [number(scalar["initial"])]
    q = number(scalar["q"])
    reference = [bisection(f, a, b, min(1e-13, eps * .001))]
    simple, hs, bound = fixed_point([phi], initial, [(a, b)], q, eps, limit)
    result, hn = newton([f], [[df]], initial, eps, limit)
    if max(abs(simple[0]-reference[0]), abs(result[0]-reference[0])) > eps + 1e-13:
        raise ValueError("Решение уравнения не прошло проверку точности")
    print(f"ЛР 2, вариант {data['variant']}. 2.1: {scalar['f']} = 0")
    print(f"epsilon={eps:g}; интервал=[{a}, {b}]; x0={initial[0]}; q={q:g}")
    print(f"Эталон (собственное деление пополам): {reference[0]:.13f}")
    print(f"Оценка ошибки простых итераций: {bound:.3e}")
    report_history("scalar", [("iteration", hs), ("newton", hn)], [f], reference, out)
    plot_svg(out / "scalar_initial.svg", "2.1. Графическое выделение корня",
             [("f(x)", sample(f, a, b)), ("x0", [(initial[0], f(initial[0]))])])


if __name__ == "__main__":
    run_program(run, 'ЛР 2.1', __file__)
