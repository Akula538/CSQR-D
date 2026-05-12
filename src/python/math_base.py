import numpy as np
from tools import*

def fit_parabola_from4(points, tol=1e-12, maxiter=100):
    P = np.asarray(points, dtype=float)
    if P.shape != (4,2):
        raise ValueError("Ожидаются 4 точки shape (4,2).")
    P0, P1, P2, P3 = P
    A = P0.copy()                 # t0 = 0 -> r(0)=A=P0
    D = P3 - P0                   # D = B + C because r(1)=A+B+C = P3 -> B+C = P3-A = D

    # функция F([t1,t2]) возвращает 2-мерный вектор: V1*s2 - V2*s1
    def F(tt):
        t1, t2 = tt
        s1 = t1*t1 - t1
        s2 = t2*t2 - t2
        V1 = (P1 - A) - D * t1
        V2 = (P2 - A) - D * t2
        return V1 * s2 - V2 * s1   # vector length 2

    # начальные приближения
    guesses = [
        (0.25, 0.5),
        (0.2, 0.8),
        (0.33, 0.66),
        (0.15, 0.4),
        (0.4, 0.7),
    ]
    def newton_solve(initial):
        t = np.array(initial, dtype=float)
        for k in range(maxiter):
            f = F(t)
            fnorm = np.linalg.norm(f)
            if fnorm < tol:
                return t, True
            # численный якобиан (2x2)
            eps = 1e-8 if k<5 else 1e-9
            J = np.zeros((2,2))
            for i in range(2):
                dt = np.zeros(2); dt[i] = eps
                J[:,i] = (F(t + dt) - f) / eps
            # решаем J * delta = -f
            try:
                delta = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return t, False
            # шаг-контроль: не дать выйти за [ -1, 2 ] диапазон чрезмерно
            # уменьшаем шаг если не уменьшается норма функции
            alpha = 1.0
            for _ in range(20):
                t_new = t + alpha*delta
                # предотвращаем попадание в s ~ 0 (t near 0 or 1) -- отклоняем такие шаги
                if not (-2.0 < t_new[0] < 3.0 and -2.0 < t_new[1] < 3.0):
                    alpha *= 0.5
                    continue
                f_new = F(t_new)
                if np.linalg.norm(f_new) < fnorm + 1e-15:
                    t = t_new
                    break
                alpha *= 0.5
            else:
                return t, False
        return t, False

    solution = None
    for g in guesses:
        tsol, ok = newton_solve(g)
        if not ok:
            continue
        t1, t2 = tsol
        # проверяем монотонность и разумность
        if 0.0 < t1 < t2 < 1.0:
            solution = (t1, t2)
            break
    if solution is None:
        # попробуем принять решения, которые не строго в (0,1) но близки
        for g in guesses:
            tsol, ok = newton_solve(g)
            if ok:
                t1, t2 = tsol
                solution = (t1, t2)
                break
    if solution is None:
        raise RuntimeError("Не удалось найти параметры t1,t2 методом Ньютона (попробуйте другие точки).")

    t1, t2 = solution
    s1 = t1*t1 - t1
    # вычисляем C из первого уравнения
    V1 = (P1 - A) - D * t1
    if abs(s1) < 1e-10:
        # for i in points:
            # print_as_point(i)
        raise RuntimeError(f"Получили слишком маленькое s1; решение вырождается.")
    C = V1 / s1
    B = D - C
    # проверка: восстановленные точки
    def r(t): return A + B*t + C*(t*t)
    reconstructed = np.vstack([r(0), r(t1), r(t2), r(1)])
    err = np.max(np.linalg.norm(reconstructed - P, axis=1))
    return A, B, C