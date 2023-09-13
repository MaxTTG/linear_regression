import random

import matplotlib

from typing import Tuple

matplotlib.use("QTAgg")
import matplotlib.pyplot as plt
import numpy as np


def test_data(k: float = 1.0, b: float = 0.1, half_disp: float = 0.05, n: int = 100, x_step: float = 0.01) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Генерируюет линию вида y = k*x + b + dy, где dy - аддитивный шум с амплитудой half_disp
    :param k: наклон линии
    :param b: смещение по y
    :param half_disp: амплитуда разброса данных
    :param n: количество точек
    :param x_step: шаг между соседними точками
    :return: кортеж значенией по x и y
    """
    x = np.arange(0, n) * x_step
    y = k * x + b + half_disp * np.random.random_sample()

    return x, y


def quadratic_regression_test_data_2d(poly_params: Tuple[float, float, float, float, float, float],
                                      args_range: float = 1.0, rand_range: float = 0.1, n_points: int = 1000) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([random.uniform(-0.5 * args_range, 0.5 * args_range) for _ in range(n_points)])
    y = np.array([random.uniform(-0.5 * args_range, 0.5 * args_range) for _ in range(n_points)])
    dz = np.array([poly_params[5] + random.uniform(-0.5 * rand_range, 0.5 * rand_range) for _ in range(n_points)])
    return x, y, poly_params[0] * x * x + poly_params[1] * x * y + poly_params[2] * y * y + poly_params[3] * x + \
           poly_params[4] * y + dz


def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, half_disp: float = 1.01, n: int = 100,
                 x_step: float = 0.01, y_step: float = 0.01) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Генерирует плоскость вида z = kx*x + ky*x + b + dz, где dz - аддитивный шум с амплитудой half_disp
    :param kx: наклон плоскости по x
    :param ky: наклон плоскости по y
    :param b: смещение по z
    :param half_disp: амплитуда разброса данных
    :param n: количество точек
    :param x_step: шаг между соседними точками по х
    :param y_step: шаг между соседними точками по y
    :returns: кортеж значенией по x, y и z
    """
    x = np.random.random_sample(n) * x_step
    y = np.random.random_sample(n) * y_step
    z = kx * x + ky * y + b + half_disp * np.random.random_sample()
    return x, y, z



def distance_k_b(x, y, k, b):
    return np.power(np.power(y - (k * x + b), 2.0).sum(), 0.5)



def distance_field(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b, где k и b являются диапазонами
    значений. Формула расстояния для j-ого значения из набора k и k-ого значения из набора b:
    F(k_j, b_k) = (Σ(yi -(k_j * xi + b_k))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: массив значений параметра k (наклоны)
    :param b: массив значений параметра b (смещения)
    :returns: поле расстояний вида F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    """
    return np.array([[distance_k_b(x, y, ki, bi) for ki in k] for bi in b])


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Линейная регрессия.\n
    Основные формулы:\n
    yi - xi*k - b = ei\n
    yi - (xi*k + b) = ei\n
    (yi - (xi*k + b))^2 = yi^2 - 2*yi*(xi*k + b) + (xi*k + b)^2 = ei^2\n
    yi^2 - 2*(yi*xi*k + yi*b) + (xi^2 * k^2 + 2 * xi * k * b + b^2) = ei^2\n
    yi^2 - 2*yi*xi*k - 2*yi*b + xi^2 * k^2 + 2 * xi * k * b + b^2 = ei^2\n
    d ei^2 /dk = - 2*yi*xi + 2 * xi^2 * k + 2 * xi * b = 0\n
    d ei^2 /db = - 2*yi + 2 * xi * k + 2 * b = 0\n
    ====================================================================================================================\n
    d ei^2 /dk = (yi - xi * k - b) * xi = 0\n
    d ei^2 /db =  yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ(yi - xi * k) = n * b\n
    ====================================================================================================================\n
    Σyi - k * Σxi = n * b\n
    Σxi*yi - xi^2 * k - xi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*(Σyi - k * Σxi) / n = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*Σyi / n + k * (Σxi)^2 / n = 0\n
    Σxi*yi - Σxi*Σyi / n + k * ((Σxi)^2 / n - Σxi^2)  = 0\n
    Σxi*yi - Σxi*Σyi / n = -k * ((Σxi)^2 / n - Σxi^2)\n
    (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n) = k\n
    окончательно:\n
    k = (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n)\n
    b = (Σyi - k * Σxi) /n\n
    :param x: массив значений по x
    :param y: массив значений по y
    :returns: возвращает пару (k, b), которая является решением задачи (Σ(yi -(k * xi + b))^2)->min
    """

    sum_x_y = (x * y).sum()

    sum_x_y = np.add.reduce(np.multiply(x, y))
    sum_x = np.add.reduce(x)
    sum_y = np.add.reduce(y)
    sum_x_2 = np.add.reduce(np.square(x))

    k = (sum_x_y - sum_x * sum_y / x.size) / (sum_x_2 - sum_x ** 2 / len(x))
    b = (sum_y - k * sum_x) / x.size

    return k, b


def bi_linear_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) ->Tuple [float, float, float]:
    """
    Билинейная регрессия.\n
    Основные формулы:\n
    zi - (yi * ky + xi * kx + b) = ei\n
    zi^2 - 2*zi*(yi * ky + xi * kx + b) + (yi * ky + xi * kx + b)^2 = ei^2\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + ((yi*ky)^2 + 2 * (xi*kx*yi*ky + b*yi*ky) + (xi*kx + b)^2)\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx + b)^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b + b^2\n
    ====================================================================================================================\n
    d Σei^2 /dkx = Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b = 0\n
    d Σei^2 /dky = Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi = 0\n
    d Σei^2 /db  = Σ-zi + yi*ky + xi*kx = 0\n
    ====================================================================================================================\n
    d Σei^2 /dkx / dkx = Σ xi^2\n
    d Σei^2 /dkx / dky = Σ xi*yi\n
    d Σei^2 /dkx / db  = Σ xi\n
    ====================================================================================================================\n
    d Σei^2 /dky / dkx = Σ xi*yi\n
    d Σei^2 /dky / dky = Σ yi^2\n
    d Σei^2 /dky / db  = Σ yi\n
    ====================================================================================================================\n
    d Σei^2 /db / dkx = Σ xi\n
    d Σei^2 /db / dky = Σ yi\n
    d Σei^2 /db / db  = n\n

    :param x: массив значений по x
    :param y: массив значений по y
    :param z: массив значений по z
    :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
    """

    H = np.zeros((3, 3))
    '''
    Hesse matrix:\n
                   | Σ xi^2;  Σ xi*yi; Σ xi |\n
    H(kx, ky, b) = | Σ xi*yi; Σ yi^2;  Σ yi |\n
                   | Σ xi;    Σ yi;    n    |\n
    '''
    H[0][0] = np.add.reduce(np.square(x))
    H[0][1] = H[1][0] = np.add.reduce(np.multiply(x, y))
    H[0][2] = H[2][0] = np.add.reduce(x)
    H[1][1] = np.add.reduce(np.square(y))
    H[1][2] = H[2][1] = np.add.reduce(y)
    H[2][2] = len(x)
    '''
                      | Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b |\n
    grad(kx, ky, b) = | Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi |\n
                      | Σ-zi + yi*ky + xi*kx                |\n
    '''
    kx = 1
    ky = 1
    b = 0

    grad = np.zeros((3, 1))
    grad[0] = np.add.reduce(-np.multiply(z, x) + ky * np.multiply(x, y) + kx * np.square(x) + x * b)
    grad[1] = np.add.reduce(-np.multiply(z, y) + kx * np.multiply(x, y) + ky * np.square(y) + y * b)
    grad[2] = np.add.reduce(-z + ky * y + kx * x)
    '''
    |kx|   |1|
    |ky| = |1|   -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
    | b|   |0|
    '''
    result = np.array([[kx], [ky], [b]])
    result = result - np.dot(np.linalg.inv(H), grad)

    return result[0, 0], result[1, 0], result[2, 0]


def quadratic_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Полином: y = Σ_j x^j * bj
    Отклонение: ei =  yi - Σ_j xi^j * bj
    Минимизируем: Σ_i(yi - Σ_j xi^j * bj)^2 -> min
    Σ_i(yi - Σ_j xi^j * bj)^2 = Σ_iyi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2
    условие минимума: d/dbj Σ_i ei = d/dbj (Σ_i yi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2) = 0
    :param x: массив значений по x
    :param y: массив значений по y
    :param order: порядок полинома
    :return: набор коэффициентов bi полинома y = Σx^i*bi
    """

    b = [x * x, x * y, y * y, x, y, np.array([1.0])]
    a_m = np.zeros((6, 6))
    b_c = np.zeros((6,))
    for row in range(6):
        b_c[row] = (b[row] * z).sum()
        for col in range(row + 1):
            a_m[row][col] = (b[row] * b[col]).sum()
            a_m[col][row] = a_m[row][col]

    a_m[5][5] = x.size
    B = np.linalg.inv(a_m) @ b_c.transpose()
    return B


def distance_field_test():
    """
    Функция проверки поля расстояний:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Задать интересующие нас диапазоны k и b (np.linspace...)
    3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.
    4) Проанализировать результат (смысл этой картинки в чём...)
    :return:
    """
    x, y = test_data()
    k = np.linspace(-5, 5)
    b = np.linspace(-5, 5)

    F = distance_field(x, y, k, b)
    plt.imshow(F)
    plt.title("Сумма квадратов растояний до линии")
    plt.show()


def linear_reg_test():
    """
    Функция проверки работы метода линейной регрессии:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Получить с помошью linear_regression значения k и b
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную прямую вида y = k*x + b
    :return:
    """
    x, y = test_data()
    k, b = linear_regression(x, y)
    plt.scatter(x, y, s=3)
    y_r = k * x + b
    plt.plot(x, y_r, color='red')
    plt.title("Проверка работы ленейной регрессии")
    plt.show()


def bi_linear_reg_test():
    """
    Функция проверки работы метода билинейной регрессии:
    1) Посчитать тестовыe x, y и z используя функцию test_data_2d
    2) Получить с помошью bi_linear_regression значения kx, ky и b
    3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить
       регрессионную плоскость вида z = kx*x + ky*y + b
    :return:
    """
    x, y, z = test_data_2d()
    kx, ky, b = bi_linear_regression(x, y, z)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='red')

    x_lin = np.linspace(x.min(), x.max(), 10)
    y_lin = np.linspace(y.min(), y.max(), 10)

    x, y = np.meshgrid(x_lin, y_lin)

    ax.plot_surface(x, y, kx * x + ky * y + b, color='blue')
    plt.title("Массив точек и регрессионная плоскость")
    plt.show()


def quadratic_regression_test():
    """
    Функция проверки работы метода полиномиальной регрессии:
    1) Посчитать тестовыe x, y используя функцию test_data
    2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную кривую. Для построения кривой использовать метод polynom
    """

    x, y, z = quadratic_regression_test_data_2d((1, 2, 3, 1, 2, 3))
    b = quadratic_regression(x, y, z)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='red')

    x_lin = np.linspace(x.min(), x.max())
    y_lin = np.linspace(y.min(), y.max())

    x, y = np.meshgrid(x_lin, y_lin)
    z = b[0] * x * x + b[1] * x * y + b[2] * y * y + b[3] * x + b[4] * y + b[5]
    ax.plot_surface(x, y, z, color='blue')
    plt.title("Проверка квадратичной регрессии")
    plt.show()


def poly_regression(x, y, order):
    A = np.zeros((order + 1, order + 1))
    C = np.zeros(order + 1)
    x_i = x.copy()
    x_j = x.copy()
    for i in range(0, order + 1):
        if i == 0:
            x_i = np.ones_like(x_i)
        else:
            x_i *= x

        C[i] = np.add.reduce(np.multiply(x_i, y))
        for j in range(0, order + 1):
            if j == 0:
                x_j = np.ones_like(x_j)
            else:
                x_j *= x
            A[i][j] = np.add.reduce(x_i * x_j)

    return np.linalg.inv(A) @ C.transpose()


def test_data_poly(b, half_disp: float = 100, n: int = 200, x_step: float = 0.02):
    noise = np.random.random_sample(n) * half_disp * 2 - half_disp
    x = np.arange(0, n) * x_step
    y = polynom(x, b) + noise
    return x, y


def poly_regression_test():
    b = np.array([1, 2, 3, 1, 2, 3])
    x, y = test_data_poly(b)
    b = poly_regression(x, y, 5)
    plt.scatter(x, y, s=3)
    y_r = polynom(x, b)
    plt.plot(x, y_r, color='red')
    plt.title("Полиномиальная регрессия")
    plt.show()


def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
       :param x: массив значений по x
       :param b: массив коэффициентов полинома
       :returns: возвращает полином yi = Σxi^j*bj
    """
    powers = np.arange(0, b.size)
    y = np.zeros(x.size)
    for i in range(0, x.size):
        y[i] = np.add.reduce(np.multiply(np.power(x[i], powers), b))
    return y


def test_data_Nd(kx, x_max, b: float = 1.0, half_disp: float = 0.5, n: int = 100):
    x = np.zeros((n, kx.size))
    for i in range(0, kx.size):
        x[:, i] = np.random.random_sample(n) * x_max[i]
    y = b + np.dot(x, kx) + np.random.random_sample(n) * half_disp * 2 - half_disp
    return x, y


def test_data_polynom_Nd(kx, x_max, dims: int = 3, maxpower: int = 3, half_disp: float = 0.5, n: int = 100):
    maxpower += 1
    x = np.zeros((n, kx.size))
    x_temp = np.zeros((n, dims))
    pull_index = 0

    for i in range(0, dims):
        x_temp[:, i] = np.random.random_sample(n) * x_max[i]

    for i in range(0, (maxpower) ** dims):
        power = np.zeros(dims)
        for j in range(0, dims):
            power[j] = (i // (maxpower ** j)) % (maxpower)
        if np.sum(power) <= maxpower - 1:
            print(f"power_{i}:{power}")
            for k in range(0, n):
                x[k, pull_index] = np.prod(np.power(x_temp[k, :], power))
            pull_index += 1
    y = np.dot(x, kx) + np.random.random_sample(n) * half_disp * 2 - half_disp
    return x, y


def n_linear_regression(x, y):
    return np.dot(np.linalg.pinv(x), y)


def n_reg_test(dim: int = 3):
    kx = np.arange(dim) + 1
    x_max = np.zeros(dim) + 10
    x, y = test_data_Nd(kx, x_max)
    print(f"n_reg_test real:{n_linear_regression(x, y)}")


def n_poly_reg_test():
    k1 = np.array([-10, 0, 1, 3, 0, 2])
    x, y = test_data_polynom_Nd(k1, np.array([10, 10]), dims=2, maxpower=2, half_disp=0.1, n=100)
    k2 = n_linear_regression(x, y)

    print(f"Заданные коэффициенты: {k1}")
    print(f"Полученные коэффициенты: {k2}")


if __name__ == "__main__":
    n_poly_reg_test();
