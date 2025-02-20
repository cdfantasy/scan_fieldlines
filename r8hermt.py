import numpy as np
import r8akherm3
import r8herm3

def r8herm_spline(x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray, B_raw: np.ndarray) -> np.ndarray:
    """
    使用三维 Hermite 样条插值函数对输入数据进行插值。

    参数:
    x_array (numpy.ndarray): x 轴上的一维数据点数组。
    y_array (numpy.ndarray): y 轴上的一维数据点数组。
    z_array (numpy.ndarray): z 轴上的一维数据点数组。
    B_raw (numpy.ndarray): 初始三维数据数组。

    返回:
    numpy.ndarray: 插值后的数据数组。

    注意:
    该函数依赖于外部库 r8akherm3。
    """
    nx = x_array.shape[0]
    ny = y_array.shape[0]
    nz = z_array.shape[0]
    ilinx, iliny, ilinz, ier = 0, 0, 0, 0
    fherm = np.zeros((8, nx, ny, nz))
    fherm[0, :, :, :] = B_raw
    x = np.asfortranarray(x_array)
    y = np.asfortranarray(y_array)
    z = np.asfortranarray(z_array)
    fherm = np.asfortranarray(fherm)
    
    r8akherm3.r8akherm3(x, y, z, fherm, ilinx, iliny, ilinz, ier)

    return fherm

def r8herm_interpolation(xget: float, yget: float, zget: float, x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray, fherm: np.ndarray) -> float:
    """
    使用三维 Hermite 样条插值函数对给定点进行插值。

    参数:
    xget (float): 插值点的 x 坐标。
    yget (float): 插值点的 y 坐标。
    zget (float): 插值点的 z 坐标。
    x_array (numpy.ndarray): x 轴上的一维数据点数组。
    y_array (numpy.ndarray): y 轴上的一维数据点数组。
    z_array (numpy.ndarray): z 轴上的一维数据点数组。
    fherm (numpy.ndarray): Hermite 样条插值后的数据数组。

    返回:
    float: 插值点的插值值。
    """
    # 初始化插值参数
    ilinx, iliny, ilinz, ier = 0, 0, 0, 0
    ict = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=int)  # 插值控制参数
    fval = np.zeros(1, dtype=float)  # 存储插值结果的数组
    x = np.asfortranarray(x_array)  # 将 x_array 转换为 Fortran 风格数组
    y = np.asfortranarray(y_array)  # 将 y_array 转换为 Fortran 风格数组
    z = np.asfortranarray(z_array)  # 将 z_array 转换为 Fortran 风格数组

    # 调用外部库函数进行插值计算
    r8herm3.r8herm3ev(xget, yget, zget, x, y, z, ilinx, iliny, ilinz, fherm, ict, fval, ier)
    
    return fval[0]  # 返回插值结果