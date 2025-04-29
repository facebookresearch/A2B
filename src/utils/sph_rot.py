import numpy as np

np.set_printoptions(precision=4)


def euler2rotationMatrix(alpha: float, beta: float, gamma: float, convention: str):
    """
    alpha: first angle of rotation in radians
    beta: second angle of rotation in radians
    gamma: third angle of rotation in radians
    convention: definition of the axes of rotation, e.g.
                for the y-convention this should be 'zyz',
                for the x-convnention 'zxz', and
                for the yaw-pitch-roll convention 'zyx'
    """
    Rx = lambda theta: np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), np.sin(theta)],
            [0.0, -np.sin(theta), np.cos(theta)],
        ]
    )
    Ry = lambda theta: np.array(
        [
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    Rz = lambda theta: np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    if convention == "zyz":
        R = Rz(gamma) @ Ry(beta) @ Rz(alpha)
    elif convention == "zxz":
        R = Rz(gamma) @ Rx(beta) @ Rz(alpha)
    elif convention == "zyx":
        R = Rx(gamma) @ Ry(beta) @ Rz(alpha)
    else:
        raise ValueError("Convention not supported")
    return R


def getSHrotMtx(Rxyz: np.ndarray, Nord: int, basisType: str="real"):
    """
    R: rotation matrix
    Nord: number of spherical harmonics coefficients to compute
    real: True if only real coefficients are computed
    """
    # check input
    assert len(Rxyz.shape) == 2 and Rxyz.shape[0] == Rxyz.shape[1]
    assert Rxyz.shape[0] == 3
    assert type(Nord) is int and Nord > 0

    # allocate total rotation matrix
    RotationMatrix = np.zeros(((Nord + 1) ** 2, (Nord + 1) ** 2))
    # zeroth-band (l=0) is invariant to rotation
    RotationMatrix[0, 0] = 1.0
    # The first band (l=1) is directly related to the rotation matrix
    R_1 = np.zeros((3, 3))
    R_1[-2 + 2, -2 + 2] = Rxyz[1, 1]
    R_1[-2 + 2, -1 + 2] = Rxyz[1, 2]
    R_1[-2 + 2, 0 + 2] = Rxyz[1, 0]
    R_1[-1 + 2, -2 + 2] = Rxyz[2, 1]
    R_1[-1 + 2, -1 + 2] = Rxyz[2, 2]
    R_1[-1 + 2, 0 + 2] = Rxyz[2, 0]
    R_1[0 + 2, -2 + 2] = Rxyz[0, 1]
    R_1[0 + 2, -1 + 2] = Rxyz[0, 2]
    R_1[0 + 2, 0 + 2] = Rxyz[0, 0]
    RotationMatrix[1:4, 1:4] = R_1
    R_lm1 = R_1

    # compute rotation matrix of each subsequent band recursively
    band_idx = 4
    for l in range(2, Nord + 1):
        R_l = np.zeros((2 * l + 1, 2 * l + 1))
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                # compute u,v,w terms of Eq.8.1 (Table I)
                d = m == 0  # the delta function d_m0
                if abs(n) == l:
                    denom = (2 * l) * (2 * l - 1)
                else:
                    denom = l * l - n * n
                u = np.sqrt((l * l - m * m) / denom)
                v = (
                    np.sqrt((1 + d) * (l + abs(m) - 1) * (l + abs(m)) / denom)
                    * (1 - 2 * d)
                    * 0.5
                )
                w = np.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d) * (-0.5)

                # computes Eq.8.1
                if u != 0:
                    u = u * fun_U(l, m, n, R_1, R_lm1)
                if v != 0:
                    v = v * fun_V(l, m, n, R_1, R_lm1)
                if w != 0:
                    w = w * fun_Wf(l, m, n, R_1, R_lm1)
                R_l[m + l, n + l] = u + v + w
        RotationMatrix[
            band_idx : band_idx + 2 * l + 1, band_idx : band_idx + 2 * l + 1
        ] = R_l
        R_lm1 = R_l
        band_idx = band_idx + 2 * l + 1

    # if the rotation matrix is needed for complex SH, then get it from the one
    # for real SH by the real-to-complex-transformation matrices
    if basisType == "complex":
       w = complex2realSHMtx(Nord)
       RotationMatrix = w.T @ RotationMatrix @ np.conj(w)
    return RotationMatrix


def fun_U(l, m, n, R_1, R_lm1):
    ret = fun_P(0, l, m, n, R_1, R_lm1)
    return ret


def fun_V(l, m, n, R_1, R_lm1):
    if m == 0:
        p0 = fun_P(1, l, 1, n, R_1, R_lm1)
        p1 = fun_P(-1, l, -1, n, R_1, R_lm1)
        ret = p1 + p0
    else:
        if m > 0:
            d = m == 1
            p0 = fun_P(1, l, m - 1, n, R_1, R_lm1)
            p1 = fun_P(-1, l, -m + 1, n, R_1, R_lm1)
            ret = p0 * np.sqrt(1 + d) - p1 * (1 - d)
        else:
            d = m == -1
            p0 = fun_P(1, l, m + 1, n, R_1, R_lm1)
            p1 = fun_P(-1, l, -m - 1, n, R_1, R_lm1)
            ret = p0 * (1 - d) + p1 * np.sqrt(1 + d)
    return ret


# function to compute term P of U,V,W (Table II)
def fun_P(i, l, a, b, R_1, R_lm1):
    ri1 = R_1[i + 1, 2]
    rim1 = R_1[i + 1, 0]
    ri0 = R_1[i + 1, 1]
    if b == -l:
        ret = ri1 * R_lm1[a + l - 1, 0] + rim1 * R_lm1[a + l - 1, 2 * l - 2]
    else:
        if b == l:
            ret = ri1 * R_lm1[a + l - 1, 2 * l - 2] - rim1 * R_lm1[a + l - 1, 0]
        else:
            ret = ri0 * R_lm1[a + l - 1, b + l - 1]
    return ret


def fun_Wf(l, m, n, R_1, R_lm1):
    if m == 0:
        raise ValueError("should not be called for m=0")
    else:
        if m > 0:
            p0 = fun_P(1, l, m + 1, n, R_1, R_lm1)
            p1 = fun_P(-1, l, -m - 1, n, R_1, R_lm1)
            ret = p0 + p1
        else:
            p0 = fun_P(1, l, m - 1, n, R_1, R_lm1)
            p1 = fun_P(-1, l, -m + 1, n, R_1, R_lm1)
            ret = p0 - p1
    return ret


def complex2realSHMtx(Norder):
    # transformation matrix of complex to rel SH
    nch = (Norder + 1) ** 2
    T_c2r = np.zeros((nch,nch), dtype=np.complex64) # maximum order
    T_c2r[0,0] = 1.0
    idx = 1
    if Norder > 0:
        for n in range(1, Norder+1):
            m = np.arange(1, n+1).reshape(-1,1)
            # form the diagonal
            diagT = np.vstack((1j*np.ones((n,1)), np.sqrt(2.0)/2.0, -1.0**m)) / np.sqrt(2.0)
            # form the antidiagonal
            adiagT = np.vstack((-1j*(-1)**np.flipud(m), np.sqrt(2.0)/2.0, np.ones((n,1)))) / np.sqrt(2.0)
            # form the transformation matrix for the specific band n
            tempT = np.diagflat(diagT) + np.fliplr(np.diagflat(adiagT))
            T_c2r[idx: idx + 2*n+1, idx: idx + 2*n+1] = tempT
            idx = idx + 2*n+1
    return T_c2r



if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    # get rotation matrix
    Nord = 4
    yaw, pitch, roll = 60, 0, 0
    Rzyx = euler2rotationMatrix(
        -yaw * np.pi / 180.0, -pitch * np.pi / 180.0, roll * np.pi / 180.0, "zyx"
    )
    print(Rzyx)
    #
    # Rzyx = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    # print(Rzyx.as_matrix())

    # compute rotation matrix in the SH domain
    Rshd = getSHrotMtx(Rzyx, Nord, basisType="complex")
    print(Rshd)

    Rshd = getSHrotMtx(Rzyx, Nord, basisType="real")
    print(Rshd)
