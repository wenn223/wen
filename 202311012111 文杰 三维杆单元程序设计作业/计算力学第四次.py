"""
三维杆单元（空间桁架单元）刚度矩阵与应力计算
作业：2-2 单元刚度方程
作者：根据课程要求实现
"""

import numpy as np


def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元的长度、方向余弦和全局刚度矩阵
    参数:
        x1, x2: 两个节点的坐标，list 或 array [x, y, z]
        E: 弹性模量 (Pa)
        A: 横截面积 (m^2)
    返回:
        L: 单元长度 (m)
        dc: 方向余弦 (cx, cy, cz)
        Ke: 6x6 全局刚度矩阵 (N/m)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L < 1e-12:
        raise ValueError("错误：两个节点重合，单元长度为零，无法计算刚度矩阵。")
    cx, cy, cz = delta / L  # 方向余弦

    # 计算 3x3 方向矩阵 C
    C = np.array([[cx * cx, cx * cy, cx * cz],
                  [cy * cx, cy * cy, cy * cz],
                  [cz * cx, cz * cy, cz * cz]])
    k0 = E * A / L
    # 组装 6x6 刚度矩阵: [ C  -C; -C   C ]
    Ke = k0 * np.block([[C, -C],
                        [-C, C]])
    return L, (cx, cy, cz), Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算单元的应变、应力和轴力
    参数:
        x1, x2: 节点坐标
        E, A: 材料属性
        de: 单元节点位移列阵 [u1, v1, w1, u2, v2, w2] (m)
    返回:
        epsilon: 轴向应变 (无量纲)
        sigma: 轴向应力 (Pa)
        N: 轴力 (N)，拉力为正
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    de = np.asarray(de, dtype=float)
    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L < 1e-12:
        raise ValueError("错误：单元长度为零，无法计算应变。")
    cx, cy, cz = delta / L
    # 应变 = B * de, B = [-cx -cy -cz cx cy cz] / L
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L
    epsilon = np.dot(B, de)
    sigma = E * epsilon
    N = sigma * A
    return epsilon, sigma, N


# ======================= 验证算例 =======================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 60)
    print("算例1：沿 x 轴的一维杆单元")
    print("=" * 60)
    x1 = [0, 0, 0]
    x2 = [2, 0, 0]
    E = 200e9  # 200 GPa
    A = 1.0e-4  # 0.0001 m^2
    de = np.array([0, 0, 0, 1.0e-3, 0, 0])  # 节点2沿x方向位移1mm

    L, dc, Ke = truss3d_element_stiffness(x1, x2, E, A)
    print(f"单元长度 L = {L:.4f} m")
    print(f"方向余弦 (cx, cy, cz) = ({dc[0]:.0f}, {dc[1]:.0f}, {dc[2]:.0f})")
    print("全局刚度矩阵 Ke (6x6):")
    print(Ke)
    # 应力计算
    eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)
    print(f"轴向应变 epsilon = {eps:.4e}")
    print(f"轴向应力 sigma = {sig / 1e6:.1f} MPa")
    print(f"轴力 N = {N:.2f} N")
    print("\n验证：应变 = 0.001/2 = 5e-4，应力 = 200e9*5e-4 = 100 MPa，轴力 = 100e6 * 1e-4 = 1e4 N ✓\n")

    print("=" * 60)
    print("算例2：空间任意方向杆单元")
    print("=" * 60)
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9  # 210 GPa
    A = 2.0e-4  # 2e-4 m^2
    de = np.array([0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3])  # 节点2位移沿(1,2,2)方向

    L, dc, Ke = truss3d_element_stiffness(x1, x2, E, A)
    print(f"单元长度 L = {L:.4f} m")
    print(f"方向余弦 (cx, cy, cz) = ({dc[0]:.4f}, {dc[1]:.4f}, {dc[2]:.4f})")
    print("刚度矩阵 Ke (6x6):")
    print(Ke)
    # 对称性检查
    if np.allclose(Ke, Ke.T):
        print("✓ 刚度矩阵对称")
    else:
        print("✗ 刚度矩阵不对称")
    # 特征值与奇异性
    eigvals = np.linalg.eigvalsh(Ke)
    print(f"刚度矩阵特征值: {eigvals}")
    zero_eig = np.sum(np.abs(eigvals) < 1e-8)
    print(f"零特征值个数: {zero_eig} (由于刚体位移，单个单元刚度矩阵秩为1，应有5个零特征值)")
    # 刚体平移检验
    rigid_de = np.ones(6)  # 所有节点同向位移1m
    force_rigid = Ke @ rigid_de
    print(f"刚体平移 (de = [1,1,1,1,1,1]) 产生的节点力: {force_rigid}")
    print("理论上应为零向量，实际最大值: {:.2e}".format(np.max(np.abs(force_rigid))))
    # 应力计算
    eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)
    print(f"轴向应变 epsilon = {eps:.4e}")
    print(f"轴向应力 sigma = {sig / 1e6:.1f} MPa")
    print(f"轴力 N = {N:.2f} N")
    # 理论值：伸长量 = (1*1/3 + 2*2/3 + 2*2/3)e-3 = 3e-3 m，应变=3e-3/3=1e-3，应力=210e9*1e-3=210 MPa，轴力=210e6*2e-4=42000 N
    print("\n验证：伸长 = (1*1/3+2*2/3+2*2/3)e-3 = 3e-3 m，应变 = 1e-3，应力 = 210 MPa，轴力 = 42000 N ✓")

    print("\n" + "=" * 60)
    print("任务4：刚度矩阵物理意义验证")
    print("=" * 60)
    # 选取第4个自由度（节点2的x方向位移）
    j = 3  # 0-index: 自由度编号0~5, 第4个为索引3
    de_unit = np.zeros(6)
    de_unit[j] = 1.0
    Fe = Ke @ de_unit
    print(f"令第{j + 1}个自由度位移 = 1，其他 = 0，得到节点力向量 Fe = Ke * e_{j + 1}:")
    print(Fe)
    print("该向量恰好等于刚度矩阵的第{}列:".format(j + 1))
    print(Ke[:, j])
    print("k_ij 的物理意义：当第 j 个自由度产生单位位移时，在第 i 个自由度上需要施加的节点力。")
    print("这与课件『单元刚度矩阵的物理意义』一致。")
    import numpy as np


    def truss3d_element_stiffness(x1, x2, E, A):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        delta = x2 - x1
        L = np.linalg.norm(delta)
        if L < 1e-12:
            raise ValueError("单元长度为零，无法计算刚度矩阵。")
        c = delta / L
        C = np.outer(c, c)
        Ke = (E * A / L) * np.block([[C, -C], [-C, C]])
        return L, tuple(c), Ke


    def truss3d_element_stress(x1, x2, E, A, de):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        de = np.asarray(de, dtype=float)
        delta = x2 - x1
        L = np.linalg.norm(delta)
        if L < 1e-12:
            raise ValueError("单元长度为零。")
        c = delta / L
        eps = ((de[3:] - de[:3]) @ c) / L
        sigma = E * eps
        N = sigma * A
        return eps, sigma, N