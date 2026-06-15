"""
FEM for Truss Structures
实现一维杆单元和二维桁架单元的直接刚度法
支持 JSON 输入，输出位移、反力、单元应力/轴力
"""

import json
import numpy as np
import sys
import os

# ---------------------------- 模型数据 ----------------------------
class Model:
    def __init__(self, data):
        self.title = data.get("Title", "")
        self.nsd = data["nsd"]          # 空间维数 1 or 2
        self.ndof = data["ndof"]        # 每个节点自由度数 (1 or 2)
        self.nnp = data["nnp"]          # 节点总数
        self.nel = data["nel"]          # 单元总数
        self.nen = data["nen"]          # 每个单元节点数 (固定为2)
        self.E = np.array(data["E"])    # 弹性模量 (nel,)
        self.CArea = np.array(data["CArea"])  # 截面积 (nel,)
        self.x = np.array(data["x"])    # 节点 x 坐标 (nnp,)
        self.y = np.array(data["y"]) if self.nsd >= 2 else np.zeros(self.nnp)
        self.IEN = np.array(data["IEN"]) - 1   # 转为0基索引 (nel, nen)

        # 边界条件
        fixed_dof = np.array(data["fixed_dof"]) - 1   # 0基自由度编号
        fixed_val = np.array(data["fixed_value"])
        self.fixed_dof = fixed_dof
        self.fixed_val = fixed_val
        force_dof = np.array(data["force_dof"]) - 1
        force_val = np.array(data["force_value"])
        self.force_dof = force_dof
        self.force_val = force_val

        # 总体自由度数
        self.neq = self.nnp * self.ndof
        # 初始化总体刚度矩阵 K, 总体力向量 f
        self.K = np.zeros((self.neq, self.neq))
        self.f = np.zeros(self.neq)

        # 施加节点力
        for dof, val in zip(self.force_dof, self.force_val):
            self.f[dof] = val

        # 将计算得到的单元长度、方向余弦存储到数组
        self.leng = np.zeros(self.nel)
        self.c = np.zeros(self.nel)   # cos(theta)
        self.s = np.zeros(self.nel)   # sin(theta)

        # 生成对号矩阵 LM (nen*ndof, nel)
        self.LM = self.set_LM()

    def set_LM(self):
        """生成对号矩阵 LM"""
        LM = np.zeros((self.nen * self.ndof, self.nel), dtype=int)
        for e in range(self.nel):
            for j in range(self.nen):
                node = self.IEN[e, j]    # 全局节点编号 (0基)
                for m in range(self.ndof):
                    ind = j * self.ndof + m
                    LM[ind, e] = node * self.ndof + m
        return LM

    def compute_element_geometry(self, e):
        """计算单元长度和方向余弦"""
        node1 = self.IEN[e, 0]
        node2 = self.IEN[e, 1]
        dx = self.x[node2] - self.x[node1]
        dy = self.y[node2] - self.y[node1]
        length = np.sqrt(dx*dx + dy*dy)
        self.leng[e] = length
        if self.ndof == 1:
            self.c[e] = 1.0
            self.s[e] = 0.0
        else:
            self.c[e] = dx / length
            self.s[e] = dy / length
        return length, self.c[e], self.s[e]

# ---------------------------- 单元刚度矩阵 ----------------------------
def element_stiffness(model, e):
    """返回单元刚度矩阵 (nen*ndof, nen*ndof)"""
    length, c, s = model.compute_element_geometry(e)
    const = model.CArea[e] * model.E[e] / length
    if model.ndof == 1:
        ke = const * np.array([[1, -1],
                               [-1, 1]])
    else:  # ndof == 2
        cc = c * c
        ss = s * s
        cs = c * s
        ke = const * np.array([[ cc,  cs, -cc, -cs],
                               [ cs,  ss, -cs, -ss],
                               [-cc, -cs,  cc,  cs],
                               [-cs, -ss,  cs,  ss]])
    return ke

# ---------------------------- 直接组装 ----------------------------
def assembly(model):
    """组装总体刚度矩阵 K"""
    for e in range(model.nel):
        ke = element_stiffness(model, e)
        lm_col = model.LM[:, e]   # 单元自由度全局编号
        for i, row in enumerate(lm_col):
            for j, col in enumerate(lm_col):
                model.K[row, col] += ke[i, j]

# ---------------------------- 边界条件处理（缩减法）--------------------
def solve_displacements_reactions(model):
    """
    缩减法求解位移和反力
    返回: d (全局位移), r (反力向量，仅在固定自由度上有值)
    """
    # 已知位移自由度 (已排序)
    d_E_dofs = np.sort(model.fixed_dof)
    d_E_vals = model.fixed_val
    # 未知位移自由度
    all_dofs = np.arange(model.neq)
    d_F_dofs = np.setdiff1d(all_dofs, d_E_dofs)

    # 分块矩阵
    K_EE = model.K[np.ix_(d_E_dofs, d_E_dofs)]
    K_FF = model.K[np.ix_(d_F_dofs, d_F_dofs)]
    K_EF = model.K[np.ix_(d_E_dofs, d_F_dofs)]
    f_F = model.f[d_F_dofs]

    # 求解 d_F
    rhs = f_F - K_EF.T @ d_E_vals
    d_F = np.linalg.solve(K_FF, rhs)

    # 重构完整位移向量
    d = np.zeros(model.neq)
    d[d_E_dofs] = d_E_vals
    d[d_F_dofs] = d_F

    # 计算反力
    r = np.zeros(model.neq)
    r[d_E_dofs] = K_EE @ d_E_vals + K_EF @ d_F

    return d, r

# ---------------------------- 后处理（单元应力/轴力）-------------------
def postprocess(model, d):
    """计算每个单元的应力 (sigma) 和轴力 (N)"""
    results = []
    for e in range(model.nel):
        # 提取单元位移
        lm_col = model.LM[:, e]
        de = d[lm_col]   # (nen*ndof,)

        length, c, s = model.compute_element_geometry(e)
        const = model.E[e] / length   # 应力系数
        if model.ndof == 1:
            sigma = const * np.array([-1, 1]) @ de
        else:
            # 应力 = E/L * [-c, -s, c, s] * de
            B = np.array([-c, -s, c, s])   # (4,)
            sigma = const * (B @ de)
        # 轴力 = 应力 * 截面积
        N = sigma * model.CArea[e]
        results.append({
            "element": e+1,
            "length": length,
            "c": c,
            "s": s,
            "stress": sigma,
            "axial_force": N
        })
    return results

# ---------------------------- 求解单个模型 ----------------------------
def run_single(json_file):
    """读入 JSON 文件并完成求解，输出中文结果"""
    print(f"\n{'='*60}")
    print(f"正在处理文件: {json_file}")
    print('='*60)
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model = Model(data)

    print(f"=== {model.title} ===")
    print(f"问题维数: {model.nsd}D, 每节点自由度数: {model.ndof}")
    print(f"节点总数: {model.nnp}, 单元总数: {model.nel}")

    # 组装
    assembly(model)

    # 检查奇异：施加边界条件前，行列式近似为0
    det_before = np.linalg.det(model.K)
    print(f"\n施加边界条件前 K 的行列式: {det_before:.3e} (奇异: {abs(det_before) < 1e-10})")

    # 求解位移和反力
    d, r = solve_displacements_reactions(model)

    # 检查缩减后的K_FF是否非奇异
    fixed_dofs = np.sort(model.fixed_dof)
    free_dofs = np.setdiff1d(np.arange(model.neq), fixed_dofs)
    K_FF = model.K[np.ix_(free_dofs, free_dofs)]
    det_after = np.linalg.det(K_FF)
    print(f"施加边界条件后 K_FF 的行列式: {det_after:.3e} (非奇异: {abs(det_after) > 1e-10})")

    # 后处理
    elem_results = postprocess(model, d)

    # 输出结果
    print("\n--- 节点位移 ---")
    for i in range(model.nnp):
        if model.ndof == 1:
            print(f"节点 {i+1}: d = {d[i]:.6f}")
        else:
            u = d[2*i]; v = d[2*i+1]
            print(f"节点 {i+1}: u = {u:.6f}, v = {v:.6f}")

    print("\n--- 固定自由度处的反力 ---")
    for dof, val in zip(model.fixed_dof, r[model.fixed_dof]):
        node = dof // model.ndof + 1
        local = dof % model.ndof
        comp = "u" if local == 0 else "v"
        print(f"自由度 {dof+1} (节点 {node}, {comp}): R = {val:.6f}")

    print("\n--- 单元计算结果 ---")
    for res in elem_results:
        print(f"单元 {res['element']}:")
        print(f"  长度 = {res['length']:.6f}")
        if model.ndof == 2:
            print(f"  cosθ = {res['c']:.6f}, sinθ = {res['s']:.6f}")
        print(f"  应力 = {res['stress']:.6f}")
        print(f"  轴力 = {res['axial_force']:.6f}")

    # 输出总体刚度矩阵（可选，用于验证）
    print("\n--- 总体刚度矩阵 K (较大时只显示前6×6) ---")
    np.set_printoptions(precision=4, suppress=True)
    if model.neq <= 6:
        print(model.K)
    else:
        print(model.K[:6, :6], "... (已截断)")
    print("求解完成。\n")

# ---------------------------- 主程序 ----------------------------
if __name__ == "__main__":
    if len(sys.argv) == 2:
        # 命令行指定单个文件
        run_single(sys.argv[1])
    else:
        # 无参数时自动处理 算例1.json 和 算例2.json
        files = ["算例1.json", "算例2.json"]
        for fname in files:
            if os.path.exists(fname):
                run_single(fname)
            else:
                print(f"警告: 文件 '{fname}' 未找到，已跳过。")