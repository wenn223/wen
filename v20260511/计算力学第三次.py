import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'font.size': 12,
    'figure.figsize': (10, 7),
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
})

# ==================== 核心计算 ====================
n = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
pi_approx = n * np.sin(np.pi / n)
error = np.abs(np.pi - pi_approx)
h = 1 / n

# 局部收敛速率
slope = np.log(error[1:] / error[:-1]) / np.log(h[1:] / h[:-1])
h_mid = np.sqrt(h[1:] * h[:-1])
error_mid = np.sqrt(error[1:] * error[:-1])

# ==================== 绘图 ====================
fig, ax = plt.subplots(facecolor='#f7f7f7')   # 图片背景浅灰
ax.set_facecolor('#f7f7f7')                  # 坐标区背景浅灰

# 1. 误差曲线（深蓝色，实心圆标记，加粗线条）
ax.loglog(h, error, 'o-', color='#2c3e50', linewidth=3, markersize=9,
          markerfacecolor='white', markeredgewidth=2,
          label=r'数值误差 $e_n = |\pi - n\sin(\pi/n)|$')

# 2. 二阶参考线
C2 = error[0] / h[0]**2
h_ref = np.array([h[-1]*0.5, h[0]*1.5])
ax.loglog(h_ref, C2 * h_ref**2, '--', color='#e74c3c', linewidth=2.5,
          label=r'$O(h^2)$ 参考线 (斜率 = 2)')

# 3. 四阶参考线
C4 = error[0] / h[0]**4
ax.loglog(h_ref, C4 * h_ref**4, '--', color='#27ae60', linewidth=2.5,
          label=r'$O(h^4)$ 参考线 (斜率 = 4)')

# 4. 局部收敛速率标注
for x, y, s in zip(h_mid, error_mid, slope):
    ax.annotate(f'{s:.2f}', (x, y),
                textcoords='offset points', xytext=(5, 5),
                fontsize=9, color='#2c3e50', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', alpha=0.9))

# 5. 说明文本框（半透明，放在左上角）
ax.text(0.05, 0.95, '曲线上数字：局部斜率\n（估计的收敛阶）',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=10, color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#aaaaaa', alpha=0.8))

# ==================== 图表装饰 ====================
ax.set_xlabel(r'单元尺寸 $h = 1/n$', fontsize=13, color='#333333')
ax.set_ylabel(r'误差 $e_n$', fontsize=13, color='#333333')
ax.set_title(r'$\pi$ 近似 $n\sin(\pi/n)$ 的收敛性分析', fontsize=15, pad=15, color='#222222')
ax.grid(True, which='both', linestyle=':', color='white', linewidth=1.5)
ax.grid(which='minor', linestyle=':', color='white', linewidth=0.8)
ax.tick_params(colors='#555555')
ax.set_xlim(1e-3, 1.2)
ax.set_ylim(1e-10, 1e3)

# 图例样式：半透明圆角框，放在右下角
ax.legend(fontsize=11, loc='lower right', framealpha=0.8,
          edgecolor='#aaaaaa', facecolor='white', fancybox=True)

# ==================== 数值表格 ====================
print(f"{'n':<6} {'π 近似值':<22} {'误差':<22} {'局部斜率'}")
print("-" * 65)
print(f"{n[0]:<6} {pi_approx[0]:<22.15f} {error[0]:<22.15f} {'---'}")
for i in range(1, len(n)):
    print(f"{n[i]:<6} {pi_approx[i]:<22.15f} {error[i]:<22.15f} {slope[i-1]:.2f}")

plt.tight_layout()
plt.show()