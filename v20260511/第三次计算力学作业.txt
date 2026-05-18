import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 全局配置
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 计算π近似值和误差
n = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
pi_approx = n * np.sin(np.pi / n)
error = np.abs(np.pi - pi_approx)
h = 1 / n

# 绘图
plt.figure(figsize=(10, 7))

# 误差曲线（蓝色）
plt.loglog(h, error, 'o-', color='#1f77b4', linewidth=3, markersize=9, label=r'Numerical Error $e_n$')

# 参考线：只改颜色，其他逻辑不变
h_ref = np.array([1e-3, 1])
# 二阶参考线：黑色虚线
plt.loglog(h_ref, 10*h_ref**2, 'k--', linewidth=2, label=r'Slope=2')
# 四阶参考线：红色虚线（和二阶线区分开）
plt.loglog(h_ref, 100*h_ref**4, 'r--', linewidth=2, label=r'Slope=4')

# 标注课件里的数值
labels = [
    (h[1], error[1], '1.46'), (h[2], error[2], '1.87'),
    (h[3], error[3], '1.97'), (h[4], error[4], '1.99'),
    (h[5], error[5], '2.00'), (h[6], error[6], '2.00'),
    (h[7], error[7], '2.00'), (h[8], error[8], '2.00'),
    (h[3], error[3]/100, '5.31'), (h[4], error[4]/100, '7.50'),
    (h[5], error[5]/100, '9.76')
]
for x, y, text in labels:
    plt.annotate(text, (x, y), xytext=(5,5), textcoords='offset points')

# 图表设置
plt.xlabel(r'Element Size $h=1/n$', fontsize=13)
plt.ylabel(r'Error $e_n=|\pi-\pi_n|$', fontsize=13)
plt.title(r'Finite Element Convergence Analysis', fontsize=15, pad=15)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper left')
plt.xlim(1e-3, 1)
plt.ylim(1e-10, 1e3)

# 输出数值表格
print("n\tπ Approximation\t\tError")
print("-"*50)
for i in range(len(n)):
    print(f"{n[i]}\t{pi_approx[i]:.15f}\t{error[i]:.15f}")

plt.tight_layout()
plt.show()