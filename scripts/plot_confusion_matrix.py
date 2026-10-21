"""
12 篇文献 GRADE 证据等级预测与金标准一致性热力图
Seaborn · 白→深绿渐变 · 3×3 混淆矩阵
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---- 中文字体 ----
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'PingFang SC', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据(最终 100% 对齐,完美分类)
# 行:AI 预测等级   列:人工金标准
# ============================================================
labels = ['高 (High)', '中 (Moderate)', '极低 (Very Low)']
matrix = np.array([
    [7, 0, 0],   # AI = 高
    [0, 0, 0],   # AI = 中
    [0, 0, 5],   # AI = 极低
], dtype=int)

n_total      = int(matrix.sum())
n_diagonal   = int(np.trace(matrix))
accuracy     = n_diagonal / n_total * 100

# ============================================================
# 自定义 colormap:白 → 深绿
# ============================================================
green_stops = ['#ffffff', '#e8f5e9', '#a5d6a7',
               '#4caf50', '#2e7d32', '#0a7a0a']
cmap = LinearSegmentedColormap.from_list('white_to_dark_green',
                                         green_stops, N=256)

# ============================================================
# 绘图
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')

sns.heatmap(
    matrix,
    annot=True, fmt='d',
    cmap=cmap, vmin=0, vmax=matrix.max(),
    cbar=True, cbar_kws={
        'shrink': 0.7,
        'label': '文献篇数 (count)',
        'orientation': 'vertical',
    },
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={'size': 22, 'weight': 'bold', 'color': '#0b0b0b'},
    linewidths=2.5, linecolor='#fcfcfb',
    square=True,
    ax=ax,
)

# ---- 轴标签 ----
ax.set_xlabel('人工金标准 · Gold Standard',
              fontsize=13, fontweight='600', labelpad=12)
ax.set_ylabel('AI 预测等级 · Predicted',
              fontsize=13, fontweight='600', labelpad=12)

# 刻度文字
ax.tick_params(axis='x', labelsize=11, labelrotation=0)
ax.tick_params(axis='y', labelsize=11, labelrotation=0, labelleft=True)

# ---- 标题 ----
plt.suptitle('12 篇文献 GRADE 证据等级预测与金标准一致性热力图',
             fontsize=15, fontweight='bold', color='#0b0b0b',
             y=0.985)

# ---- 顶部摘要框(对齐率 + 指标)----
metrics_text = (f'总样本 N = {n_total}    ·    '
                f'对角命中 = {n_diagonal}    ·    '
                f'准确率 = {accuracy:.2f}%    ·    '
                f"Cohen's κ = 1.000")
ax.text(0.5, 1.045, metrics_text,
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=10.5, color='#52514e',
        bbox=dict(boxstyle='round,pad=0.4', fc='#f0faf0',
                  ec='#1baf7a', lw=0.8))

# ---- 底部脚注 ----
fig.text(0.5, 0.005,
         '色阶:白色 = 0 篇(无错分)  ·  深绿 = 高密度命中  ·  '
         '对角线以外均为白色 = 零误判',
         ha='center', fontsize=9.5, color='#898781', style='italic')

plt.tight_layout()
plt.subplots_adjust(top=0.86, bottom=0.16)

# ---- 保存 + 显示 ----
out_path = r'D:\OncoImmunoRAG\result\grade_confusion_matrix.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='#fcfcfb')
print(f'[OK] Saved (300 DPI): {out_path}')
plt.show()
