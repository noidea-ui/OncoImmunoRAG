"""
系统证据提取与分级准确率迭代优化历程
组合图:分组柱状图(准确率)+ 折线图(幻觉率,带圆圈),双 Y 轴
300 DPI 输出
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---- 中文字体 ----
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'PingFang SC', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ---- 数据 ----
stages = ['Batch-1\n(基准)',
          'Batch-2\n(规则链路调优)',
          'Batch-3\n(目标文献路由+注册表)']
study_design  = [75.00, 83.33, 100.00]   # 研究设计准确率
grade         = [58.33, 66.67, 100.00]   # GRADE 分级准确率
hallucination = [25.00, 16.67,   8.33]   # 数值幻觉率(右轴)

# ---- 用户指定配色 ----
COLOR_STUDY        = '#2E86AB'   # 蓝
COLOR_GRADE        = '#A23B72'   # 紫红
COLOR_HALLUCIN     = '#F18F01'   # 橙

# ---- 布局 ----
x = np.arange(len(stages))
width = 0.32

fig, ax1 = plt.subplots(figsize=(12, 7.5), dpi=300)
fig.patch.set_facecolor('#fcfcfb')
ax1.set_facecolor('#fcfcfb')

# ---- 左轴:准确率柱状图 ----
bars1 = ax1.bar(x - width/2, study_design, width,
                label='研究设计准确率', color=COLOR_STUDY,
                edgecolor='white', linewidth=0.6, zorder=3)
bars2 = ax1.bar(x + width/2, grade, width,
                label='GRADE 分级准确率', color=COLOR_GRADE,
                edgecolor='white', linewidth=0.6, zorder=3)

ax1.set_ylabel('准确率 Accuracy (%)', fontsize=12.5,
               color='#0b0b0b', labelpad=10, fontweight='600')
ax1.set_ylim(0, 115)
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=11, color='#0b0b0b')
ax1.tick_params(axis='y', labelcolor='#0b0b0b', labelsize=10)
ax1.tick_params(axis='x', labelsize=11)

# 柱顶数据标签(加粗)
for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8,
                 f'{h:.2f}%', ha='center', va='bottom',
                 fontsize=10.5, fontweight='bold', color='#0b0b0b')

# ---- 右轴:折线图(幻觉率) ----
ax2 = ax1.twinx()
line, = ax2.plot(x, hallucination,
                 color=COLOR_HALLUCIN, linewidth=2.2,
                 marker='o', markersize=10,
                 markerfacecolor='white',
                 markeredgecolor=COLOR_HALLUCIN,
                 markeredgewidth=2.2,
                 label='数值幻觉率', zorder=4)

ax2.set_ylabel('幻觉率 Hallucination Rate (%)', fontsize=12.5,
               color=COLOR_HALLUCIN, labelpad=10, fontweight='600')
ax2.set_ylim(0, 35)
ax2.tick_params(axis='y', labelcolor=COLOR_HALLUCIN, labelsize=10)

# 折线节点数据标签(每个圆圈旁显示数值)
for xi, yi in zip(x, hallucination):
    ax2.text(xi, yi + 1.5, f'{yi:.2f}%',
             ha='center', va='bottom',
             fontsize=10.5, fontweight='bold',
             color=COLOR_HALLUCIN)

# ---- 网格(仅左轴) ----
ax1.grid(True, linestyle='--', alpha=0.45,
         axis='y', color='#898781', linewidth=0.7, zorder=1)
ax1.set_axisbelow(True)

# ---- 边框细线 ----
for spine in ax1.spines.values():
    spine.set_edgecolor('#c3c2b7'); spine.set_linewidth(0.8)
for spine in ax2.spines.values():
    spine.set_edgecolor('#c3c2b7'); spine.set_linewidth(0.8)

# ---- 标题 ----
plt.title('系统证据提取与分级准确率迭代优化历程',
          fontsize=16, fontweight='bold', color='#0b0b0b', pad=18)

# ---- 合并图例(顶部,水平排列) ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                    loc='upper left', frameon=True, fontsize=10.5,
                    edgecolor='#e1e0d9', facecolor='#fcfcfb',
                    ncol=1, framealpha=0.95)
legend.get_frame().set_linewidth(0.8)

# ---- 脚注 ----
fig.text(0.5, 0.015,
         '左轴 = 准确率(越高越好,柱状图)   ·   右轴 = 数值幻觉率(越低越好,折线图)',
         ha='center', fontsize=9.5, color='#898781', style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# ---- 保存 + 显示 ----
out_path = r'D:\OncoImmunoRAG\result\performance_optimization.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#fcfcfb')
print(f'[OK] Saved (300 DPI): {out_path}')
plt.show()
