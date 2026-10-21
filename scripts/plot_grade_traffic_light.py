"""
GRADE 证据质量总结图 · 12 篇尿路上皮癌免疫治疗文献
交通信号灯风格 · 红绿黄三色编码
300 DPI · 适合学术发表

数据设计:
  - RCT 优先(7 篇) + 单臂试验(5 篇) 分组排序
  - 5 个 GRADE 维度
  - 评级:Low / Some concerns / High
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---- 中文字体 ----
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'PingFang SC', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据:12 篇文献 × 5 个 GRADE 维度
# ============================================================
papers = [
    # ---------- RCT (n = 7) — 排序在前 ----------
    {'name': 'KEYNOTE-045\n(Pembrolizumab)',
     'ratings': ['Low', 'Low',     'Low',  'Low',    'Low']},
    {'name': 'KEYNOTE-052\n(Pembrolizumab)',
     'ratings': ['Low', 'Low',     'Low',  'Some',   'Low']},
    {'name': 'CheckMate-275\n(Nivolumab)',
     'ratings': ['Low', 'Some',    'Some', 'Some',   'Low']},
    {'name': 'IMvigor-211\n(Atezolizumab)',
     'ratings': ['Low', 'Some',    'Low',  'Some',   'Low']},
    {'name': 'JAVELIN-100\n(Avelumab)',
     'ratings': ['Low', 'Low',     'Low',  'Low',    'Low']},
    {'name': 'EV-301\n(Enfortumab Vedotin)',
     'ratings': ['Low', 'Low',     'Low',  'Low',    'Low']},
    {'name': 'EV-302\n(EV + Pembrolizumab)',
     'ratings': ['Low', 'Low',     'Low',  'Low',    'Low']},
    # ---------- 单臂试验 (n = 5) — 排序在后 ----------
    {'name': 'TROPHY-U-01\n(Sacituzumab Govitecan)',
     'ratings': ['High', 'High',   'Some', 'High',   'Some']},
    {'name': 'THOR-2\n(Erdafitinib)',
     'ratings': ['High', 'Some',   'Low',  'Some',   'Some']},
    {'name': 'THOR-3\n(Erdafitinib)',
     'ratings': ['High', 'Some',   'Low',  'Some',   'Some']},
    {'name': 'Real-world Cohort\n(2023, multi-center)',
     'ratings': ['High', 'High',   'Low',  'High',   'Some']},
    {'name': 'Case Series\n(Nadofaragene)',
     'ratings': ['High', 'High',   'Some', 'High',   'High']},
]

domains = [
    '研究局限性\nRoB',
    '不一致性\nInconsistency',
    '间接性\nIndirectness',
    '精确性\nImprecision',
    '发表偏倚\nPublication Bias',
]

# 数值编码
rating_map = {'Low': 0, 'Some': 1, 'High': 2}
matrix = np.array([[rating_map[r] for r in p['ratings']] for p in papers])

# ============================================================
# 自定义交通信号灯配色
# ============================================================
colors = [
    (0.00, '#1baf7a'),   # Low → 绿色
    (0.50, '#fff5b8'),   # Some concerns → 浅黄
    (1.00, '#e34948'),   # High → 红色
]
cmap = LinearSegmentedColormap.from_list('traffic_light',
                                         [c[1] for c in colors], N=256)

# ============================================================
# 画布
# ============================================================
fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=300)
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')

sns.heatmap(
    matrix,
    cmap=cmap, vmin=0, vmax=2,
    cbar=False,
    linewidths=2.5, linecolor='#fcfcfb',
    xticklabels=domains,
    yticklabels=[p['name'] for p in papers],
    annot=False,
    ax=ax,
    square=False,
)

# ============================================================
# 单元格文字标注(每格内显示评级)
# ============================================================
display_map = {'Low': 'Low', 'Some': 'Some\nconcerns', 'High': 'High'}
for i in range(len(papers)):
    for j in range(len(domains)):
        r = papers[i]['ratings'][j]
        # 在深色背景(绿/红)用白字,在浅黄背景用深色字
        text_color = 'white' if r in ('Low', 'High') else '#0b0b0b'
        ax.text(j + 0.5, i + 0.5, display_map[r],
                ha='center', va='center',
                color=text_color, fontsize=10.5, fontweight='bold',
                linespacing=1.0)

# ============================================================
# 分组分隔线(RCT 与单臂之间)
# ============================================================
n_rct = 7
ax.axhline(y=n_rct, color='#0b0b0b', linewidth=2.2, linestyle='-')
# 在分隔线右侧加一个组标识标签
ax.text(len(domains) + 0.05, (n_rct - 0.5),
        '↑ 单臂试验边界',
        fontsize=8.5, color='#52514e', va='center')

# ============================================================
# 左侧分组标识(在 y 轴外)
# ============================================================
# RCT 标签
ax.text(-0.18, (n_rct - 1) / 2, 'RCT\n(n = 7)',
        transform=ax.get_yaxis_transform(),
        rotation=90, ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2E86AB')
# 单臂标签
ax.text(-0.18, n_rct + (len(papers) - n_rct) / 2, '单臂\n(n = 5)',
        transform=ax.get_yaxis_transform(),
        rotation=90, ha='center', va='center',
        fontsize=11, fontweight='bold', color='#A23B72')

# ============================================================
# 轴样式
# ============================================================
ax.tick_params(axis='x', labelsize=10.5, labelrotation=0,
               colors='#0b0b0b', pad=4)
ax.tick_params(axis='y', labelsize=10, colors='#0b0b0b',
               labelrotation=0, pad=4)
ax.set_xlabel('')
ax.set_ylabel('')

# ============================================================
# 标题 + 副标题
# ============================================================
fig.suptitle('GRADE 证据质量总结图:12 篇尿路上皮癌免疫治疗文献',
             fontsize=16, fontweight='bold', color='#0b0b0b',
             y=0.985)
fig.text(0.5, 0.94,
         'Traffic-Light Plot · 5 GRADE Domains · RCT 优先 + 单臂试验分组',
         ha='center', fontsize=10.5, color='#52514e', style='italic')

# ============================================================
# 图例(右侧色块)
# ============================================================
legend_elements = [
    mpatches.Patch(facecolor='#1baf7a',
                   edgecolor='white', linewidth=1,
                   label='Low risk(低风险 · 绿色)'),
    mpatches.Patch(facecolor='#fff5b8',
                   edgecolor='#c98500', linewidth=1,
                   label='Some concerns(中度关注 · 黄色)'),
    mpatches.Patch(facecolor='#e34948',
                   edgecolor='white', linewidth=1,
                   label='High risk(高风险 · 红色)'),
]
legend = ax.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    frameon=True, fontsize=10.5,
    edgecolor='#e1e0d9', facecolor='#fcfcfb',
    title='GRADE 评级图例',
    title_fontsize=11,
)
legend.get_frame().set_linewidth(0.8)
legend.get_title().set_fontweight('bold')

# ============================================================
# 右下角补充摘要
# ============================================================
n_low  = int((matrix == 0).sum())
n_some = int((matrix == 1).sum())
n_high = int((matrix == 2).sum())
total  = matrix.size

# 用色块标记代替 emoji(确保字体兼容)
fig.text(1.02, 0.55,
         f'单元格汇总 (n = {total})\n\n'
         f'■ Low = {n_low}  ({n_low/total*100:.1f}%)\n'
         f'■ Some = {n_some}  ({n_some/total*100:.1f}%)\n'
         f'■ High = {n_high}  ({n_high/total*100:.1f}%)',
         fontsize=10, color='#52514e',
         bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f7',
                   ec='#e1e0d9', lw=0.8))

# ============================================================
# 底部脚注
# ============================================================
fig.text(0.5, 0.01,
         '评级依据 GRADE Handbook · RCT 默认从 High certainty 起步,'
         '单臂试验因缺乏对照默认从 Low certainty 起步',
         ha='center', fontsize=9, color='#898781', style='italic')

plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.06, left=0.13, right=0.78)

out_path = r'D:\OncoImmunoRAG\result\grade_summary_plot.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='#fcfcfb')
print(f'[OK] Saved (300 DPI): {out_path}')
print(f'     Low: {n_low}, Some: {n_some}, High: {n_high} (total {total})')
plt.show()
