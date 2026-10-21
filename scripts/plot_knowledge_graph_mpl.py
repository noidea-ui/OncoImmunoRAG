"""
尿路上皮癌免疫治疗文献知识图谱 · matplotlib 渲染版(v3 终极布局)
分层径向布局:中心 + 3 环
"""
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Circle

# ---- 中文字体 ----
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'PingFang SC', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ---- 配色 ----
C_DISEASE  = '#2C3E50'
C_MODALITY = '#2E86AB'
C_DRUG     = '#A23B72'
C_CHEMO    = '#4A3AA7'
C_BIOMARK  = '#008300'
C_OUTCOME  = '#F18F01'

EDGE_TREAT   = '#7F8C8D'
EDGE_BIOMARK = '#1baf7a'
EDGE_OUTCOME = '#c55181'

# ---- 画布(终极尺寸:17 x 14) ----
W, H = 17, 14
CX, CY = 8.5, 6.85

# ---- 极坐标 ----
def polar(cx, cy, r, deg):
    rad = math.radians(deg)
    return (cx + r * math.sin(rad), cy + r * math.cos(rad))

# ---- 节点定义 ----
center = {'id':'UC','label':'尿路上皮癌','pos':(CX,CY),'r':1.00,'color':C_DISEASE}

modalities = [
    {'id':'M1','label':'PD-1/PD-L1\n抑制剂','color':C_MODALITY},
    {'id':'M2','label':'ADC 药物','color':C_MODALITY},
    {'id':'M3','label':'FGFR 靶向治疗','color':C_MODALITY},
    {'id':'M4','label':'化疗','color':C_MODALITY},
    {'id':'M5','label':'联合治疗','color':C_MODALITY},
    {'id':'M6','label':'免疫微环境','color':C_MODALITY},
]
R_MOD = 2.5
for i, m in enumerate(modalities):
    m['pos'] = polar(CX, CY, R_MOD, i*60)
    m['r']   = 0.80

middle_nodes = [
    {'id':'D1','label':'Pembrolizumab','color':C_DRUG},
    {'id':'D2','label':'Nivolumab','color':C_DRUG},
    {'id':'D3','label':'Atezolizumab','color':C_DRUG},
    {'id':'D4','label':'Avelumab','color':C_DRUG},
    {'id':'D5','label':'Enfortumab\nVedotin','color':C_DRUG},
    {'id':'D6','label':'Sacituzumab\nGovitecan','color':C_DRUG},
    {'id':'D7','label':'Erdafitinib','color':C_DRUG},
    {'id':'D8','label':'Cisplatin /\nGemcitabine','color':C_CHEMO},
    {'id':'B1','label':'PD-L1 表达','color':C_BIOMARK},
    {'id':'B2','label':'TMB\n肿瘤突变负荷','color':C_BIOMARK},
    {'id':'B3','label':'FGFR3 突变','color':C_BIOMARK},
    {'id':'B4','label':'Nectin-4 表达','color':C_BIOMARK},
]
R_MID = 4.6
for i, n in enumerate(middle_nodes):
    n['pos'] = polar(CX, CY, R_MID, -90 + 15 + i*30)
    n['r']   = 0.70

outcome_nodes = [
    {'id':'O1','label':'总生存期\nOS','color':C_OUTCOME},
    {'id':'O2','label':'无进展生存期\nPFS','color':C_OUTCOME},
    {'id':'O3','label':'客观缓解率\nORR','color':C_OUTCOME},
    {'id':'O4','label':'完全缓解\nCR','color':C_OUTCOME},
    {'id':'O5','label':'治疗耐药','color':C_OUTCOME},
    {'id':'O6','label':'安全性\n不良反应','color':C_OUTCOME},
]
R_OUT = 5.5
for i, n in enumerate(outcome_nodes):
    n['pos'] = polar(CX, CY, R_OUT, -90 + 30 + i*60)
    n['r']   = 0.82

by_id = {n['id']: n for n in ([center] + modalities + middle_nodes + outcome_nodes)}

# ---- 边 ----
edges = [
    ('UC','M1','治疗',EDGE_TREAT,'solid'),
    ('UC','M2','治疗',EDGE_TREAT,'solid'),
    ('UC','M3','治疗',EDGE_TREAT,'solid'),
    ('UC','M4','治疗',EDGE_TREAT,'solid'),
    ('UC','M5','治疗',EDGE_TREAT,'solid'),
    ('UC','M6','调控',EDGE_TREAT,'dashed'),
    ('M1','D1','分类',EDGE_TREAT,'solid'),
    ('M1','D2','分类',EDGE_TREAT,'solid'),
    ('M1','D3','分类',EDGE_TREAT,'solid'),
    ('M1','D4','分类',EDGE_TREAT,'solid'),
    ('M2','D5','分类',EDGE_TREAT,'solid'),
    ('M2','D6','分类',EDGE_TREAT,'solid'),
    ('M3','D7','分类',EDGE_TREAT,'solid'),
    ('M4','D8','分类',EDGE_TREAT,'solid'),
    ('M5','D5','联合',EDGE_TREAT,'solid'),
    ('M5','D1','联合',EDGE_TREAT,'solid'),
    ('B1','D1','预测响应',EDGE_BIOMARK,'dashed'),
    ('B1','D3','预测响应',EDGE_BIOMARK,'dashed'),
    ('B2','M1','预测响应',EDGE_BIOMARK,'dashed'),
    ('B3','D7','靶点',EDGE_BIOMARK,'dashed'),
    ('B4','D5','靶点',EDGE_BIOMARK,'dashed'),
    ('D1','O1','延长',EDGE_OUTCOME,'dotted'),
    ('D1','O3','提高',EDGE_OUTCOME,'dotted'),
    ('D2','O1','延长',EDGE_OUTCOME,'dotted'),
    ('D5','O3','提高',EDGE_OUTCOME,'dotted'),
    ('D5','O4','诱导',EDGE_OUTCOME,'dotted'),
    ('D5','D1','联合',EDGE_OUTCOME,'dotted'),
    ('D6','O3','提高',EDGE_OUTCOME,'dotted'),
    ('D7','O3','提高',EDGE_OUTCOME,'dotted'),
    ('D8','O3','提高',EDGE_OUTCOME,'dotted'),
    ('M5','O4','提高',EDGE_OUTCOME,'dotted'),
    ('O5','UC','预后相关',EDGE_OUTCOME,'dotted'),
    ('O5','M6','介导耐药',EDGE_OUTCOME,'dotted'),
    ('O6','M1','irAE',EDGE_OUTCOME,'solid'),
]

# ---- 画布 ----
fig = plt.figure(figsize=(W, H), dpi=140)
fig.patch.set_facecolor('#fcfcfb')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.set_axis_off()

# ---- 标题(顶部安全区) ----
ax.text(8.5, 13.55, '尿路上皮癌免疫治疗文献知识图谱',
        ha='center', fontsize=18, fontweight='bold', color='#0b0b0b')
ax.text(8.5, 13.18, 'Knowledge Graph · Urothelial Carcinoma Immunotherapy · 基于 12 篇核心文献',
        ha='center', fontsize=10, color='#52514e', style='italic')

# ---- 节点绘制 ----
def draw_node(ax, n, font_size=9.5):
    x, y = n['pos']
    r = n['r']
    circle = Circle((x, y), r, facecolor=n['color'],
                    edgecolor='#fcfcfb', linewidth=2.2, zorder=3)
    ax.add_patch(circle)
    lines = n['label'].split('\n')
    if len(lines) == 1:
        ax.text(x, y, lines[0], ha='center', va='center',
                fontsize=font_size, fontweight='600', color='white', zorder=4)
    else:
        line_h = font_size * 0.11  # 数据单位(紧凑行高)
        total = (len(lines) - 1) * line_h
        for i, ln in enumerate(lines):
            ax.text(x, y - total/2 + i*line_h, ln, ha='center', va='center',
                    fontsize=font_size, fontweight='600', color='white', zorder=4)

# ---- 边绘制 ----
def draw_edge(ax, e):
    src, tgt, label, color, style = e
    s = by_id[src]; t = by_id[tgt]
    x1, y1 = s['pos']; x2, y2 = t['pos']
    dx, dy = x2-x1, y2-y1
    d = math.hypot(dx, dy)
    ux, uy = dx/d, dy/d
    sx = x1 + ux*s['r']; sy = y1 + uy*s['r']
    tx = x2 - ux*t['r']; ty = y2 - uy*t['r']
    ls = {'solid':'-','dashed':(0,(5,3)),'dotted':(0,(1.5,3))}[style]
    ax.plot([sx, tx], [sy, ty], color=color, linewidth=1.0, linestyle=ls, alpha=0.65, zorder=2)
    if label:
        mx, my = (sx+tx)/2, (sy+ty)/2
        nx, ny = -uy, ux
        ox, oy = nx*0.16, ny*0.16
        ax.text(mx+ox, my+oy, label, ha='center', va='center',
                fontsize=7.8, fontweight='600', color=color, zorder=3,
                bbox=dict(boxstyle='round,pad=0.18', fc='#fcfcfb',
                          ec=color, lw=0.5, alpha=0.96))

# 边先画(底层)
for e in edges:
    draw_edge(ax, e)

# 节点后画(顶层)
for n in modalities:
    draw_node(ax, n, font_size=9.0)
for n in middle_nodes:
    draw_node(ax, n, font_size=8.2)
for n in outcome_nodes:
    draw_node(ax, n, font_size=8.6)
draw_node(ax, center, font_size=12.5)

# ---- 左上图例(缩小,避开 O1 OS 节点) ----
lx, ly = 0.4, 12.6
ax.add_patch(FancyBboxPatch((lx, ly-2.5), 2.8, 2.5,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              fc='#f9f9f7', ec='#e1e0d9', lw=0.8, zorder=2))
ax.text(lx+0.15, ly-0.10, '图例 · LEGEND', fontsize=10, fontweight='bold', color='#0b0b0b')
ax.plot([lx+0.15, lx+2.65], [ly-0.24, ly-0.24], color='#e1e0d9', lw=0.8)

legend_items = [
    (C_DISEASE,  '疾病(中心)'),
    (C_MODALITY, '治疗模式'),
    (C_DRUG,     '具体药物'),
    (C_CHEMO,    '化疗药物'),
    (C_BIOMARK,  '生物标志物'),
    (C_OUTCOME,  '临床结局'),
]
for i, (c, lbl) in enumerate(legend_items):
    yy = ly - 0.45 - i*0.25
    ax.add_patch(Circle((lx+0.30, yy), 0.10, facecolor=c, ec='#fcfcfb', lw=1.2, zorder=3))
    ax.text(lx+0.50, yy-0.03, lbl, fontsize=9, color='#52514e', va='center')

ax.plot([lx+0.15, lx+2.65], [ly-2.02, ly-2.02], color='#e1e0d9', lw=0.8)
ax.text(lx+0.15, ly-2.16, '边类型', fontsize=9, fontweight='600', color='#898781')
ax.plot([lx+0.20, lx+0.65], [ly-2.32, ly-2.32], color=EDGE_TREAT, lw=1.2)
ax.text(lx+0.72, ly-2.36, '实线 = 治疗/分类', fontsize=8.5, color='#52514e', va='center')
ax.plot([lx+0.20, lx+0.65], [ly-2.42, ly-2.42], color=EDGE_BIOMARK, lw=1.2, linestyle=(0,(4,2)))
ax.text(lx+0.72, ly-2.46, '虚线 = 生物标志物', fontsize=8.5, color='#52514e', va='center')

# ---- 右上角:文献覆盖(避开 O3 ORR 节点) ----
rx, ry = 13.6, 12.6
ax.add_patch(FancyBboxPatch((rx, ry-2.5), 3.2, 2.5,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              fc='#f9f9f7', ec='#e1e0d9', lw=0.8))
ax.text(rx+0.15, ry-0.10, '文献覆盖(节选)', fontsize=10, fontweight='bold', color='#0b0b0b')
ax.plot([rx+0.15, rx+3.05], [ry-0.24, ry-0.24], color='#e1e0d9', lw=0.8)

coverage = [
    ('KEYNOTE-045/052',  'Pembrolizumab'),
    ('CheckMate-275',    'Nivolumab'),
    ('IMvigor-211',      'Atezolizumab'),
    ('JAVELIN-100',      'Avelumab'),
    ('EV-301/302',       'Enfortumab Vedotin'),
    ('TROPHY-U-01',      'Sacituzumab Govitecan'),
    ('THOR-2/3',         'Erdafitinib'),
    ('MVAC / GC',        'Cisplatin/Gemcitabine'),
]
for i, (a, b) in enumerate(coverage):
    yy = ry - 0.48 - i*0.23
    ax.text(rx+0.15, yy, a, fontsize=8.3, color='#0b0b0b', fontweight='600')
    ax.text(rx+1.45, yy, b, fontsize=8.3, color='#52514e')

ax.text(rx+0.15, ry-2.30, '共 12 篇核心文献', fontsize=8.8, color='#0b0b0b', fontweight='600', style='italic')

# ---- 底部说明 ----
ax.text(8.5, 0.45,
        '中心 = 疾病 · 内环 = 治疗模式 · 中环 = 药物 + 生物标志物 · 外环 = 临床结局',
        ha='center', fontsize=9.5, color='#898781', style='italic')

out_path = r'D:\OncoImmunoRAG\result\knowledge_graph.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
print(f'[OK] Knowledge graph saved to: {out_path}')
plt.show()
