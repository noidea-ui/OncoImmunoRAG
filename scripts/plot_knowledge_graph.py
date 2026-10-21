"""
尿路上皮癌免疫治疗文献知识图谱
基于 12 篇核心文献的典型内容生成的模拟知识图谱

布局:中心节点 + 三层环形结构
"""
import math

# ============================================================
# 配置
# ============================================================
W, H = 1500, 1100
CX, CY = 750, 555

# 用户指定配色(沿用上一张图)
C_DISEASE  = '#2C3E50'   # 深石板灰 — 中心疾病
C_MODALITY = '#2E86AB'   # 蓝 — 治疗模式
C_DRUG     = '#A23B72'   # 紫红 — 药物
C_CHEMO    = '#4A3AA7'   # 紫 — 化疗药物
C_BIOMARK  = '#008300'   # 绿 — 生物标志物
C_OUTCOME  = '#F18F01'   # 橙 — 临床结局

EDGE_TREAT   = '#7F8C8D'   # 治疗关系 — 中灰
EDGE_BIOMARK = '#1baf7a'   # 标志物关系 — 绿
EDGE_OUTCOME = '#c55181'   # 结局关系 — 品红

# ============================================================
# 节点极坐标计算
# ============================================================
def polar(cx, cy, r, deg_from_top):
    """角度从顶端起算,顺时针为正"""
    rad = math.radians(deg_from_top)
    return (cx + r * math.sin(rad), cy - r * math.cos(rad))

# 中心节点
center = {'id': 'UC', 'label': '尿路上皮癌',
          'pos': (CX, CY), 'r': 58, 'color': C_DISEASE, 'cls': 'disease'}

# 内环(治疗模式,6 个,等分 360°)
modalities = [
    {'id': 'M1', 'label': 'PD-1/PD-L1\n抑制剂', 'color': C_MODALITY, 'cls': 'modality'},
    {'id': 'M2', 'label': 'ADC 药物',          'color': C_MODALITY, 'cls': 'modality'},
    {'id': 'M3', 'label': 'FGFR 靶向治疗',     'color': C_MODALITY, 'cls': 'modality'},
    {'id': 'M4', 'label': '化疗',              'color': C_MODALITY, 'cls': 'modality'},
    {'id': 'M5', 'label': '联合治疗',          'color': C_MODALITY, 'cls': 'modality'},
    {'id': 'M6', 'label': '免疫微环境',        'color': C_MODALITY, 'cls': 'modality'},
]
R_MOD = 200
for i, m in enumerate(modalities):
    m['pos'] = polar(CX, CY, R_MOD, i * 60)
    m['r']   = 46

# 中环(药物 + 生物标志物,12 个,起始 -90° + 15° 偏移避免与内环对齐)
middle_nodes = [
    {'id': 'D1', 'label': 'Pembrolizumab',         'color': C_DRUG,    'cls': 'drug'},
    {'id': 'D2', 'label': 'Nivolumab',             'color': C_DRUG,    'cls': 'drug'},
    {'id': 'D3', 'label': 'Atezolizumab',          'color': C_DRUG,    'cls': 'drug'},
    {'id': 'D4', 'label': 'Avelumab',              'color': C_DRUG,    'cls': 'drug'},
    {'id': 'D5', 'label': 'Enfortumab\nVedotin',   'color': C_DRUG,    'cls': 'drug'},
    {'id': 'D6', 'label': 'Sacituzumab\nGovitecan', 'color': C_DRUG,   'cls': 'drug'},
    {'id': 'D7', 'label': 'Erdafitinib',           'color': C_DRUG,    'cls': 'drug'},
    {'id': 'D8', 'label': 'Cisplatin /\nGemcitabine', 'color': C_CHEMO, 'cls': 'chemo'},
    {'id': 'B1', 'label': 'PD-L1 表达',            'color': C_BIOMARK, 'cls': 'biomarker'},
    {'id': 'B2', 'label': 'TMB\n肿瘤突变负荷',     'color': C_BIOMARK, 'cls': 'biomarker'},
    {'id': 'B3', 'label': 'FGFR3 突变',            'color': C_BIOMARK, 'cls': 'biomarker'},
    {'id': 'B4', 'label': 'Nectin-4 表达',         'color': C_BIOMARK, 'cls': 'biomarker'},
]
R_MID = 370
for i, n in enumerate(middle_nodes):
    n['pos'] = polar(CX, CY, R_MID, -90 + 15 + i * 30)
    n['r']   = 36

# 外环(临床结局,6 个)
outcome_nodes = [
    {'id': 'O1', 'label': '总生存期\nOS',     'color': C_OUTCOME},
    {'id': 'O2', 'label': '无进展生存期\nPFS', 'color': C_OUTCOME},
    {'id': 'O3', 'label': '客观缓解率\nORR',   'color': C_OUTCOME},
    {'id': 'O4', 'label': '完全缓解\nCR',      'color': C_OUTCOME},
    {'id': 'O5', 'label': '治疗耐药',          'color': C_OUTCOME},
    {'id': 'O6', 'label': '安全性\n不良反应',  'color': C_OUTCOME},
]
R_OUT = 525
for i, n in enumerate(outcome_nodes):
    n['pos'] = polar(CX, CY, R_OUT, -90 + 30 + i * 60)
    n['r']   = 40

# 索引便于引用
by_id = {n['id']: n for n in ([center] + modalities + middle_nodes + outcome_nodes)}

# ============================================================
# 边(关系)
# ============================================================
# 格式: (源, 目标, 关系标签或 None, 颜色, 样式)
# 样式: solid / dashed / dotted
edges = [
    # --- 中心 → 治疗模式 ---
    ('UC', 'M1', '治疗',            EDGE_TREAT,   'solid'),
    ('UC', 'M2', '治疗',            EDGE_TREAT,   'solid'),
    ('UC', 'M3', '治疗',            EDGE_TREAT,   'solid'),
    ('UC', 'M4', '治疗',            EDGE_TREAT,   'solid'),
    ('UC', 'M5', '治疗',            EDGE_TREAT,   'solid'),
    ('UC', 'M6', '调控',            EDGE_TREAT,   'dashed'),

    # --- 治疗模式 → 具体药物 ---
    ('M1', 'D1', '分类',            EDGE_TREAT,   'solid'),
    ('M1', 'D2', '分类',            EDGE_TREAT,   'solid'),
    ('M1', 'D3', '分类',            EDGE_TREAT,   'solid'),
    ('M1', 'D4', '分类',            EDGE_TREAT,   'solid'),
    ('M2', 'D5', '分类',            EDGE_TREAT,   'solid'),
    ('M2', 'D6', '分类',            EDGE_TREAT,   'solid'),
    ('M3', 'D7', '分类',            EDGE_TREAT,   'solid'),
    ('M4', 'D8', '分类',            EDGE_TREAT,   'solid'),

    # --- 联合治疗 → 联合方案 ---
    ('M5', 'D5', '联合',            EDGE_TREAT,   'solid'),
    ('M5', 'D1', '联合',            EDGE_TREAT,   'solid'),

    # --- 药物 → 生物标志物(预测响应) ---
    ('B1', 'D1', '预测响应',        EDGE_BIOMARK, 'dashed'),
    ('B1', 'D3', '预测响应',        EDGE_BIOMARK, 'dashed'),
    ('B2', 'M1', '预测响应',        EDGE_BIOMARK, 'dashed'),
    ('B3', 'D7', '靶点',            EDGE_BIOMARK, 'dashed'),
    ('B4', 'D5', '靶点',            EDGE_BIOMARK, 'dashed'),

    # --- 药物 → 临床结局 ---
    ('D1', 'O1', '延长',            EDGE_OUTCOME, 'dotted'),
    ('D1', 'O3', '提高',            EDGE_OUTCOME, 'dotted'),
    ('D2', 'O1', '延长',            EDGE_OUTCOME, 'dotted'),
    ('D5', 'O3', '提高',            EDGE_OUTCOME, 'dotted'),
    ('D5', 'O4', '诱导',            EDGE_OUTCOME, 'dotted'),
    ('D5', 'D1', '联合提高',        EDGE_OUTCOME, 'dotted'),
    ('D6', 'O3', '提高',            EDGE_OUTCOME, 'dotted'),
    ('D7', 'O3', '提高',            EDGE_OUTCOME, 'dotted'),
    ('D8', 'O3', '提高',            EDGE_OUTCOME, 'dotted'),
    ('M5', 'O4', '提高',            EDGE_OUTCOME, 'dotted'),
    ('O5', 'UC', '预后相关',        EDGE_OUTCOME, 'dotted'),
    ('O5', 'M6', '介导耐药',        EDGE_OUTCOME, 'dotted'),
    ('O6', 'M1', '免疫相关\n不良反应', EDGE_OUTCOME, 'solid'),
]

# ============================================================
# SVG 生成
# ============================================================
def node_svg(n):
    x, y = n['pos']
    r = n['r']
    lines = n['label'].split('\n')
    if len(lines) == 1:
        ts = [f'<text x="{x}" y="{y+4}" text-anchor="middle" font-size="13" font-weight="600" fill="#fcfcfb">{lines[0]}</text>']
    else:
        line_h = 15
        total = line_h * (len(lines) - 1)
        ts = []
        for i, ln in enumerate(lines):
            dy = -total/2 + i*line_h + 4
            ts.append(f'<text x="{x}" y="{y+dy}" text-anchor="middle" font-size="12" font-weight="600" fill="#fcfcfb">{ln}</text>')
    # 节点圆 + 白色描边
    return (f'<g class="node">'
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{n["color"]}" '
            f'stroke="#fcfcfb" stroke-width="2.5"/>'
            f'{"".join(ts)}'
            f'</g>')

def edge_svg(e):
    src, tgt, label, color, style = e
    s = by_id[src]; t = by_id[tgt]
    x1, y1 = s['pos']; x2, y2 = t['pos']
    # 计算单位向量,从源节点边缘开始绘制
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    ux, uy = dx/d, dy/d
    # 起点
    sx, sy = x1 + ux * s['r'], y1 + uy * s['r']
    # 终点(避开目标节点圆)
    tx, ty = x2 - ux * t['r'], y2 - uy * t['r']

    dash = {'solid': '', 'dashed': '8,4', 'dotted': '2,4'}[style]
    out = [f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
           f'stroke="{color}" stroke-width="1.4" stroke-dasharray="{dash}" opacity="0.75"/>']

    if label:
        # 边的中点处放置标签
        mx, my = (sx+tx)/2, (sy+ty)/2
        # 垂直于边的偏移量
        nx, ny = -uy, ux
        # 标签背景框大小估算
        char_w = 7
        w = len(label.split('\n')[0]) * char_w + 12
        h = 14 * label.count('\n') + 18
        bx = mx + nx*8 - w/2
        by = my + ny*8 - h/2
        # 拆行
        lines = label.split('\n')
        texts = []
        for i, ln in enumerate(lines):
            tdy = -h/2 + 14 + i*14
            texts.append(f'<text x="{mx + nx*8}" y="{my + ny*8 + tdy}" '
                         f'text-anchor="middle" font-size="10" fill="{color}" '
                         f'font-weight="600">{ln}</text>')
        out.append(f'<rect x="{bx}" y="{by}" width="{w}" height="{h}" '
                   f'rx="3" fill="#fcfcfb" stroke="{color}" stroke-width="0.6" opacity="0.95"/>')
        out.extend(texts)
    return ''.join(out)

svg_parts = []
svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
                 f'font-family="system-ui, -apple-system, Segoe UI, "PingFang SC", "Microsoft YaHei", sans-serif">')
svg_parts.append(f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>')

# 标题
svg_parts.append('<text x="750" y="48" text-anchor="middle" font-size="22" font-weight="700" fill="#0b0b0b">'
                 '尿路上皮癌免疫治疗文献知识图谱</text>')
svg_parts.append('<text x="750" y="74" text-anchor="middle" font-size="12" fill="#52514e">'
                 'Knowledge Graph · Urothelial Carcinoma Immunotherapy · 基于 12 篇核心文献</text>')
svg_parts.append('<line x1="580" y1="86" x2="920" y2="86" stroke="#e1e0d9" stroke-width="1"/>')

# 先画所有边(底层)
for e in edges:
    svg_parts.append(edge_svg(e))

# 再画所有节点(顶层)
for n in modalities + middle_nodes + outcome_nodes:
    svg_parts.append(node_svg(n))
svg_parts.append(node_svg(center))

# ============================================================
# 图例
# ============================================================
legend_x, legend_y = 40, 130
svg_parts.append(f'<g transform="translate({legend_x},{legend_y})">')
svg_parts.append('<rect width="240" height="240" rx="6" fill="#f9f9f7" stroke="#e1e0d9" stroke-width="1"/>')
svg_parts.append('<text x="16" y="24" font-size="12" font-weight="700" fill="#0b0b0b">图例 · LEGEND</text>')
svg_parts.append('<line x1="16" y1="32" x2="224" y2="32" stroke="#e1e0d9"/>')

legend_items = [
    (C_DISEASE,  '疾病 (中心)'),
    (C_MODALITY, '治疗模式'),
    (C_DRUG,     '具体药物'),
    (C_CHEMO,    '化疗药物'),
    (C_BIOMARK,  '生物标志物'),
    (C_OUTCOME,  '临床结局'),
]
for i, (c, lbl) in enumerate(legend_items):
    yy = 50 + i*22
    svg_parts.append(f'<circle cx="28" cy="{yy}" r="9" fill="{c}" stroke="#fcfcfb" stroke-width="1.5"/>')
    svg_parts.append(f'<text x="46" y="{yy+4}" font-size="11" fill="#52514e">{lbl}</text>')

# 边样式说明
svg_parts.append('<line x1="16" y1="195" x2="224" y2="195" stroke="#e1e0d9"/>')
svg_parts.append('<text x="16" y="210" font-size="10" fill="#898781" font-weight="600">边类型</text>')
svg_parts.append('<line x1="16" y1="222" x2="44" y2="222" stroke="#7F8C8D" stroke-width="1.4"/>')
svg_parts.append('<text x="50" y="225" font-size="10" fill="#52514e">实线 = 治疗/分类</text>')
svg_parts.append('<line x1="16" y1="234" x2="44" y2="234" stroke="#1baf7a" stroke-width="1.4" stroke-dasharray="8,4"/>')
svg_parts.append('<text x="50" y="237" font-size="10" fill="#52514e">虚线 = 生物标志物</text>')
svg_parts.append('</g>')

# ============================================================
# 右下角:文献覆盖说明
# ============================================================
svg_parts.append('<g transform="translate(1220,130)">')
svg_parts.append('<rect width="240" height="240" rx="6" fill="#f9f9f7" stroke="#e1e0d9"/>')
svg_parts.append('<text x="16" y="24" font-size="12" font-weight="700" fill="#0b0b0b">文献覆盖 · COVERAGE</text>')
svg_parts.append('<line x1="16" y1="32" x2="224" y2="32" stroke="#e1e0d9"/>')

coverage = [
    ('KEYNOTE-045/052', 'Pembrolizumab'),
    ('CheckMate-275',   'Nivolumab'),
    ('IMvigor-130/211', 'Atezolizumab'),
    ('JAVELIN-100',     'Avelumab'),
    ('EV-301/302',      'Enfortumab Vedotin'),
    ('TROPHY-U-01',     'Sacituzumab Govitecan'),
    ('THOR-2/3',        'Erdafitinib'),
    ('MVAC / GC',       'Cisplatin/Gemcitabine'),
    ('综合 / Meta',     '联合治疗'),
    ('TME 研究',        '免疫微环境'),
    ('转化研究',        '生物标志物'),
    ('生存分析',        'OS / PFS'),
]
for i, (a, b) in enumerate(coverage):
    yy = 52 + i*15
    svg_parts.append(f'<text x="20" y="{yy}" font-size="10" fill="#0b0b0b" font-weight="600">{a}</text>')
    svg_parts.append(f'<text x="120" y="{yy}" font-size="10" fill="#52514e">{b}</text>')
svg_parts.append('</g>')

# 底部说明
svg_parts.append('<text x="750" y="1075" text-anchor="middle" font-size="10" '
                 'fill="#898781" font-style="italic">'
                 '中心节点 = 疾病 · 内环 = 治疗模式 · 中环 = 具体药物 + 生物标志物 · 外环 = 临床结局</text>')

svg_parts.append('</svg>')

out_path = r'D:\OncoImmunoRAG\result\knowledge_graph.svg'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(svg_parts))
print(f'[OK] Knowledge graph saved to: {out_path}')
