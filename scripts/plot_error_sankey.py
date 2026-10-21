"""
系统迭代过程中错误类型消解路径图
Sankey 图:错误源 → 修复动作 → 最终结果
Plotly 交互式 + 高清 PNG
"""
import plotly.graph_objects as go
import plotly.io as pio

# ---- 中文安全字体 ----
CHINESE_FONT = 'Microsoft YaHei, PingFang SC, SimHei, sans-serif'

# ============================================================
# 数据设计
# ============================================================
# 左侧(错误来源):4 个节点
# 中间(修复动作):4 个节点
# 右侧(最终结果):2 个节点

node_labels = [
    # 0–3 错误源
    '跨文献数据污染',            # 0 — 5 项
    '引用归因幻觉',              # 1 — 4 项
    '检索偏倚',                  # 2 — 3 项
    'P 值误判',                  # 3 — 1 项
    # 4–7 修复动作
    '目标文献路由',              # 4
    '区块过滤/后处理正则',       # 5
    'MMR 去冗余',                # 6
    'Prompt 规则注入',           # 7
    # 8–9 最终结果
    '已消除错误',                # 8 — 12 项(醒目绿)
    '残留口径差异',              # 9 — 1 项
]

# 节点配色(按列语义区分)
node_colors = [
    '#e34948',  # 跨文献数据污染 — 红
    '#eb6834',  # 引用归因幻觉 — 橙红
    '#F18F01',  # 检索偏倚 — 橙
    '#eda100',  # P 值误判 — 黄
    '#2E86AB',  # 目标文献路由 — 蓝
    '#4a3aa7',  # 区块过滤 — 紫
    '#008300',  # MMR 去冗余 — 绿
    '#A23B72',  # Prompt 规则注入 — 紫红
    '#0ca30c',  # 已消除错误 — 醒目绿(高饱和)
    '#898781',  # 残留口径差异 — 中灰
]

# ============================================================
# 连边:(源, 目标, 值, 说明)
# ============================================================
links = [
    # ---- 错误源 → 修复动作 ----
    # 跨文献数据污染 (5) → 主要靠"目标文献路由"
    (0, 4, 4, '数据污染'),
    (0, 5, 1, '残留片段'),
    # 引用归因幻觉 (4) → 主要靠"Prompt 规则注入"
    (1, 7, 3, '幻觉片段'),
    (1, 5, 1, '正则补抓'),
    # 检索偏倚 (3) → "MMR 去冗余"为主
    (2, 6, 2, '冗余片段'),
    (2, 5, 1, '过滤补强'),
    # P 值误判 (1) → "Prompt 规则注入"
    (3, 7, 1, '判读错误'),

    # ---- 修复动作 → 最终结果 ----
    # 12 项流向"已消除错误"
    (4, 8, 4, '路由消除'),
    (5, 8, 3, '正则消除'),
    (6, 8, 2, '去冗余消除'),
    (7, 8, 3, '规则消除'),
    # 1 项流向"残留口径差异"(VESPER ITT 差异)
    (7, 9, 1, 'VESPER\nITT 差异'),
]

source_idx = [l[0] for l in links]
target_idx = [l[1] for l in links]
values     = [l[2] for l in links]
link_labels = [f"{l[3]} · {l[2]} 项" for l in links]

# ============================================================
# 连边颜色:按目标语义着色
# 已消除(目标 8)→ 醒目绿;残留(目标 9)→ 灰色;
# 中间过渡 → 用源节点颜色淡化
# ============================================================
def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})'

link_colors = []
for s, t, v, lab in links:
    if t == 8:                         # 已消除错误 → 醒目绿色流
        link_colors.append('rgba(12, 163, 12, 0.50)')
    elif t == 9:                       # 残留差异 → 灰色流
        link_colors.append('rgba(137, 135, 129, 0.55)')
    else:                              # 中间过渡 → 沿用源色
        link_colors.append(hex_to_rgba(node_colors[s], 0.35))

# ============================================================
# 绘图
# ============================================================
fig = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=22,
        thickness=26,
        line=dict(color='#fcfcfb', width=2),
        label=node_labels,
        color=node_colors,
        hovertemplate='%{label}<extra></extra>',
    ),
    link=dict(
        source=source_idx,
        target=target_idx,
        value=values,
        color=link_colors,
        label=link_labels,
        hovertemplate='%{label}<br>数量: %{value} 项<extra></extra>',
    ),
)])

# ---- 节点值标注(在每个节点右侧显示总量) ----
node_totals_in = [0] * len(node_labels)
node_totals_out = [0] * len(node_labels)
for s, t, v, _ in links:
    node_totals_out[s] += v
    node_totals_in[t]  += v

# 节点名后面追加总数,作为图例增强
node_labels_display = []
for i, lab in enumerate(node_labels):
    if i in (8, 9):  # 终点节点
        node_labels_display.append(f"{lab}\n<{node_totals_in[i]} 项>")
    elif i <= 3:      # 起点节点
        node_labels_display.append(f"{lab}\n<{node_totals_out[i]} 项>")
    else:             # 中间修复节点
        node_labels_display.append(f"{lab}\n→{node_totals_in[i]} 项")

# ---- 标题与布局 ----
fig.update_layout(
    title=dict(
        text='系统迭代过程中错误类型消解路径图',
        x=0.5, xanchor='center', y=0.96,
        font=dict(size=22, color='#0b0b0b',
                  family=CHINESE_FONT),
    ),
    font=dict(family=CHINESE_FONT, size=12, color='#0b0b0b'),
    paper_bgcolor='#fcfcfb',
    plot_bgcolor='#fcfcfb',
    width=1500,
    height=820,
    margin=dict(l=20, r=20, t=80, b=60),
)

# ---- 更新节点标签(含总数) ----
fig.data[0].node.label = node_labels_display

# ---- 注释:阶段分隔说明 ----
annotations = [
    dict(x=0.05, y=1.04, xref='paper', yref='paper', xanchor='left',
         text='<b>错误来源 (Sources)</b><br><span style="font-size:10px;color:#898781">13 项待处理</span>',
         showarrow=False, align='left',
         font=dict(family=CHINESE_FONT, size=12, color='#52514e')),
    dict(x=0.50, y=1.04, xref='paper', yref='paper', xanchor='center',
         text='<b>修复动作 (Fix Pipeline)</b><br><span style="font-size:10px;color:#898781">四道工序协同</span>',
         showarrow=False, align='center',
         font=dict(family=CHINESE_FONT, size=12, color='#52514e')),
    dict(x=0.95, y=1.04, xref='paper', yref='paper', xanchor='right',
         text='<b>最终结果 (Outcome)</b><br><span style="font-size:10px;color:#0ca30c"><b>92.3% 消解</b></span>',
         showarrow=False, align='right',
         font=dict(family=CHINESE_FONT, size=12, color='#52514e')),
    # 底部脚注
    dict(x=0.5, y=-0.04, xref='paper', yref='paper', xanchor='center',
         text='流宽 ∝ 错误数量   ·   绿色汇聚 = 已消除(12 项)   ·   灰色分支 = 残留口径差异(1 项 · VESPER ITT)',
         showarrow=False,
         font=dict(family=CHINESE_FONT, size=10, color='#898781')),
]
fig.update_layout(annotations=annotations)

# ============================================================
# 输出
# ============================================================
out_html = r'D:\OncoImmunoRAG\result\error_sankey.html'
out_png  = r'D:\OncoImmunoRAG\result\error_sankey.png'

fig.write_html(out_html, include_plotlyjs='cdn')
print(f'[OK] Interactive HTML: {out_html}')

# 尝试导出 PNG(kaleido 可用时)
try:
    fig.write_image(out_png, width=1500, height=820, scale=2)
    print(f'[OK] High-res PNG (2x): {out_png}')
except Exception as e:
    print(f'[WARN] PNG 导出失败(kaleido 未安装): {e}')
    print('       HTML 仍可用,可浏览器右键另存为图片')

# 在 Jupyter / 浏览器中显示
try:
    fig.show()
except Exception:
    pass
