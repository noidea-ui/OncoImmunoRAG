# OncoImmunoRAG → WSL 迁移交接记录

> 生成时间：2026-10-21
> 适用场景：Windows 开发环境 → WSL 开发环境
> 本文档仅记录**已核实的事实**、**静态分析结论**与**未验证事项**，不含任何凭据、密钥或原始对话记录。

---

## 0. 阅读约定

本文用三种标记区分证据强度，请勿混用：

| 标记 | 含义 |
|---|---|
| **[已验证]** | 在 Windows 环境实际执行并观察到结果 |
| **[静态]** | 通过读码 / AST / grep 得出，**未实际触发** |
| **[未验证]** | 尚未获得证据，迁移后需重新确认 |

---

## 1. 项目目标与当前工作目标

### 1.1 项目身份

- **仓库名 / 目录名**：`OncoImmunoRAG`
- **分发名**：`oncoimmuno-rag`（`pyproject.toml` 的 `[project].name`）
- **包版本**：`0.1.0`
- **项目类型**：面向肿瘤免疫治疗文献的检索增强生成（RAG）+ 证据分级系统
- **上游描述**：`pyproject.toml:4` — "RAG system for oncology immunotherapy literature: PDF ingestion, vector retrieval, evidence grading, and evaluation."

### 1.2 目标能力（产品设想，**非当前实现状态**）

1. 检索、筛选、组织与临床问题相关的研究文献（尤其 RCT）
2. 对试验结果做基于文献证据的 **RoB 2** 偏倚风险评估
3. 按**结局**组织证据并做 **GRADE** 证据确定性评价（含"量化 GRADE"自定义评分）
4. 从论文正文、表格、**图表**中提取临床结局、效应量与不确定性，并可追溯到来源
5. 把单篇分析、跨研究综合结论与可视化看板连接起来

> ⚠️ **重要**：以上 5 条是**产品目标**，不是现状。当前代码实际形态是"**单篇文献的自动评分卡**"，目标 2（RoB 2）与目标 3 的"按结局组织"**在代码中完全不存在可运行实现**。详见 §4。

### 1.3 当前工作目标（本次交接时点）

**只做三件事**：
1. 环境搭建与恢复
2. 理解已有功能
3. 核实已有缺陷

**明确不做**：不新增业务功能、不做算法重构、不做 UI 改版、不做无关清理、不修复缺陷（缺陷只记录，不动代码）。

---

## 2. 技术栈、目录结构与核心模块

### 2.1 技术栈 **[已验证：依赖已安装并可 import]**

| 层 | 技术 | 锁定版本 | 来源 |
|---|---|---|---|
| 语言 | Python | **3.12.15** | `pyproject.toml:9` 声明 `>=3.10,<3.15` |
| 依赖管理 | Poetry | 2.5.1 | `poetry.lock` `lock-version = "2.1"` |
| RAG 框架 | llama-index | 0.14.23 | `llama-index-core` |
| PDF 读取 | llama-index-readers-file + pypdf | 0.6.0 / 6.14.2 | `PDFReader` 后端 |
| 表格抽取 | 自研（正则 + 文本流） | — | `src/ingestion/table_extractor.py` |
| 向量库 | ChromaDB | 0.6.3 | `llama-index-vector-stores-chroma` 0.6.0 |
| 向量库适配层 | llama-index-vector-stores-chroma | 0.6.0 | **本轮新增声明**，见 §6.3 |
| 嵌入模型 | sentence-transformers 本地推理 | 5.6.1 | `BAAI/bge-large-en-v1.5`（**权重未下载**） |
| 深度学习 | torch | 2.13.0 | |
| LLM 客户端 | **裸 `openai` SDK**（非 llama-index 封装） | 1.109.1 | 指向 OpenAI 兼容端点 |
| 分级输出 | pydantic `BaseModel`（`extra="forbid"`） | — | `src/grading/grade_evaluator.py:204` |
| 前端 | Streamlit | 1.66.0 | `app.py:36`，**本轮新增声明** |
| 测试 / 检查 | pytest / pytest-cov / ruff | 8.4.2 / 6.3.0 / 0.16.0 | |

**关键约束（迁移时必须复核）**：
- `poetry.lock` 顶层 `python-versions = ">=3.10,<3.15"`，但锁内 **`numpy 2.5.1` 声明 `python-requires >=3.12`** → **实际只能在 3.12 / 3.13 / 3.14 安装，3.10 与 3.11 不可用** **[已验证：通过读取 lock 的 `python-requires` 字段确认]**
- `torch` 2.13.0 在 Windows 依赖 `nvidia-*` 系列 Linux 轮子；这些在 lock 中带平台 marker，**Windows 上自动跳过** **[已验证：25 个包被正确跳过]**

### 2.2 目录结构

```
OncoImmunoRAG/
├── app.py                    # Streamlit 三 Tab 前端（848 行）—— 实际程序入口
├── config.yaml               # 运行时配置（当前不含任何密钥）
├── pyproject.toml            # 依赖与工具配置
├── poetry.lock               # 依赖锁文件（应纳入版本控制）
├── .gitignore                # 忽略规则
├── .gitattributes            # 换行/二进制策略（LF 文本 + binary 资产）
├── .env.example              # 环境变量模板（无秘密值）
├── src/                      # 【活跃实现】
│   ├── cli.py                # 空桩（仅 print 占位）
│   ├── ingestion/            # pdf_parser / table_extractor / vector_store / append_pdfs
│   ├── retrieval/            # query_engine / selector / pubmed_client / temp_indexer
│   ├── grading/              # grade_evaluator（1587 行，核心算法）
│   ├── evaluation/           # 只有空 __init__.py，无任何模块
│   └── utils/                # config_loader / prompts / trial_registry
├── src_00/                   # 【历史快照，非活跃实现】pre-rerank 版 src/ 副本
├── scripts/                  # 16 个评测批处理 / 绘图脚本
├── tests/                    # 4 个 test_*.py + 2 个评测脚本 + results/ 产物
├── data/
│   ├── raw/                  # 12 篇原始 PDF（被 gitignore）
│   ├── vector_db/            # Chroma 主库（被 gitignore）
│   ├── vector_db_22/         # Chroma 第二库（被 gitignore）
│   └── benchmark_results/    # 12 个评测 JSON
├── verification/             # 5 篇 PDF + results/ 5 JSON + _summary
├── holdout/                  # 5 篇 PDF + results/ + rerank_ab/ + search_ab/
├── result/                   # 16 个图表产物
└── .venv/                    # 本地虚拟环境（被 gitignore，1.3 GB）
```

### 2.3 核心模块与调用关系 **[静态 + 部分已验证]**

```
config_loader.load_config()  →  AppConfig（frozen dataclass）
        │
        ├─► ingestion.pdf_parser.parse_pdfs(config=cfg)
        │      ├─ load_pdfs()        → SimpleDirectoryReader + PDFReader → 每页一个 Document
        │      └─ chunk_documents()  → _split_document_by_sections() → SentenceSplitter(1024/200)
        │                              + table_extractor.extract_table_summaries()
        │
        ├─► ingestion.vector_store.build_index()  → HuggingFaceEmbedding + ChromaDB
        └─► ingestion.append_pdfs.append_index()  → 增量追加

retrieval.query_engine.QueryEngine.retrieve()      # MMR 过取 → rerank_hits → select_top_k
retrieval.selector.{rerank_hits, select_top_k}     # 纯 Python，零第三方依赖（便于单测）
retrieval.temp_indexer.TempIndexer                 # 临时 Chroma 集合，app.py 未调用其 retrieve()
retrieval.pubmed_client.{search_pubmed, fetch_details}  # NCBI E-utilities，无缓存

grading.grade_evaluator.grade_extraction(chunks)  # LLM 抽取 → 归一化 → 注册表覆盖
                                                   # → composite_score 覆盖 → RCT floor → 等级映射
utils.trial_registry                              # 人工权威表（6 条），被 grading + selector 共用
utils.prompts                                      # 统计学显著性规则 + Discussion 段处理规则
```

**导入方式**：`src/` 下的包用**绝对导入**（`from ingestion.xxx import`），依赖 `src/` 位于 `sys.path`：
- `app.py:29-31` 手动 `sys.path.insert(0, _SRC)`
- `pyproject.toml:51` `pythonpath = ["src"]`（pytest）

**`src_00/` 未被任何代码引用** **[已验证：全仓 `.py` 搜索 `src_00` 零命中；`pythonpath` 与 `app.py` 均只指向 `src`]** → 纳入版本控制不会造成导入遮蔽。但它是 `src/` 的旧副本，其 `query_engine.py` 带有与 `src/` **相同**的 `QueryEngine.query` 缩进缺陷。

---

## 3. 已完成的勘察结论

### 3.1 已确认为事实 **[已验证]**

| 结论 | 依据 |
|---|---|
| 现有代码是**单篇文献评分卡**，非循证辅助平台 | Tab1/Tab2/Tab3 均对单文档调用 `grade_extraction`；跨论文路径不可达（§4 D1） |
| **RoB 2 完全不存在** | 全仓 `RoB` / `rob2` / `Cochrane` / `signalling` / `D1..D5` **0 命中**；`risk_of_bias` 只是 5 个 GRADE 降级域之一的扁平字段（`grade_evaluator.py:207`） |
| **不存在结局层 GRADE** | `GRADEResult`（`grade_evaluator.py:204-235`）13 个字段**无一个是 list/嵌套模型**，`extra="forbid"`；`grade_extraction` 内无结局循环 |
| **不存在跨研究综合引擎** | 全仓唯一跨论文代码 `_QA_PROMPT_STR` 依赖 `QueryEngine.query`，而该方法不存在 → 运行时不可达 |
| **图表能力 = 0** | `forest` / `kaplan` / `median survival` / `OCR` / `image_url` / `base64` / `multimodal` / `pdfimages` / `fitz` / `camelot` / `pdfplumber` 全仓 **0 命中**；`PDFReader()` 无参构造；两个 LLM 调用点的 `messages` content 一律是 `str` |
| **表格结构实测全毁** | 对 `data/vector_db_22/chroma.sqlite3` 只读 SQL 查询：25 张表中 `column_count>=2` 的有 **0** 张，16 张被压成单列，9 张退化为围栏文本块 |
| **无页码/块级溯源** | `GRADEResult` 无 `page` / `chunk_id` / `table_label` 字段；四个结果目录 grep `page` → **0 命中**；`grade_extraction` 的 `chunks` / `chunk_file_names` 消费后即丢弃（docstring 自述 "informational only"，`grade_evaluator.py:1276-1278`） |
| **无 CI 提取器** | 无 `_extract_ci`；95% CI 仅作为自由文本留在 `effect_size` 字段，不参与任何计算 |
| **章节分类大面积失效** | 12 篇 PDF 实测 413 chunks：`primary` 27（6.5%）、`secondary` 70（16.9%）、`unknown` **300（72.6%）**、`table_summary` 16（3.9%）；IMvigor010 的 `primary` = **0** |
| **落盘结果含可核实的临床事实错误** | `data/benchmark_results/McGregor-DAD.json:20,26,32,46,47,49` — 5 个检索块中 3 个来自 EV-302，N=886 与 HR=0.47 均为 EV-302 的数值，LLM 还把 I 期单臂试验写成"随机对照试验"；另有 Nadofaragene（注册表明写单臂，LLM 判 RCT）、NIAGARA（伪造"开放标签"，实为双盲）、KEYNOTE-361（同一句内既称双盲又称开放标签）、Mamede meta 分析（被当单篇 RCT，合并 N=2220） |

### 3.2 仍未验证的判断 **[未验证]**

| 事项 | 为何未验证 | 迁移后如何验证 |
|---|---|---|
| `QueryEngine.query` 修复后，全局问答能否产出合规输出 | 需实例化 `QueryEngine` → 触发嵌入模型下载 + LLM 调用 | 授权后端到端验证 |
| 章节分类失效（72.6% unknown）是否真的拉低检索质量 | 需真实向量库 + 嵌入模型 + 完整 top-K 观察 | 重建索引后比对 top-K 的 `section_type` 分布 |
| `per_file_cap` 绕过（§4 D7）在真实向量库上的影响幅度 | 纯 Python 已复现机制，但真实库的分布未知 | 用真实检索结果统计单篇占比 |
| 跨文献污染在**实时**检索下的发生率 | 已落盘数据可证污染曾发生；当前实时路径未跑 | 实时检索后检查 `effect_size` 与 canonical 文件是否一致 |
| `_apply_rct_floor` 的 "RCT 下限" 是否在真实数据上被频繁触发 | 需完整 benchmark 重跑 | 修复缺陷后重跑 12 篇基准并统计 |
| Streamlit 各 Tab 的实际交互行为 | 仅验证了服务能启动并返回 HTTP 200，未做交互 | 人工点击三个 Tab |
| 22 篇论文中是否存在**同一试验的重复收录** | 仅从 `app.py` 的人工列表与 `holdout/results/Mamede-MetaAnalysis.json` 推断 | 建立 study identity 索引时核对 |
| `evaluation/` 与 `holdout/` 两组 10 篇 PDF 是否需要纳入向量库 | 现状：向量库只含 `data/raw` 的 12 篇，但 UI 宣称"22 篇库内" | 与项目负责人确认语料范围 |

---

## 4. 缺陷清单

> 排序按严重程度。每条标注 **[静态]** / **[已验证]**。
> 本轮**未修复任何一条**，代码保持原样。

### 🔴 D1 — `QueryEngine` 没有 `query()` 方法（跨论文问答全线死路）

- **位置**：`src/retrieval/query_engine.py:435`（`class _NodeWithScore`）、`:450`（`def query`）
- **机制**：`query` 方法被误缩进到紧邻的辅助类 `_NodeWithScore` 内，而不在 `class QueryEngine`（`:291`）内。`QueryEngine` 实际只有 `__init__` / `from_config` / `retrieve` / `_retrieve_canonical_chunks` / `set_rerank_enabled` / `get_rerank_enabled`。
- **证据 [已验证]**：AST 解析 → `QueryEngine` 成员中无 `query`；`hasattr(QueryEngine,'query')` = `False`，`hasattr(_NodeWithScore,'query')` = `True`。
- **影响**：
  - `app.py:829` `engine.query(ga_query)` → `AttributeError`（被 `except Exception` 捕获，UI 静默显示"检索失败"）
  - 模块级 `query()`（`query_engine.py:525`）→ 同错
  - CLI 自测 `main()`（`query_engine.py:574`）→ 同错
  - **后果：产品目标 5（宏观综合）完全不可达**
- **附带**：方法体内部调用 `self.retrieve(question)`，而 `_NodeWithScore` 无此方法（`__slots__ = ("node","score")`）→ 即便能调用也会二次报错。
- **`src_00/retrieval/query_engine.py` 有同样缺陷**（`:407` / `:422`）。

### 🔴 D2 — 跨文献污染：提示词要求按文件名归类，但片段里根本没有文件名

- **位置**：`src/grading/grade_evaluator.py:333-346`（`_CROSS_PAPER_PROMPT_GUARD`）vs `:464-469`（`_format_chunks`）
- **机制**：守卫规则 1 要求"先把片段按文件名归类"，规则 2/3/4 要求只从 canonical 文件的片段提取；但 `_format_chunks` 只输出 `[片段 N]` 标签，**不含文件名、不含 `file_name` 元数据、不含 section 类型**。
- **证据 [已验证]**：`_format_chunks(["来自论文A的片段内容","来自论文B的片段内容"])` 实际输出 `['[片段 1]', '来自论文A的片段内容', '', '[片段 2]', '来自论文B的片段内容']`。
- **影响**：守卫对多文件输入**在设计上不可执行**；LLM 无从判断某块属于哪篇论文。

### 🔴 D3 — `effect_size` 零 Python 侧校验，别篇效应量可决定最终评分

- **位置**：`src/grading/grade_evaluator.py:952`（`_extract_hr(result.effect_size)`）、`:971`（`_extract_pvalue_composite`）
- **机制**：`sample_size` 尚有 `trial_registry.canonical_sample_size` 兜底（且只覆盖 6/22 篇）；`effect_size` **完全没有**。`_extract_hr` 直接从该字符串取第一个 HR 并用于 composite score。
- **证据 [已验证：落盘产物]**：`data/benchmark_results/McGregor-DAD.json` — `:46` `sample_size=886`（EV-302 的 N，McGregor-DAD 实际 N=24）、`:47` `effect_size="HR=0.47, 95% CI 0.38-0.58"`（EV-302 的 HR）、`:49` `grade_reasoning` 把这些写成 McGregor-DAD 自身结果。
- **影响**：跨文献数字污染进入评分与最终展示。

### 🔴 D4 — "量化 GRADE" 的提示词阈值与 Python 实现不一致

- **位置**：提示词 `src/grading/grade_evaluator.py:408-412` vs 实现 `:1039-1059`（`_grade_to_bucket`）
- **证据 [已验证]**：

  | 区间 | 提示词 | Python | 一致? |
  |---|---|---|---|
  | ≥ 8.5 | 高 | 高 | ✓ |
  | 5.0 ≤ x < 6.0 | 中 | **低** | ✗ |
  | 3.5 ≤ x < 5.0 | 低 | 低 | ✓ |
  | 2.6 ≤ x < 3.5 | 低 | **极低** | ✗ |

  实测：`score=5.9` → 提示词判"中"、代码判"低"；`score=3.4` → 提示词判"低"、代码判"极低"。
- **影响**：提示词自 `:419` 起已自相矛盾；LLM 侧推理文本与最终等级口径不一致。Python 生效（分数总是 Python 重算，`:1422-1438`）。

### 🔴 D5 — 触发器(A) 规格 −3.0，实现 −1.5

- **位置**：提示词 `:380-381` vs 实现 `:938-941`
- **机制**：代码注释假设存在一个"常规 −1.5 的 confounding 惩罚"要叠加，但 5 域循环（`:911-917`）只含 `risk_of_bias / inconsistency / indirectness / imprecision / publication_bias`，**不含 `confounding_by_subsequent_therapy`** → 无可叠加项，净效果只有 −1.5。
- **证据 [已验证]**：`RCT` + `confounding_by_subsequent_therapy="是"` → `composite_score = 8.5`（规格应为 7.0），breakdown 字面写 `"降级:极严重[A:confounding=是 & RCT]→-1.5"`。
- **影响**：RCT + 后续治疗混杂的惩罚比规格弱 1.5 分。

### 🔴 D6 — 章节分类大面积失效（72.6% `unknown`）

- **位置**：`src/ingestion/pdf_parser.py:129-150`（`_split_document_by_sections`）、`:72-86`（`_SECTION_RULES`）
- **机制**：① `PDFReader` 默认 `return_full_document=False` → **每页一个 Document**，分类按页进行；② 章节正则要求标题**独占一行**且 `(?im)^` 锚定 → 页首的延续内容、跨页的章节标题都匹配不到；③ 页首到首个标题之间的内容一律归入 `unknown`。
- **证据 [已验证]**：12 篇 PDF 实测 413 chunks → `primary` 27（6.5%）/ `secondary` 70（16.9%）/ `unknown` **300（72.6%）**/ `table_summary` 16（3.9%）；IMvigor010 `primary=0`。
- **影响**：`selector.select_top_k` 的填充顺序是 primary → unknown → secondary，而重排层给 unknown 最低权重（`_RERANK_SECTION_WEIGHT` 中 unknown = 0.00，primary = +0.30）→ 系统性地把 72.6% 的语料判为低价值。

### 🟠 D7 — `per_file_cap` 被回填循环完全绕过

- **位置**：`src/retrieval/selector.py:365`（`per_file_cap = max(1, (top_k+1)//2)`）、`:129-152`（`_take_with_cap`）、`:466-482`（回填循环）
- **机制**：`_take_with_cap` 的 `counts` 字典每次调用都重置（`:142`），4 个桶各自独立计额；回填循环（`:466-482`）**不做任何 per-file 计额**，且 `primary_skipped` 恰好是"因超限被跳过"的集合，回填时原样放回。
- **证据 [已验证]**：单一文件 A.pdf 的 primary 块池 → top_k=5/8/12/25 选出 5/8/12/25 块，**单篇占比 100%**。
- **影响**：`selector.py:26-27` docstring 声称"Per-file caps prevent one paper from monopolising the top-K"，实际不成立。

### 🟠 D8 — Poetry 依赖声明缺失（本轮已修复，待确认）

- **位置**：`pyproject.toml`（原缺 2 项）
- **导入点**：
  - `llama_index.vector_stores.chroma` → `src/ingestion/vector_store.py:26`、`src/ingestion/append_pdfs.py:32`、`src/retrieval/query_engine.py:36`、`src/retrieval/temp_indexer.py:47`（**4 个模块**）
  - `streamlit` → `app.py:36`
- **兼容性依据 [已验证]**：`llama-index` 0.14.23 **不**依赖 `llama-index-vector-stores-chroma`（必须单独声明）；`llama-index-vector-stores-chroma` 0.6.0 要求 `chromadb>=0.5.17`（锁 0.6.3 ✓）与 `llama-index-core>=0.13,<0.15`（锁 0.14.23 ✓）；`streamlit` 1.65.0/1.66.0 要求 `numpy<3`（锁 2.5.1 ✓）、`pandas<4`（锁 2.3.3 ✓）、Python ≥3.10/3.11 ✓。
- **证据 [已验证]**：修复前 `ingestion.vector_store` / `ingestion.append_pdfs` / `retrieval.query_engine` / `retrieval.temp_indexer` 全部 `ModuleNotFoundError: No module named 'llama_index.vector_stores'`；`streamlit` 同样缺失。修复后 4 个模块 + streamlit 全部 import 成功。
- **状态**：**已修改 `pyproject.toml` 与 `poetry.lock`，当前为未暂存状态，未 commit。**

### 🟠 D9 — 跨文献检测器在 UI 路径永不触发，且其文案断言了未验证的事实

- **位置**：`src/grading/grade_evaluator.py:1318-1326` vs `app.py:159`、`app.py:212`
- **机制**：`app.py` 两处 `chunk_file_names` 恒为单元素列表 → `len(unique_sources) > 1` 永假 → 遥测不产生。
- **证据 [已验证]**：`app.py:159` `chunk_file_names = [paper["file_name"]] * len(used_chunks)`；`app.py:212` `chunk_file_names=[file_name] * len(used_chunks)`。
- **附加**：该遥测文案 `"数据仅来自 canonical_file_name=… 的片段"` 断言了一个**代码从未验证**的事实。

### 🟠 D10 — `risk_of_bias` 不是 RoB 2，且"未报告"被塌缩为"中"

- **位置**：`src/grading/grade_evaluator.py:207`（字段定义）、`:445`（提示词要求 LLM 输出）、`:174-199`（`_RISK_FALLBACKS` / `_normalize_risk`）
- **证据 [已验证]**：`_normalize_risk` 实测 —— `未提及` / `未知` / `未报告` / `不适用` / `n/a` / `''` **全部 → `'中'`**。
- **影响**：无法评估的状态被伪装成确定的"中"，且该"中"在 composite score 中直接扣 1.0 分。**不可评估 ≠ 中等偏倚**，这是方法学层面的信息损失。

### 🟠 D11 — `_apply_rct_floor` 可在无新证据的情况下把 `risk_of_bias` 从"高"改写为"低"

- **位置**：`src/grading/grade_evaluator.py:1185-1251`（判定条件 `:1202-1210`），覆盖动作 `:1228-1251`
- **机制**：前置条件为 `study_design ∈ {RCT,系统评价,Meta分析}` 且 `sample_size >= 300` 且 `P < 0.05` 且无"严重证据"正则命中 → `risk_of_bias` 与 `imprecision` 被压为"低"。判定**从不查看 CI 宽度**。
- **状态 [静态]**：代码路径已确认；触发频率需重跑 benchmark 才能统计。

### 🟠 D12 — `trial_registry.multiple_primary_endpoints` 定义后从未被读取

- **位置**：`src/utils/trial_registry.py:64`（定义）vs `src/grading/grade_evaluator.py:1098`（`_has_multiple_primary_endpoints`，改用对 LLM 散文的中文短语启发式）
- **证据 [已验证：全仓 grep]**：该字段名在全仓仅 **1 处定义、0 处读取**。
- **影响**：人工权威表里的双主终点标记被无声忽略。

### 🟠 D13 — Streamlit 关闭重排层，与模块默认矛盾

- **位置**：`app.py:119`（`set_rerank_enabled_default(False)`）vs `src/retrieval/selector.py:310`（`_RERANK_ENABLED_DEFAULT: bool = True`）
- **证据 [已验证：grep]**：两处均确认。
- **影响**：整套 2026-09 规格的证据加权重排逻辑在 UI 主路径**完全不生效**。`holdout/rerank_ab/` 存在 A/B 对比产物，说明该层曾被评估过。

### 🟠 D14 — pytest 覆盖率门禁失效

- **位置**：`pyproject.toml:52` `addopts = "-q --cov=oncoimmuno_rag --cov-report=term-missing"`
- **机制**：`oncoimmuno_rag` 不是可导入包（`package-mode = false`，实际包名为 `ingestion` / `retrieval` / `grading` / `utils` / `evaluation`）。
- **证据 [已验证]**：实际运行报 3 条 warning —— `Module oncoimmuno_rag was never imported` → `No data was collected` → `Failed to generate report: No data to report`。
- **临时绕行（不改配置）**：`pytest tests/ -o addopts="" -p no:cacheprovider -q`

### 🟠 D15 — `load_config(raise_if_missing=False)` 仍抛异常

- **位置**：`src/utils/config_loader.py:119-139`
- **机制**：`raise_if_missing=False` 时走 `return _coerce({})`，而 `_coerce` 对缺失必需键抛 `ValueError`。
- **证据 [已验证]**：实测 `ValueError: config.yaml is missing required keys: ['embedding_model', 'chunk_size', 'chunk_overlap']` —— docstring 承诺的"返回默认构造的配置"未实现。

### 🟡 D16 — `append_index` 返回 `None` 但类型标注为 `VectorStoreIndex`

- **位置**：`src/ingestion/append_pdfs.py:53`（标注 `-> tuple[VectorStoreIndex, list, set[str], int]`）、`:107`、`:127`（实际 `return None, ...`）
- **状态 [静态]**：需向量库才能实际调用，未复现。

### 🟡 D17 — `_detect_endpoints` 遍历 `frozenset`，但 docstring 声称保持插入顺序

- **位置**：`src/ingestion/table_extractor.py:157-166`、`_endpoint_keyword_set()` 返回 `frozenset`
- **证据 [已验证]**：返回类型确为 `frozenset`。
- **影响**：命中关键词的顺序不确定；`contains_endpoints` 元数据顺序不稳定。

### 🟡 D18 — 两个"冒烟测试"文件贡献 0 个测试

- **位置**：`tests/test_config_smoke.py`（只有 `main()`）、`tests/test_new_deps_smoke.py`（只有模块级 print）
- **证据 [已验证]**：`--collect-only` 只收集到 15 个测试；这两个文件各 0 个。
- **附加**：`test_new_deps_smoke.py` 在 **import 阶段**就 `import sentence_transformers` → 拉起 torch；若缺包会导致整个 pytest run 在收集阶段 error。

### 🟡 D19 — README 三处与实现不符（按要求保留，未修改）

- `README.md:33` 的 `poetry run oncoimmuno-rag` **不存在** —— `pyproject.toml` 无 `[tool.poetry.scripts]` 段，且 `src/cli.py` 是空桩（仅打印占位文本）
- `src/evaluation/` 只有 3 行空 `__init__.py`（`__all__ = []`），**无任何 evaluation 模块**；README 描述的目录结构与实际不符
- README 写 "load & clean PDFs"，但代码中**无独立清洗阶段**（仅章节切片的 `.strip()`）

### 🟡 D20 — ruff 检查不通过

- **证据 [已验证]**：全仓 `ruff check .` → **248 个问题**（其中 `src_00/` 旧快照占 39，活跃 `src/` 占 54；`app.py` 30、`scripts/` 约 90、`tests/` 11）。
- 规则分布：E501(71) 行过长、I001(28) import 排序、UP035/UP045/UP006(53) pyupgrade、W292(14) 缺尾换行、F541(12) 空 f-string、F401(11) 未用 import、E701/E702(21) 单行多语句等。
- `ruff format --check .` → **47 个文件需重排**（活跃 `src/`+`app.py`+`tests/` 子集为 20 个）。
- **绝大多数为风格项，非功能缺陷**；本轮未做任何修改。

### 🟡 D21 — Tab1 只用 PDF 前 12 块，且 chat 输入从不进入提示词

- **位置**：`app.py:158`（`used_chunks = list(chunks[: max(top_k, 1)])`，`top_k=12`）、`app.py:492-537`
- **影响**：`chunk_size=1024, overlap=200` → 约 9,888 字符 ≈ 全文的 9–12%（Results / Conclusions 通常在窗口外）；`st.chat_input` 的文本仅被 append 到 session_state，**从不进入任何提示词**，但 placeholder 宣传"讨论部分有哪些外部研究"。
- **状态 [静态]**：未做交互验证。

### 🟡 D22 — 语料范围与 UI 宣称不一致

- `app.py:428` 宣称"22 篇库内"，但向量库 `data/vector_db` 只由 `data/raw` 的 12 篇构建；`verification/` 与 `holdout/` 的 10 篇**未入索引**（仅作为 Tab1 的直读 PDF 源）。

---

## 5. Git 状态与需要保留的数据

### 5.1 Git 现状 **[已验证]**

```
仓库路径 : D:/program/OncoImmunoRAG/.git
远程     : origin → https://github.com/noidea-ui/OncoImmunoRAG.git
分支     : main（跟踪 origin/main，已同步）
提交数   : 1
HEAD     : ced4543  "chore: 建立安全的 Git 版本控制基线"
```

**⚠️ 远程仓库在首次推送时为空**（`git ls-remote` 返回 0 refs），因此 `ced4543` 是唯一提交，**没有需要保护的历史**。

**暂存区（18 个文件，已暂存未提交）**：
```
.gitattributes
.gitignore
src_00/cli.py
src_00/evaluation/__init__.py
src_00/grading/__init__.py
src_00/grading/grade_evaluator.py
src_00/ingestion/__init__.py
src_00/ingestion/append_pdfs.py
src_00/ingestion/pdf_parser.py
src_00/ingestion/table_extractor.py
src_00/ingestion/vector_store.py
src_00/retrieval/__init__.py
src_00/retrieval/query_engine.py
src_00/retrieval/selector.py
src_00/utils/__init__.py
src_00/utils/config_loader.py
src_00/utils/prompts.py
src_00/utils/trial_registry.py
```

**未暂存修改（2 个文件）**：
- `pyproject.toml` — +2 个依赖声明 + 6 行注释（§4 D8 的修复）
- `poetry.lock` — 171→184 包（+13 新增 / 0 删除 / 1 处版本变动）

**未跟踪（6 个目录，均为有意保留的产物）**：
```
data/  holdout/  result/  tests/results/  verification/
```

### 5.2 需要保留的数据 **[已验证：全部在磁盘，mtime 未变]**

| 路径 | 内容 | 体积 | Git 状态 | 迁移建议 |
|---|---|---|---|---|
| `data/raw/` | 12 篇原始 PDF | 30.9 MB | 已忽略 | **需迁移**（`tests/test_parser.py` 依赖） |
| `verification/` | 5 篇 PDF + 5 个 results JSON | 5.2 MB | PDF 忽略 / results 未跟踪 | **需迁移**（含评测结果） |
| `holdout/` | 5 篇 PDF + results + rerank_ab + search_ab | 5.1 MB | PDF 忽略 / results 未跟踪 | **需迁移**（含 holdout 基准） |
| `data/vector_db/` | Chroma 主库（13.9 MB sqlite + HNSW 索引） | 13.7 MB | 已忽略 | **可重建**，见 §9 风险 |
| `data/vector_db_22/` | Chroma 第二库 | 23.3 MB | 已忽略 | 同上 |
| `data/benchmark_results/` | 12 个评测 JSON | 24.8 KB | 未跟踪 | 建议迁移（含已知错误样本，迁移后需重跑） |
| `data/benchmark_results.xlsx` | 汇总表 | 6.6 KB | 未跟踪 | 同上 |
| `tests/results/` | 18 个（grade_eval 13 + review 5） | 78.1 KB | 未跟踪 | 建议迁移 |
| `result/` | 16 个图表 | 3.6 MB | 未跟踪 | 可选迁移 |
| `src_00/` | 16 个 .py 历史快照 | 305 KB | **已暂存** | 已在 Git 中，迁移后由 Git 恢复 |
| `poetry.lock` | 依赖锁 | 664 KB | 已跟踪 | 已在 Git 中 |
| `.venv/` | Windows 虚拟环境 | **1.3 GB** | 已忽略 | **不要迁移**，WSL 需重建 |

**ChromaDB 完整性确认**：`data/vector_db/chroma.sqlite3` mtime = **Aug 20 16:26**、`data/vector_db_22/chroma.sqlite3` mtime = **Sep 28 19:13** → **本轮工作未写入向量库**。

### 5.3 本轮已发生的文件修改（全部可回滚）

| 文件 | 改动 | 目的 | 回滚方式 |
|---|---|---|---|
| `pyproject.toml` | +2 依赖 | 补齐代码直接 import 但未声明的包（§4 D8） | `git checkout -- pyproject.toml` |
| `poetry.lock` | 171→184 | 配合上述依赖声明 | `cp D:/Python/poetry.lock.before-depfix.bak poetry.lock`（**备份在仓库外，迁移前请另存一份**） |
| `.gitignore` | 移除 6 条死规则 + 加说明注释 | 已删除的零字节垃圾文件不再需要忽略规则 | `git checkout -- .gitignore` |
| `.gitattributes` | 新增 | 统一 LF 换行 + 二进制资产保护 | `git rm --cached .gitattributes` |
| `src_00/**` | 16 个文件纳入暂存 | 历史快照纳入基线 | `git rm --cached -r src_00` |
| `.env.example` | 新增（无秘密值） | 环境变量模板 | `git rm --cached .env.example` |
| `.coverage` | 被 pytest 重新生成 | 测试副产物（已 gitignore） | 无需处理 |

**已删除的文件**（仅此 6 个，均为 0 字节、经严格校验后精确删除）：
`Fetch`、`Use`、`unset`、`在`、`如果`、`配置前请清除以下环境变量，它们的优先级高于配置文件，会覆盖`

---

## 6. Windows 环境搭建进度（实际执行过的命令与结果）

### 6.1 解释器选型：核实过程

| 候选 | 版本 | 结论 | 依据 |
|---|---|---|---|
| `D:/Miniconda/envs/myenv/python.exe` | 3.10.21 | ✗ 弃用 | 锁内 `numpy 2.5.1` 要求 `>=3.12` |
| `D:/Miniconda/base` | 3.9.1 | ✗ 弃用 | 低于 `requires-python >= 3.10` |
| `D:/msys64/ucrt64/bin/python.exe` | 3.12.11 | ✗ 弃用 | `sysconfig.get_platform()` = `mingw_x86_64_ucrt_gnu` → pip 只找 mingw 轮子，torch/numpy/chromadb 无此轮子；且该解释器**无 pip 模块** |
| `D:/Python/python.exe` | 3.14.0a5 | ✗ 弃用 | alpha 版、embeddable 布局 |
| conda 安装的 CPython | **3.12.15** | ✓ **采用** | `win-amd64` 轮子可用 |

### 6.2 实际执行过的命令（按顺序，全部成功）

```bash
# 1) 建立隔离环境
conda create -y -p "D:/program/OncoImmunoRAG/.venv" python=3.12 pip
#   → 成功，Python 3.12.15

# 2) 转换为标准 venv 布局（conda 的 python.exe 在根目录，非标准布局）
mv .venv "D:/Python/py312-base"
"D:/Python/py312-base/python.exe" -m venv --copies .venv
#   → 成功；.venv/Scripts/python.exe 存在，platform tag = win-amd64
#   → 基座放在 D:/Python/py312-base（非 conda 注册的普通目录，未触碰任何已有 conda env）

# 3) 工具链
.venv/Scripts/python.exe -m pip install --upgrade pip          # → pip 26.2.2
.venv/Scripts/python.exe -m pip install "poetry>=2.0,<3.0"     # → Poetry 2.5.1

# 4) 锁文件校验（修改前）
POETRY_VIRTUALENVS_CREATE=false .venv/Scripts/python.exe -m poetry check --lock
#   → "All set!"（锁与 pyproject 一致）

# 5) 按现有锁文件安装（未运行 poetry update）
POETRY_VIRTUALENVS_CREATE=false .venv/Scripts/python.exe -m poetry install
#   → 171 包安装完成；其中 25 个按平台/版本 marker 正确跳过

# 6) 最小依赖修复（2 个缺失声明）
#    修改 pyproject.toml 后：
POETRY_VIRTUALENVS_CREATE=false .venv/Scripts/python.exe -m poetry lock
#   → 171 → 184 包
POETRY_VIRTUALENVS_CREATE=false .venv/Scripts/python.exe -m poetry install
#   → 新增 13 包安装完成
POETRY_VIRTUALENVS_CREATE=false .venv/Scripts/python.exe -m poetry check --lock
#   → "All set!"

# 7) 验证
.venv/Scripts/python.exe -m pytest tests/ -o addopts="" -p no:cacheprovider -q
#   → 15 passed
.venv/Scripts/python.exe -m ruff check .          # → 248 errors
.venv/Scripts/python.exe -m ruff format --check .  # → 47 files would be reformatted
```

### 6.3 依赖修复的完整差异 **[已验证]**

`pyproject.toml` 新增（仅这两行 + 6 行注释）：
```toml
"llama-index-vector-stores-chroma>=0.6.0,<1.0.0",
"streamlit>=1.60.0,<2.0.0",
```

`poetry.lock` 差异：**+13 包 / −0 包 / 1 处版本变动**
- 新增 13：`llama-index-vector-stores-chroma` 0.6.0、`streamlit`（按 python marker 拆为 1.65.0 / 1.66.0 两条）、`altair`、`itsdangerous`、`jsonschema`、`jsonschema-specifications`、`pyarrow`、`pydeck`、`python-multipart`、`referencing`、`rpds-py`、`toml`、`watchdog`
- 版本变动 1：`nvidia-cusparse` 12.8.2.51 → 12.6.3.3 —— **Linux CUDA 包，Windows 上不安装，无运行时影响**
- 删除 0；`content-hash` 由 `4108f813…` 更新为 `b98cd5bf…`

### 6.4 安装状态核验 **[已验证]**

- 锁内 171 包 → **146 个版本完全一致 / 0 处偏差**
- 25 个正确跳过：18 个 Linux CUDA（`nvidia-*` / `cuda-*` / `triton`）、5 个 py<3.11 回移植包（`exceptiongroup`、`importlib-metadata`、`tomli`、`zipp`、`async-timeout`）、1 个非 Windows（`uvloop`）
- 额外 28 个为 Poetry 自身依赖（工具链，非项目依赖）
- 导入核验：`utils.config_loader` / `utils.prompts` / `utils.trial_registry` / `ingestion.table_extractor` / `ingestion.pdf_parser` / `ingestion.vector_store` / `ingestion.append_pdfs` / `retrieval.selector` / `retrieval.query_engine` / `retrieval.temp_indexer` / `retrieval.pubmed_client` / `grading.grade_evaluator` / `streamlit` —— **全部 import 成功**

---

## 7. WSL 迁移后需要重新配置的内容

### 7.1 必须重建（Windows 产物不可复用）

| 项 | 说明 |
|---|---|
| **`.venv/`（1.3 GB）** | Windows 虚拟环境，**不要复制**。WSL 需重新创建 |
| **基座解释器** | `D:/Python/py312-base` 是 Windows 构建，WSL 需自备 Python 3.12（Ubuntu 24.04 的 `python3.12` 可用） |
| **poetry** | WSL 侧需重新安装（`pipx install poetry` 或官方 installer） |
| **系统依赖** | Ubuntu 需 `build-essential`、`python3.12-dev`、`libpq-dev`（chromadb/pgvector 相关）、`git`、`curl`。**注意 chromadb 0.6.3 在 Linux 上会拉取 `onnxruntime` / `tokenizers` 等，轮子可用性需实测** |
| **嵌入模型权重** | `BAAI/bge-large-en-v1.5`（约 1.3 GB）从未下载过，WSL 首次实例化 `HuggingFaceEmbedding` 时会拉取 —— 请确认磁盘空间与网络 |
| **`.claude/settings.local.json`** | 被 gitignore，含 Windows 机器相关的权限白名单，**需在 WSL 侧重建**（可从 Claude Code 的 `/config` 重新生成） |

### 7.2 配置文件（内容与密钥）

`config.yaml` **当前不含任何密钥**，仅含非敏感项：
```yaml
embedding_model: "BAAI/bge-large-en-v1.5"
llm_model: "MiniMax-Text-01"
llm_base_url: "https://api.minimaxi.com/v1"
chunk_size: 1024
chunk_overlap: 200
```

`.env.example` 已存在且**不含秘密值**。`config.yaml` 中已无 `openai_api_key` 键。

### 7.3 API 凭据（**本文档不记录任何实际值**）

- **环境变量名**（由代码确认，`src/utils/config_loader.py:27`、`src/retrieval/query_engine.py:128,135`）：
  - `OPENAI_API_KEY` —— **优先级：环境变量 > config.yaml**（`config_loader.py:60-63`）
  - `LLM_BASE_URL` —— **优先级：config.yaml > 环境变量** > `https://api.openai.com/v1`
  - `LLM_MODEL` —— **优先级：config.yaml > 环境变量** > `gpt-4o-mini`
- **注入方式**：
  - WSL 侧用 `~/.bashrc` / `~/.zshrc` export，或 `direnv`，或 `systemd --user` 环境
  - **注意**：项目**没有** `python-dotenv` 依赖，**不会自动加载 `.env`**；若要沿用 `.env` 文件，需手动 `set -a; . ./.env; set +a`（`.env` 已被 gitignore）
- **务必在 WSL 侧重新注入凭据** —— 不要把密钥写进任何被 Git 跟踪的文件
- **MiniMax 侧旧 Key 轮换状态：未知，需项目负责人确认**（见 §9）

### 7.4 Claude Code 的 WSL 侧配置

- Claude Code 在 WSL 内运行需单独安装与配置（不是 Windows 侧配置的自动同步）
- `.claude/settings.local.json` 被 gitignore，WSL 侧需重建权限白名单
- 若从 Windows 侧调用 WSL 中的 Claude Code，注意两边的路径与 shell 差异
- **不要把含凭据的 shell 历史或环境转储写入仓库**

### 7.5 路径相关

- 代码中所有路径都通过 `resolve_config_path()`（`config_loader.py:38-42`）从 `__file__` 推导 `parents[2]`，**不依赖当前工作目录** → Windows → WSL 路径变化**不影响**代码逻辑
- `.gitattributes` 已设 `* text=auto eol=lf` → WSL clone 后工作区即为 LF，与 Windows 侧一致，**不会出现换行符冲突**
- ⚠️ `core.autocrlf=true` 是 Windows 本地配置，不随 Git 仓库迁移

---

## 8. 下一次开始工作的具体步骤

> **前提：不得先做算法重构或功能开发。** 以下步骤只做"确认状态 → 重建环境 → 跑现有测试 → 继续核实缺陷"。

### 步骤 1 — 核对迁移后的文件状态（不要跳过）

```bash
cd <WSL 路径>/OncoImmunoRAG

git status --short --branch          # 预期：18 个已暂存文件 + 2 个未暂存(pyproject.toml, poetry.lock) + 6 个未跟踪目录
git log --oneline -1                 # 预期：ced4543
git diff --cached --name-only | wc -l   # 预期：18
git diff --name-only                 # 预期：poetry.lock, pyproject.toml
git stash list                       # 预期：空
```

**如果暂存区内容与上表不符**（例如因复制目录而丢失了 index 状态）：
- **不要** `git reset --hard`、**不要** `git checkout .`、**不要** `git clean`
- 先用 `git diff --cached --name-only` 记录现状，再逐个 `git add` 显式补齐

### 步骤 2 — 建立隔离环境

```bash
python3.12 -m venv .venv          # 或 python3 -m venv .venv（需确认版本 ≥3.12）
.venv/bin/python -m pip install --upgrade pip
# poetry 建议装在系统级（pipx），避免污染项目 venv：
pipx install poetry                 # 或按 https://python-poetry.org 的官方 installer
poetry --version
```

**不要**把 poetry 装进项目 `.venv`（Windows 侧这样做导致 urllib3 等被 poetry 的依赖约束牵动，且多出 28 个工具包）。

### 步骤 3 — 用现有锁文件安装依赖

```bash
poetry config virtualenvs.in-project true --local   # 会写 pyproject.toml / poetry.toml，注意这会改动文件
poetry install                                         # 使用现有 poetry.lock，不执行 poetry update
poetry check --lock                                   # 预期 "All set!"
```

**若 `poetry install` 报锁文件与 pyproject 不一致** —— 说明 `pyproject.toml` 的 2 行依赖改动未随 `poetry.lock` 一起迁移，或反之。处理顺序：
1. 确认 `poetry.lock` 的 `content-hash` 是否为 `b98cd5bfaabc0ef6d5db9f841f747405af499d6aeb44116cbc29b9dca046af2d`
2. 若不是，先备份 `poetry.lock`，再执行 `poetry lock` 重算
3. **不要**用 `poetry update`（会升级既有包，掩盖问题）

**迁移前后 Linux / Windows 的锁文件差异是正常的** —— `poetry lock` 会按平台重算 marker。若在 WSL 重算 lock，**先备份 Windows 版**，并对比新增/删除的包是否只是平台相关。

### 步骤 4 — 运行现有离线测试（不需任何凭据）

```bash
export PYTHONPATH="$PWD/src"
.venv/bin/python -m pytest tests/ -o addopts="" -p no:cacheprovider -q
#   预期：15 passed（14 个 test_table_extractor + 1 个 test_parser）

.venv/bin/python -m pytest tests/test_table_extractor.py tests/test_parser.py -q
#   注意：这一条会触发 addopts 里的 --cov=oncoimmuno_rag 并打印 3 条 warning（§4 D14），
#         属已知现象，不是新问题。

.venv/bin/python -m ruff check . --output-format=concise | tail -2
.venv/bin/python -m ruff format --check . | tail -1
#   预期与 Windows 侧一致：248 errors / 47 files would be reformatted
```

**若 `test_parser.py` 失败** → 说明 `data/raw/` 的 12 篇 PDF 未随迁移带过来（它读 `sorted(raw_dir.glob("*.pdf"))[0]`）。

### 步骤 5 — 继续核实缺陷（只确认，不修改）

按优先级复核，每条都是**只读或纯计算**操作：

1. **D1**：`python -c "from retrieval.query_engine import QueryEngine; print(hasattr(QueryEngine,'query'))"` → 预期 `False`
2. **D4 / D5**：调用 `_grade_to_bucket` 与 `_compute_composite_score_breakdown` 复核阈值与触发器(A) 数值
3. **D6**：对 `data/raw/` 12 篇跑 `_classify_section` 统计 `section_type` 分布 → 预期 `primary` ≈ 6.5%
4. **D7**：用 `retrieval.selector.select_top_k` + 合成 hit 复核单篇占比 → 预期 100%
5. **D15**：`load_config("<不存在的路径>", raise_if_missing=False)` → 预期 `ValueError`
6. **D2 / D3 / D8 / D10 / D12 / D13 / D14**：均为 grep / import 级复核，逐条对照 §4

**需要凭据或网络才能验证的**（暂缓，待批准）：D11 的触发频率、§3.2 表中的全部条目。

### 步骤 6 — 暂不执行

- ❌ 不做算法重构
- ❌ 不修复 §4 中的任何缺陷
- ❌ 不运行完整 benchmark / 批量 LLM 分级（会覆盖 `tests/results/`、`data/benchmark_results/`）
- ❌ 不重建 ChromaDB（除非 §9 决定需要）
- ❌ 不 commit / push

---

## 9. 未决事项与迁移风险

### 9.1 必须由项目负责人确认

| # | 事项 | 说明 |
|---|---|---|
| 1 | **MiniMax 旧 Key 是否已轮换** | 仓库侧已无明文密钥（`config.yaml` 已移除 `openai_api_key`，全仓扫描 0 命中）；但**旧 Key 曾以明文存在于 `config.yaml`**，轮换与否只能在控制台核实。**WSL 侧需注入新 Key**。本文档不记录任何密钥值 |
| 2 | **`pyproject.toml` + `poetry.lock` 的改动是否接受** | 当前为未暂存状态。`poetry.lock` 差异为纯增量（+13 包 / −0 包 / 1 个不安装的 Linux CUDA 包版本变动）。若要回滚，需 `poetry.lock.before-depfix.bak`（**当前在 `D:/Python/`，迁移前请另存**） |
| 3 | **是否提交本轮改动** | 暂存区 18 个文件 + 未暂存 2 个文件均未提交。是否合并为一次提交 |
| 4 | **是否需要真实凭据的端到端验证** | 需要：① 确认旧 Key 已轮换并提供新 Key（仅经环境变量注入）② 允许下载 1.3 GB 嵌入模型 ③ 允许调用 LLM。这是验证 D1/D2/D3 的唯一途径 |
| 5 | **22 篇论文中 `verification/` + `holdout/` 的 10 篇是否应入向量库** | 现状只索引了 `data/raw` 的 12 篇，但 UI 宣称"22 篇库内"（D22） |
| 6 | **`src_00/` 的长期定位** | 已按决定纳入基线。是保留为历史参考，还是后续归档/删除？注意其 `query_engine.py` 带与 `src/` 相同的 D1 缺陷，**运行应用务必用 `src/`** |
| 7 | **README 三处偏差（D19）的处理时机** | 按当前指示保留为后续任务 |
| 8 | **"量化 GRADE" 是否允许对外表述为"GRADE 证据等级"** | 这需要医学循证专家裁定（自定义 0–10 分 ≠ 官方 GRADE 确定性），不是工程决策 |

### 9.2 迁移风险

| 风险 | 等级 | 说明与缓解 |
|---|---|---|
| **Linux 轮子可用性** | 中 | chromadb 0.6.3 在 Linux 会拉 `onnxruntime`、`tokenizers` 等；Windows 侧装的是 win_amd64 轮子，Linux 侧是 manylinux 轮子，可能缺版本。**缓解**：步骤 3 若失败，先看具体错误，不要用 `poetry update` 掩盖 |
| **lock 文件跨平台重算** | 中 | 在 WSL 执行 `poetry lock` 会按平台重算 marker，可能与 Windows 版产生差异。**缓解**：先备份 Windows 版 lock，对比差异应仅为平台相关包 |
| **ChromaDB 跨平台可移植性** | 中 | `data/vector_db/chroma.sqlite3` + HNSW 索引是平台无关的二进制，**理论上可复制**；但嵌入模型换了运行环境后向量仍有效（同一模型）。**建议：迁移初期不复制，用 `python -m ingestion.vector_store` 重建并对比 chunk 数**。若重建，`reset=True` 会 `shutil.rmtree` 整个目录 —— **务必先备份** |
| **`test_parser.py` 依赖本地 PDF** | 低 | 需 `data/raw/*.pdf` 随迁；缺失则该测试失败 |
| **`.claude/settings.local.json` 不迁移** | 低 | 被 gitignore，WSL 侧需重建权限白名单 |
| **凭据未随迁** | 低（预期） | 必须在 WSL 侧重新注入；不要写进任何被跟踪文件 |
| **`poetry.lock` 中 nvidia-* 包** | 低 | Windows 上跳过、Linux 上可能安装（`nvidia-cusparse` 等 manylinux 轮子）。若不需要 GPU，**可考虑加 `tool.poetry` 的平台 marker 排除**，但这属于额外改动，本轮未做 |
| **`.venv` 若被误复制** | 低 | 1.3 GB 且不可用。**不要复制**；如已复制请删除后重建 |
| **落盘评测结果含已知错误** | 中（数据质量） | `tests/results/` 与 `data/benchmark_results/` 中的 JSON 含可核实的临床事实错误（§3.1 末行）。**这些不是可靠基准**；修复缺陷并重跑之前，不要把它们当作"正确答案"或训练/验收依据 |

### 9.3 已知的环境侧遗留

- `.venv` 内含 Poetry 自身的 28 个依赖包（若在 WSL 改用 pipx 安装 poetry，`.venv` 会更贴近锁文件）
- `poetry.lock` 中 `nvidia-cusparse` 版本被动过一次（12.8.2.51 → 12.6.3.3），属无运行时影响的平台包
- `.coverage` 已被 pytest 重新生成（已 gitignore）

---

## 10. 附：给接手者的三点提醒

1. **跑测试不需要任何凭据、不需要下载模型。** `tests/test_table_extractor.py`（14 个）全是内联字符串字面量 + 纯 Python selector；`tests/test_parser.py` 只需本地 PDF。**唯一会拉起 torch 的是 `tests/test_new_deps_smoke.py` 的 import**（它没有测试函数，但 import 失败会导致整个 pytest run 收集阶段 error）。

2. **`grade_evaluator.py`（1587 行）与 `query_engine.py`（587 行）目前零测试覆盖。** 整个 `src/` 只有 3 个模块有真实覆盖（`table_extractor`、`pdf_parser` 部分、`selector` 部分）。**任何对评分的改动都没有回归网** —— 这是在动手改算法之前最该先补的东西。

3. **本项目当前形态与产品目标差距很大，且差距是结构性的。** 产品目标 2（RoB 2）与目标 3（按结局组织 GRADE）**没有代码基础**，需要从零设计并由医学循证专家参与裁定语义；目标 5（跨研究综合）的入口是死的。**在补齐这些之前，任何"证据等级"输出都应视为研究原型，不能直接用于临床沟通。**

---

*本文件由 Claude Code 生成，记录截至 2026-10-21 的实际状态。文件内不含任何 API Key、Token、密码或其他凭据。*
