# ARCHITECTURE.md — OncoImmunoRAG

> 本文档描述**当前代码的真实状态**，不是目标架构，也不描述尚不存在的能力。
> 每条陈述都可对照 `git ls-files` 中的源码验证。已知缺陷见 `CURRENT_ISSUES.md`。

---

## 1. 系统形态

单体 Streamlit 应用 + 4 个纯 Python 包。**无 HTTP API、无数据库服务、无前端构建、无容器化**。

```
┌──────────────────────────────────────────────┐
│  浏览器 http://localhost:8501                 │
└──────────────────┬───────────────────────────┘
                   │ Streamlit 1.66
┌──────────────────▼───────────────────────────┐
│  app.py (848 行)                              │
│  UI 渲染 · 业务编排 · 静态文献表 · 会话缓存    │
└──────────────────┬───────────────────────────┘
                   │  模块级 from X import Y
   ┌───────────────┼───────────────┬──────────────┐
   ▼               ▼               ▼              ▼
 grading/      retrieval/      ingestion/     utils/
               query_engine    pdf_parser    config_loader
               selector        table_extractor  prompts
               temp_indexer    vector_store  trial_registry
               pubmed_client   append_pdfs
                   │
                   └──► grading.grade_extractor  ⚠️ 分层倒置
                              │
                              ▼
              ┌───────────────────────────────┐
              │ utils.config_loader           │
              │ （配置读取 + 数据目录定位）    │
              └───────────────────────────────┘
                   │
                   ▼
        config.yaml / 环境变量 / data/vector_db
                   │
                   ▼
        OpenAI 兼容端点 (MiniMax) · ChromaDB · NCBI E-utilities
```

## 2. 模块依赖关系（Mermaid）

```mermaid
graph TD
    APP["app.py<br/>848 行"]

    GE["grading.grade_evaluator<br/>1587 行"]
    QE["retrieval.query_engine<br/>587 行"]
    SEL["retrieval.selector<br/>502 行"]
    TI["retrieval.temp_indexer<br/>279 行"]
    PC["retrieval.pubmed_client<br/>299 行"]
    PP["ingestion.pdf_parser<br/>340 行"]
    TE["ingestion.table_extractor<br/>337 行"]
    VS["ingestion.vector_store<br/>174 行"]
    AP["ingestion.append_pdfs<br/>164 行"]
    CL["utils.config_loader<br/>149 行"]
    PR["utils.prompts<br/>48 行"]
    TR["utils.trial_registry<br/>207 行"]

    APP --> GE
    APP --> QE
    APP --> SEL
    APP --> TI
    APP --> PC
    APP --> CL

    QE --> SEL
    QE --> CL
    QE --> PR
    QE -.->|"懒加载"| TR

    GE --> CL
    GE --> PR
    GE --> TR

    AP --> PP
    AP --> VS
    VS --> PP
    VS --> CL
    AP --> CL
    PP --> TE
    PP --> CL
    TE -.->|"函数内延迟导入<br/>规避循环"| PP

    TI --> GE
    PC --> REQ["requests<br/>(声明于 pyproject)"]

    classDef warn fill:#f8d7da,stroke:#c82333,color:#000
    classDef stub fill:#e2e3e5,stroke:#6c757d,color:#000,stroke-dasharray: 4 3
    class TI warn
    class EV stub
```

**⚠️ `src/evaluation/` 是空壳包**（仅 `__init__.py` 中的 `__all__ = []`，无任何代码）。README 宣称的 "Evaluation harness" 不存在，故图中不绘制。

### 已知结构问题（记录，未修复）

| 问题 | 位置 |
|---|---|
| 分层倒置：检索层依赖评分层 | `retrieval/temp_indexer.py` → `grading/grade_evaluator.py` |
| 路径解析惯例重复 5 次 | `resolve_config_path().parent / "data" / X`，见 grade_evaluator:242、append_pdfs:90、pdf_parser:160、vector_store:42、query_engine:102 |
| 第 6 种写法，绕过 config_loader | `table_extractor.py:317,321` 用 `Path(__file__).resolve().parents[2]` |
| 陈旧源码副本 | `src_00/`（缺 2 模块，selector 少 225 行），已 gitignore |

---

## 3. 模块职责

| 模块 | 唯一职责 | 不负责 |
|---|---|---|
| `app.py` | 渲染 UI、编排调用、维护会话状态 | 不做检索算法、不做评分规则 |
| `utils/config_loader.py` | 读取 `config.yaml` + 环境变量，返回 `AppConfig`；定位 `data/` 目录 | 不做业务判断 |
| `utils/prompts.py` | 提供提示词文本片段 | 不发请求 |
| `utils/trial_registry.py` | 手工维护的研究设计覆盖表，**在评分时覆盖 LLM 输出** | 不发请求 |
| `ingestion/pdf_parser.py` | 加载 PDF、切块、附加表格摘要 | 不写数据库 |
| `ingestion/table_extractor.py` | 从 chunk 中抽取表格并生成摘要 | 不切块 |
| `ingestion/vector_store.py` | 构建/写入 ChromaDB 索引 | 不解析 PDF |
| `ingestion/append_pdfs.py` | 增量入库编排 | 不做检索 |
| `retrieval/query_engine.py` | 检索编排：MMR 超采样 → canonical 补召回 → rerank → select_top_k | 不发 LLM 请求（**其 `query()` 因缩进错误不可用**） |
| `retrieval/selector.py` | 证据等级 rerank + top_k 选取 | 不做嵌入、不连库 |
| `retrieval/temp_indexer.py` | 会话级临时 collection，支持上传 PDF 即时检索 | 不做评分规则（但会调用 grade_extraction） |
| `retrieval/pubmed_client.py` | NCBI E-utilities 检索与详情拉取 | 不做评分 |
| `grading/grade_evaluator.py` | 调用 LLM 产出 `GRADEResult` 并按 Python 规则重算综合分 | 不检索 |

---

## 4. 数据流

### 4.1 已实测跑通的主链路

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as app.py
    participant P as pdf_parser
    participant T as table_extractor
    participant G as grade_evaluator
    participant L as MiniMax 端点
    participant R as trial_registry

    U->>A: 选择文献 → 点击「评估」
    A->>P: load_pdfs(file_path)
    P->>P: pypdf 逐页 extract_text
    P->>T: extract_table_summaries(docs)
    T-->>P: 表格摘要 Document
    P-->>A: chunk_documents(...) 文本块
    A->>G: grade_extraction(chunks, cfg, canonical_file_name)
    G->>R: resolve_study_design(file_name, llm_default)
    R-->>G: 覆盖后的 study_design
    G->>L: chat.completions（提示词 + 规则）
    L-->>G: JSON（13 字段）
    G->>G: Python 规则重算 composite_score<br/>⚠️ 覆盖 LLM 分数（ISSUE-013）
    G-->>A: GRADEResult
    A-->>U: 渲染等级/得分/徽章
```

**实测耗时**：UI 端到端 10.3s；直接调用 `grade_extraction()` 15.2s。

### 4.2 检索链路

```mermaid
flowchart TD
    Q[用户问题] --> OF["over_fetch = min(max(top_k*5,25),80)"]
    OF --> MMR["retriever: MMR, threshold=0.5<br/>similarity_top_k=over_fetch"]
    MMR --> CC{"canonical_file<br/>已提供?"}
    CC -->|是| CAN["_retrieve_canonical_chunks(top_n=6)<br/>定向补召回目标文献"]
    CC -->|否| RR
    CAN --> RR["rerank_hits(enabled, registry_lookup)"]
    RR --> SEL["select_top_k(raw, self._top_k, canonical_file)"]
    SEL --> OUT["list[NodeWithScore]<br/>元数据在 .metadata 属性"]
```

**⚠️ 关键约束（实测）**：
- `top_k` **不是 `retrieve()` 的参数**，是构造时固定的 `self._top_k`（默认 **5**）。传入 `top_k=` 抛 `TypeError`。
- 返回值是 **llama-index 原生 `NodeWithScore`**，元数据通过 `.metadata` 属性访问，**不支持 `.get()` 字典访问**。
- ⚠️ 模块内存在同名自定义类 `_NodeWithScore`（`query()` 因缩进错误被埋在其中），**与返回值不是同一个类**。

### 4.3 入库链路（离线脚本，不在 Web 运行时执行）

```mermaid
flowchart LR
    PDF[data/raw/*.pdf] --> P[pdf_parser.parse_pdfs]
    P --> T[table_extractor]
    T --> V[vector_store.build_index]
    V --> C[(data/vector_db<br/>collection: oncoimmuno_rag<br/>406 chunks / 12 篇)]
    C -.->|"元数据 file_path 烘焙<br/>D:\OncoImmunoRAG\..."| W[⚠️ 跨平台失效]
```

---

## 5. 关键数据结构

### `GRADEResult`（pydantic，`grade_evaluator.py`）

13 字段，**全部必填，除最后两个**：

| 字段 | 类型 |
|---|---|
| `risk_of_bias` / `inconsistency` / `indirectness` / `imprecision` / `publication_bias` | `Literal['高','中','低']` |
| `confounding_by_subsequent_therapy` | `Literal['是','否','未报告']` |
| `study_design` | `Literal['RCT','观察性','系统评价','Meta分析','单臂试验']` |
| `sample_size` | `int` |
| `effect_size` | `str` |
| `overall_grade` | `Literal['高','中','低','极低']` |
| `grade_reasoning` | `str` |
| `composite_score` | `float`（可选） |
| `score_breakdown` | `str`（可选） |

⚠️ 5 个偏倚维度是**裸字符串**，不是嵌套对象。`scripts/` 下 7 个绘图脚本消费这些 JSON。

⚠️ **`study_design` 的受控词表是 `单臂试验`**，而 `app.py` 的 `design_gold` 用 `单臂` —— 两套词表，同一概念不同字面量。

### 双真相源（⚠️ 已知漂移风险）

| 位置 | 规模 | 是否参与评分 |
|---|---|---|
| `app.py:70-93` `_PAPERS_RAW` | 22 条 | ❌ 仅显示 |
| `utils/trial_registry.py:69` `TRIAL_REGISTRY` | 6 条 | ✅ **覆盖 LLM 输出** |

`design_gold` 全仓仅 2 处引用，均在 `app.py`（赋值 + 第 372 行显示），**从不与 LLM 输出比对** —— 界面上的「金标准设计」是装饰性的。

### 其他数据结构

均为 `dataclass`，字段见 `docs/interfaces.md`：`QueryResult`、`TempIndexResult`、`ExtractedTable`、`PubMedHit`、`PubMedDetail`、`TrialInfo`、`AppConfig`。

---

## 6. 外部依赖

| 类型 | 依赖 | 缺失后果 |
|---|---|---|
| 嵌入模型 | `BAAI/bge-large-en-v1.5`（1.3 GB，HF 托管） | 检索完全不可用；首次需联网 |
| LLM 端点 | OpenAI 兼容 API（`config.yaml` 的 `llm_base_url`） | 评分不可用 |
| 凭据 | 环境变量 `OPENAI_API_KEY` | `get_query_engine()` 返回 `None`，主链路静默失效 |
| 向量库 | 本地 ChromaDB `data/vector_db` | 检索不可用 |
| 论文 | 22 个 PDF（`data/raw` 12 + `verification` 5 + `holdout` 5） | Tab 1/2/3 不可用 |
| 外部服务 | NCBI E-utilities | 仅 Tab 2 的 PubMed 检索不可用 |

**未声明但被直接 import 的依赖**：`requests`（`pubmed_client.py`）—— 2026-10-23 已补入 `pyproject.toml`。

**缺失依赖**：`matplotlib`、`plotly`、`seaborn` —— `scripts/` 下 **8 个绘图脚本当前无法运行**。

---

## 7. 程序启动流程

```mermaid
flowchart TD
    S["streamlit run app.py"] --> P1["app.py:38 st.set_page_config"]
    P1 --> P2["app.py:56-64 sys.path 注入 src/ 后导入 6 个模块"]
    P2 --> P3["定义 _PAPERS_RAW（22 条静态表）"]
    P3 --> P4["@st.cache_resource get_query_engine()<br/>⚠️ 无 API Key → 直接返回 None"]
    P4 --> P5["渲染标题 · 2 个 st.metric 徽章 · 3 个 Tab"]
    P5 --> P6{"用户操作"}
    P6 -->|"选文献 + 点评估"| R1["grade_paper → grade_extraction"]
    P6 -->|"PubMed 检索"| R2["search_pubmed → 临时索引 → 评分"]
    P6 -->|"上传 PDF"| R3["index_temp_document"]
```

`get_query_engine()` 是**惰性**的：仅在需要时构造，且**在无 API Key 时直接返回 `None`**，不构造 `QueryEngine`、不加载嵌入模型。