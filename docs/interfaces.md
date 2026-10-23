# docs/interfaces.md — 接口契约

> 本文档记录**现有公共接口的真实签名**（由 AST 提取，非人工推测）。
> 契约描述现状，**不重写实现**。修改接口前必须先按本文档「调用方清单」核查兼容性影响。

---

## 通用约定

| 项 | 约定 |
|---|---|
| 包导入根 | `src/`（由 `pyproject.toml` 的 `pythonpath = ["src"]` 与 `app.py` 的 `sys.path` 注入共同保证） |
| 类型标注 | 覆盖率 96%，公共函数均有标注 |
| 异常 | 全仓**无裸 `except:`**；`except Exception` 共 18 处，`app.py` 占 10 |
| 日志 | ⚠️ **全仓 0 处使用 `logging`**，全部用 `print()`（47 处）。测试无法捕获输出 |
| 随机种子 | ⚠️ **全仓 0 处设置**，涉及采样的结果不可复现 |
| 破坏性修改 | 当前**无版本化契约**。任何签名变更视为破坏性，需项目负责人批准 |

---

## utils.config_loader

**唯一职责**：读取 `config.yaml` + 环境变量，返回 `AppConfig`；定位项目根与 `data/` 目录。
**依赖**：无内部依赖。**被 5 个模块依赖**，是事实上的共享内核。

| 符号 | 签名 |
|---|---|
| `resolve_config_path` | `(path: str \| os.PathLike[str] \| None = None) -> Path` |
| `load_config` | `(path: str \| os.PathLike[str] \| None = None, *, raise_if_missing: bool = True) -> AppConfig` |
| `AppConfig.effective_openai_api_key` | `(self) -> str` — **方法** |
| `AppConfig.has_openai_api_key` | **`@property` → `bool`** — ⚠️ 是属性不是方法，调用它抛 `TypeError: 'bool' object is not callable` |

`AppConfig` 字段：`openai_api_key: str`、`embedding_model: str`、`chunk_size: int`、`chunk_overlap: int`、`extras: dict[str, Any] = {}`

**优先级（实测确认）**
- `OPENAI_API_KEY`：环境变量 **>** `config.yaml`
- `llm_base_url` / `llm_model`：**`config.yaml` >** 环境变量（改 `.env` 中这两项**无效**）
- 二者最终进入 `AppConfig.extras`（实测键名：`llm_base_url`、`llm_model`）

**⚠️ 已知缺陷**
- `load_config(raise_if_missing=False)` **仍然抛 `ValueError`**（参数未生效）→ ISSUE-015
- `AppConfig` **没有 `llm_base_url` 属性**，只能经 `.extras` 取 → 文档易误导
- 无凭据时加载即产生 `UserWarning`（非异常）

**副作用**：读 `config.yaml` 文件。
**对应测试**：`tests/test_config_smoke.py`（⚠️ **当前贡献 0 个用例**，只有 `main()`）。

---

## utils.trial_registry

**唯一职责**：手工维护的研究设计覆盖表，在评分时**覆盖 LLM 输出**。
**依赖**：无。

| 符号 | 签名 |
|---|---|
| `lookup` | `(file_name: str) -> Optional[TrialInfo]` |
| `resolve_study_design` | `(file_name: str, llm_default: str) -> tuple[str, str]` |
| `canonical_sample_size` | `(file_name: str) -> int` |
| `TRIAL_REGISTRY` | `Mapping[str, TrialInfo]`，当前 **6 条** |

`TrialInfo` 字段：`trial_name: str`、`trial_type: str`、`study_design: str`、`sample_size: int = 0`、`nct_id: str = ''`、`notes: str = ''`、`multiple_primary_endpoints: bool = False`

**约定**：key 是 chunk 的 `file_name` 元数据值（**含扩展名**）。**未命中不报错**，回退到调用方给的默认值 —— 覆盖表是「增量式」的。

**⚠️ 已知缺陷**
- `study_design` 用 `单臂试验`，`app.py` 的 `design_gold` 用 `单臂` —— **两套受控词表**
- 只覆盖 6 篇，另外 16 篇依赖 LLM 判断
- **零测试覆盖**

---

## utils.prompts

**唯一职责**：提供提示词文本片段。纯字符串函数，**不发请求**。

| 符号 | 签名 |
|---|---|
| `statistical_significance_rules` | `() -> str` |
| `discussion_section_rule` | `() -> str` |

**⚠️ 与 Python 侧规则引擎不一致**：`statistical_significance_rules()` 内嵌的量化阈值（8.5/5.0/2.6/0）与 `grade_evaluator` 的 Python 阈值（8.5/**6.0**/**3.5**/0）**双轨** → ISSUE-013，实测 `LLM=7.0 / Python=2.0`。

---

## ingestion.pdf_parser

**唯一职责**：加载 PDF、切块、附加表格摘要。**不写数据库**。

| 符号 | 签名 |
|---|---|
| `load_pdfs` | `(data_dir: str \| Path \| None = None, files: list[str \| Path] \| None = None) -> list[Document]` |
| `chunk_documents` | `(docs: list[Document], chunk_size: int \| None = None, chunk_overlap: int \| None = None, *, include_tables: bool = True) -> list[Document]` |
| `parse_pdfs` | `(data_dir: str \| Path \| None = None, files: list[str \| Path] \| None = None, config: AppConfig \| None = None, *, include_tables: bool = True) -> tuple[list[Document], list[Document]]` |

`__all__ = ["load_pdfs", "chunk_documents", "parse_pdfs", "main"]`

**约定**：`data_dir=None` 时默认 `resolve_config_path().parent / "data" / "raw"`。
**副作用**：读 PDF 文件（`pypdf` 逐页 `extract_text`）；异常页面**静默吞掉**（`except Exception: txt = ""`）。
**测试**：`tests/test_parser.py`（**1 个用例**，覆盖不足）

---

## ingestion.table_extractor

**唯一职责**：从 chunk 中抽取表格并生成摘要。不切块。

| 符号 | 签名 |
|---|---|
| `extract_table_summaries` | `(file_docs: Sequence[Document]) -> list[Document]` |
| `ExtractedTable` | dataclass：`label`、`title`、`markdown`、`endpoints: tuple[str,...]`、`column_count: int`、`is_supplementary: bool`、`raw_body: str` |

**依赖**：`table_extractor` → `pdf_parser` 是**函数内延迟导入**（第 319 行），用于规避循环依赖。**不是**模块级循环导入。
**⚠️ 路径惯例不一致**：第 317、321 行用 `Path(__file__).resolve().parents[2]` 定位项目根，**绕过 `config_loader`**。
**测试**：`tests/test_table_extractor.py`（**14 个用例**，当前测试基线主体）

---

## ingestion.vector_store

**唯一职责**：构建/写入 ChromaDB 索引。

| 符号 | 签名 |
|---|---|
| `build_index` | `(*, reset: bool = True, collection_name: str = _DEFAULT_COLLECTION_NAME) -> tuple[VectorStoreIndex, list[Document]]` |
| `run_self_test` | `(index: VectorStoreIndex) -> None` |
| `_DEFAULT_COLLECTION_NAME` | `"oncoimmuno_rag"` |

**⚠️ 副作用（高危）**：`reset=True`（**默认值**）会调用 `_reset_vector_db(path)` **删除并重建 `data/vector_db`**。调用前必须确认。
**数据库位置**：`resolve_config_path().parent / "data" / "vector_db"`

---

## retrieval.selector

**唯一职责**：证据等级 rerank + top_k 选取。**不做嵌入、不连库**。

| 符号 | 签名 |
|---|---|
| `rerank_hits` | `(hits: Iterable[object], *, enabled: bool = True, registry_lookup=None) -> List[object]` |
| `select_top_k` | `(hits: Iterable[object], top_k: int = _DEFAULT_TOP_K, *, canonical_file: Optional[str] = None) -> List[object]` |
| `get_rerank_enabled_default` | `() -> bool` |
| `set_rerank_enabled_default` | `(enabled: bool) -> None` |

**模块级全局状态**：`rerank_hits` 由 `get/set_rerank_enabled_default` 控制的**进程级开关**。测试中修改会影响同进程内其他用例 → **必须恢复原值**。

**⚠️ 已知缺陷**
- `per_file_cap` 被回填循环**完全绕过**：top_k=5/8/12/25 实测单篇占比均 **100%** → ISSUE-020
- 返回的 `NodeWithScore` 无类型标注，且**没有 `.get()`**，元数据在 `.metadata`
- **零测试覆盖**

---

## retrieval.query_engine

**唯一职责**：检索编排。**不发 LLM 请求**（其 `query()` 不可用，见下）。

| 符号 | 签名 |
|---|---|
| `QueryEngine.from_config` | `(cls, cfg) -> QueryEngine` |
| `QueryEngine.retrieve` | `(self, question: str, *, canonical_file: str \| None = None) -> list` |
| `QueryEngine.set_rerank_enabled` | `(self, enabled: bool) -> None` |
| `QueryEngine.get_rerank_enabled` | `(self) -> bool` |
| `QueryResult` | dataclass：`sources: list[str]`、`source_nodes: object` |

**⚠️ 调用方必须知道的四条约束**

1. **没有 `query()` 方法。** 实测公开方法仅 `['from_config','get_rerank_enabled','retrieve','set_rerank_enabled']`。`query` 误缩进在私有类 `_NodeWithScore` 内 → ISSUE-008，跨研究综合**不可用**。
2. **`top_k` 不是参数。** 它是构造时固定的 `self._top_k`（默认 **5**）。传 `top_k=` 抛 `TypeError`。
3. **返回值是 llama-index 原生 `NodeWithScore`**，元数据在 `.metadata` 属性，**不支持 `.get()`**。与模块内的私有类 `_NodeWithScore` **同名不同物**。
4. **`from_config` 成本高**：冷启动加载 1.3 GB 嵌入模型，实测 **58.8s**。测试中应 mock。

**构造副作用**：连接 ChromaDB、加载嵌入模型、可能联网下载。
**零测试覆盖**。

---

## retrieval.temp_indexer

**唯一职责**：会话级临时 collection，支持上传 PDF 即时检索。

| 符号 | 签名 |
|---|---|
| `TempIndexer.open` | `(self) -> None` |
| `TempIndexer.close` | `(self) -> None` |
| `TempIndexer.add_text` | `(self, text: str, *, metadata: dict \| None = None, file_name: str \| None = None) -> list[str]` |
| `TempIndexer.retrieve` | `(self, question: str, top_k: int = 5) -> list[str]` |
| `index_temp_document` | `(text: str, metadata: dict \| None = None, *, file_name: str \| None = None, chunk_size: int = _DEFAULT_CHUNK_SIZE, chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP) -> TempIndexResult` |
| `clear_temp_collection` | `() -> None` |

`TempIndexResult`：`chunks: list[str]`、`collection_name: str = ''`、`file_name: str = ''`、`metadata: dict`

**⚠️ 分层倒置**：本模块（检索层）导入 `grading.grade_evaluator`（评分层）。
**副作用**：读写临时 Chroma collection。`clear_temp_collection` **删除**该 collection。

---

## retrieval.pubmed_client

**唯一职责**：NCBI E-utilities 检索与详情拉取。不做评分。

| 符号 | 签名 |
|---|---|
| `search_pubmed` | `(query: str, *, max_results: int = 10, mindate: str = '2021/01/01', maxdate: str = '2026/12/31', field_filter: str \| None = 'Title/Abstract', lang: str = 'eng', fetch_summaries: bool = True) -> list[PubMedHit]` |
| `fetch_details` | `(pmid: str, *, sleep_seconds: float = 0.4) -> PubMedDetail` |
| `PubMedHit.short_label` | `(self) -> str` |

`PubMedHit`：`pmid: str`、`title: str = ''`、`authors: list[str]`、`journal: str = ''`、`year: str = ''`、`abstract: str = ''`
`PubMedDetail`：`pmid`、`title`、`authors`、`journal`、`year`、`abstract`、`mesh_terms: list[str]`、`publication_types: list[str]`、`doi: str = ''`

**副作用**：**真实网络请求**到 `https://eutils.ncbi.nlm.nih.gov/entrez/eutils`。`fetch_details` 有 0.4s 节流。**测试必须 mock**。
**直接依赖 `requests`**（2026-10-23 已补入 `pyproject.toml`）。
**零测试覆盖**。

---

## grading.grade_evaluator

**唯一职责**：调用 LLM 产出 `GRADEResult`，并按 Python 规则重算综合分。

| 符号 | 签名 |
|---|---|
| `grade_extraction` | `(chunks: Sequence[str], *, llm: OpenAI \| None = None, cfg: AppConfig \| None = None, canonical_file_name: str \| None = None, chunk_file_names: Sequence[str] \| None = None) -> GRADEResult` |
| `GRADEResult` | pydantic，13 字段（见下） |

**`GRADEResult` 字段**

| 字段 | 类型 | 必填 |
|---|---|---|
| `risk_of_bias` / `inconsistency` / `indirectness` / `imprecision` / `publication_bias` | `Literal['高','中','低']` | ✅ |
| `confounding_by_subsequent_therapy` | `Literal['是','否','未报告']` | ✅ |
| `study_design` | `Literal['RCT','观察性','系统评价','Meta分析','单臂试验']` | ✅ |
| `sample_size` | `int` | ✅ |
| `effect_size` | `str` | ✅ |
| `overall_grade` | `Literal['高','中','低','极低']` | ✅ |
| `grade_reasoning` | `str` | ✅ |
| `composite_score` | `float` | ❌ |
| `score_breakdown` | `str` | ❌ |

**⚠️ 5 个偏倚维度是裸字符串，不是嵌套对象**（实测 `risk_of_bias == "高"`）。`scripts/` 下 7 个绘图脚本消费这些 JSON。

**⚠️ 已知缺陷（全部未修复）**
- `effect_size` **零 Python 侧校验**，别篇 HR 直接参与评分 → ISSUE-014
- 量化阈值双轨，实测 `LLM=7.0 / Python=2.0`，Python 覆盖 → ISSUE-013
- `_normalize_risk` 把「未提及/未知/未报告/不适用/n/a/空」**全部塌缩为「中」** → ISSUE-015
- 触发器(A) 规格 −3.0、实现 −1.5
- 1587 行，**零测试覆盖**

**副作用**：真实 LLM 网络请求（实测 15.2s）；可选写 `data/vector_db`。
**依赖**：`config_loader`、`prompts`、`trial_registry`。
**测试注入点**：`llm` 参数可注入 mock —— **这是本模块唯一可测的入口**。

---

## ingestion.append_pdfs

**唯一职责**：增量入库编排。

| 符号 | 签名 |
|---|---|
| `append_index` | `(files, *, reset: bool = False) -> None` |

**⚠️ 副作用**：写 ChromaDB。

---

## src.cli

**占位符，无可用接口。** 打印 `"placeholder. Add subcommands for ingest / query / evaluate."`。
⚠️ README 的 `poetry run oncoimmuno-rag` **无效**（`package-mode = false`，无 console script）。

---

## src.evaluation

**空壳包。** 仅 `__init__.py` 中的 `__all__ = []`，无任何代码。README 宣称的 "Evaluation harness" 不存在。

---

## 调用方清单（接口变更前必查）

| 被改接口 | 调用方 |
|---|---|
| `load_config` | `app.py`、`grade_evaluator`、`query_engine`、`vector_store`、`append_pdfs`、`pdf_parser` |
| `resolve_config_path` | 同上 5 个模块 |
| `QueryEngine.retrieve` | `app.py` |
| `select_top_k` / `rerank_hits` | `query_engine` |
| `grade_extraction` | `app.py`、`temp_indexer` |
| `trial_registry.lookup` | `query_engine`（懒加载）、`grade_evaluator` |
| `parse_pdfs` | `vector_store`、`append_pdfs` |
| `extract_table_summaries` | `pdf_parser` |
| `build_index` | `append_pdfs`、`vector_store.main` |
| `search_pubmed` / `fetch_details` | `app.py`（Tab 2） |
| `index_temp_document` | `app.py`（上传 PDF） |
| `AppConfig.*` | 全部业务模块 |