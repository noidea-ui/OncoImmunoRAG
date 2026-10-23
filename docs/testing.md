# docs/testing.md — 测试体系

> 当前基线：**108 passed + 5 xfailed**（2026-10-23）。
> 全部用例**不需要** API Key、网络、GPU、ChromaDB、嵌入模型、PDF 文件。
> **原有 15 个用例行为未变。**

---

## 1. 唯一基线命令

```powershell
.venv\Scripts\python.exe -m pytest tests/ -o addopts="" -p no:cacheprovider -q
```

Linux/WSL：

```bash
.venv/bin/python -m pytest tests/ -o addopts="" -p no:cacheprovider -q
```

### 为什么必须加 `-o addopts=""`

`pyproject.toml` 中：

```toml
addopts = "-q --cov=oncoimmuno_rag --cov-report=term-missing"
```

**`oncoimmuno_rag` 这个包不存在**（实际包名为 `ingestion`/`retrieval`/`grading`/`utils`，且 `package-mode = false`）。后果：

- pytest 产生 **3 条 warning**，覆盖率报告为空
- **覆盖率门禁静默失效** —— 看起来跑了覆盖率，实际什么都没测

⚠️ **红线：不要为了让默认命令好看来修改这个配置**，那属于改变现有工程状态。绕开它，直到负责人批准修 `pyproject.toml`。

### 为什么加 `-p no:cacheprovider`

避免在无写权限环境或 CI 中因 `.pytest_cache` 写入失败而中断。

---

## 2. 当前测试清单

### 原有基线（勿破坏）

| 文件 | 用例数 | 覆盖对象 |
|---|---|---|
| `tests/test_table_extractor.py` | 14 | `ingestion.table_extractor` |
| `tests/test_parser.py` | 1 | `ingestion.pdf_parser` |

### 2026-10-23 新增

| 文件 | 用例数 | 覆盖对象 | 外部依赖 | marker |
|---|---|---|---|---|
| `tests/test_selector.py` | 18 | `retrieval.selector`（原 0 覆盖） | 无 | unit / contract |
| `tests/test_grade_evaluator_contract.py` | 18 | `grading.grade_evaluator`（原 0 覆盖，1587 行） | **mock LLM** | unit / contract |
| `tests/test_pubmed_client_contract.py` | 13 | `retrieval.pubmed_client`（原 0 覆盖） | **mock requests** | unit / contract |
| `tests/test_query_engine_contract.py` | 10 | `retrieval.query_engine`（原 0 覆盖） | 无（不加载模型） | unit / contract |
| `tests/test_config_loader.py` | 21 | `utils.config_loader` + `utils.trial_registry`（原 0 覆盖） | 无 | unit / contract |
| `tests/test_ingestion_pipeline.py` | 12 | `pdf_parser → table_extractor` 流水线 | 无（合成文档） | integration |
| `tests/test_config_smoke.py` | 1 | 配置读取（原 0 用例） | 无 | unit |
| `tests/test_new_deps_smoke.py` | 5 | 依赖声明核对（原 0 用例） | 部分 requires_model | unit |
| **合计** | **108 passed + 5 xfailed** | | | |

✅ **全部不需要** API Key、网络、GPU、ChromaDB、嵌入模型、真实 PDF。

### 非测试文件（勿当作测试）

| 文件 | 实际是什么 |
|---|---|
| `tests/build_review_table.py` | 脚本，不会被 pytest 收集 |
| `tests/eval_batch_grade.py` | 脚本，**会调用真实 LLM**，不会被 pytest 收集 |

---

## 3. 测试分类约定

新增测试必须用 marker 标注，便于分层执行：

| Marker | 含义 | 运行条件 |
|---|---|---|
| `@pytest.mark.unit` | 纯逻辑，无 I/O | 无条件 |
| `@pytest.mark.contract` | 接口契约（类型/字段/异常） | 无条件 |
| `@pytest.mark.integration` | 端到端流程 | 需要数据文件 |
| `@pytest.mark.requires_api` | 需要 LLM 凭据 | `OPENAI_API_KEY` |
| `@pytest.mark.requires_model` | 需要嵌入模型 | 1.3 GB |
| `@pytest.mark.requires_network` | 需要外网 | NCBI / HF / LLM |

### 分层运行命令

```bash
# 快速单元测试（CI 默认跑这个，秒级）
pytest -m unit -o addopts=""

# 契约测试
pytest -m contract -o addopts=""

# 完整（含集成）
pytest -o addopts="" -m "not requires_api and not requires_model and not requires_network"

# 需要外部服务的（手动执行）
pytest -o addopts="" -m requires_api
```

⚠️ **尚未在 `pyproject.toml` 中注册这些 marker。** 使用前需先在 `[tool.pytest.ini_options].markers` 中声明，否则 pytest 产生 warning。

---

## 4. 已知难以测试的模块

| 模块 | 行数 | 测试 | 障碍 | 可测入口 |
|---|---|---|---|---|
| `grading/grade_evaluator.py` | 1587 | **0** | 直接调 LLM | ✅ `grade_extraction(..., llm=<mock>)` —— **唯一入口** |
| `retrieval/query_engine.py` | 587 | **0** | 构造即加载 1.3 GB 模型 + 连 Chroma | 需 mock `from_config` |
| `retrieval/selector.py` | 502 | **0** | 无外部依赖 | ✅ **纯函数，最容易补测** |
| `retrieval/temp_indexer.py` | 279 | **0** | 写临时 collection | 可 mock |
| `retrieval/pubmed_client.py` | 299 | **0** | 真实网络请求 | mock `requests` |
| `utils/trial_registry.py` | 207 | **0** | 纯字典 | ✅ 纯逻辑，最容易补测 |

**两个结构性障碍**

1. **全仓 0 处使用 `logging`，全部 `print()`（47 处）。** 测试无法用 `caplog` 捕获输出，也无法断言诊断信息。
2. **全仓 0 处设置随机种子。** 任何涉及采样的结果不可复现。

### `selector` 的全局状态陷阱

`rerank_hits` 受模块级开关 `get/set_rerank_enabled_default` 控制。测试中修改会**污染同进程内的其他用例**：

```python
def test_rerank_enabled():
    original = selector.get_rerank_enabled_default()
    try:
        selector.set_rerank_enabled_default(True)
        ...
    finally:
        selector.set_rerank_enabled_default(original)   # 必须恢复
```

---

## 5. 尚未建立的测试（后续任务）

| 类别 | 状态 | 备注 |
|---|---|---|
| 单元测试（`selector`） | ✅ 已建立 | 18 用例 |
| 单元测试（`trial_registry`） | ✅ 已建立 | 11 用例 |
| 单元测试（`config_loader`） | ✅ 已建立 | 21 用例 |
| 单元测试（`temp_indexer`） | ❌ 未建立 | 可 mock collection |
| **契约测试** | ✅ 已建立（6 个模块） | 跨成员/跨模型集成的关键 |
| 集成测试（解析→切块→表格） | ✅ 已建立 | 12 用例，合成文档 |
| 集成测试（**真实 PDF** → 入库） | ❌ 未建立 | 需 PDF，数据未来由服务器提供 |
| 检索**质量**测试（命中率等） | ❌ 未建立 | 需向量库 + 模型，且依赖已知缺陷 |
| CI | ❌ 未建立 | 无 `.github/`，见 `CONTRIBUTING.md` |

### 5 个 xfail —— 已知缺陷的可执行记录

| 用例 | 对应缺陷 |
|---|---|
| `test_query_engine_contract.py::test_query_method_exists` | ISSUE-008 |
| `test_config_loader.py::test_load_config_raise_if_missing_false_returns_default` | ISSUE-015 |
| `test_ingestion_pipeline.py::test_most_chunks_are_classified_as_a_real_section` | ISSUE-019 |
| `test_selector.py::test_public_symbols_are_exported_in_dunder_all` | ISSUE-049 |
| `test_pubmed_client_contract.py::test_hit_year_is_a_string` | ISSUE-052 |

⚠️ **修复对应缺陷时**，xfail 会转为 XPASS（strict 模式下 pytest 会**失败**并提示你去掉 xfail 标记）。这是刻意的：防止缺陷被静默「修好」却无人更新测试。

---

## 6. 回归红线

**每次改动后必须复测，且原有 15 个用例的行为不得变化。**

### 环境基线（2026-10-23 实测）

| 检查项 | 期望值 |
|---|---|
| `pytest tests/ -o addopts="" -q` | **108 passed, 5 xfailed** |
| 其中原有基线用例 | **15 passed，行为无变化** |
| `Streamlit` HTTP | **200** |
| `AppTest` 执行 `app.py` | **0 exception** |
| `QueryEngine.from_config` | 约 **58.8s**（首次，含模型下载） |
| `retrieve()` | 约 **0.53s** / **5 hits** |
| `grade_extraction()` | 约 **15.2s** / **13 字段** |
| UI 端到端点击 | 约 **10–15s**，0 异常 |
| `ruff check tests/`（新增文件） | **All checks passed** |
| `ruff check .`（全仓） | 248 errors（**存量，不要求下降**） |

### ⚠️ 契约测试**不能**锁死有缺陷的行为

以下行为**已被确认为缺陷**，测试**不得**把它们固化为"正确"：

| 行为 | Issue |
|---|---|
| 量化阈值双轨（LLM 7.0 → Python 2.0 覆盖） | ISSUE-013 |
| `effect_size` 无校验，别篇 HR 混入 | ISSUE-014 |
| `per_file_cap` 被绕过，单篇 100% | ISSUE-020 |
| `_normalize_risk` 把「未报告」塌缩为「中」 | ISSUE-015 |
| 章节分类 75.8% 落到 `unknown` | ISSUE-019 |

**契约测试只断言接口形状**（字段名、类型、异常类型、流转完整性），**不断言具体评分数值**。

⚠️ 发现这些缺陷被修改时，**记录到 `CURRENT_ISSUES.md` 并停下**，等负责人决定是修缺陷还是保留现状。

---

## 7. 测试样本原则

| 规则 | 说明 |
|---|---|
| 优先**合成的小文本** | 不依赖真实论文 |
| 不依赖大量真实 PDF | 集成测试用最小样本 |
| 不依赖外部 API | 全部 mock |
| 不依赖 GPU | CI 无 GPU |
| 真实数据仅用于**手工验证** | 不进单元测试 |

理由：CI 需可稳定复现，且 Lancet Oncol / NEJM 论文的再分发授权未确认（见 ISSUE-026）。

---

## 8. 已知失效的门禁

| 项 | 状态 |
|---|---|
| `--cov=oncoimmuno_rag` | ❌ 包不存在，静默失效 |
| `test_config_smoke.py` | ✅ 已修为真实用例（ISSUE-050） |
| `test_new_deps_smoke.py` | ✅ 已修为真实用例 + 消除收集期污染（ISSUE-050/051） |
| GitHub Actions | ❌ 不存在 |

## 9. 已注册的 marker

已在 `pyproject.toml` 的 `[tool.pytest.ini_options].markers` 中声明：

```
unit, contract, integration,
requires_api, requires_model, requires_network, requires_gpu
```

⚠️ `requires_api` / `requires_network` 目前**没有任何用例使用** —— 等需要接真实外部服务时再补，不要预先写空标记。