# AGENTS.md — OncoImmunoRAG 代理工作规则

> **适用范围**：任何 AI 编程代理（Claude Code、Cursor、Copilot、GPT 等）在本仓库工作前**必须先完整阅读本文件**。
> 本文件是硬约束，不是建议。与通用最佳实践冲突时，**以本文件为准**。

---

## 0. 最高原则（违反即回退）

1. **不改变现有可运行行为。** 当前基线：Streamlit 页面 200 + AppTest 0 异常、`retrieve()` 0.53s、`grade_extraction()` 15.2s、`pytest` 15 passed。任何变更后必须复测这四项。
2. **不改业务逻辑、不做重构、不换技术栈。** 本仓库当前阶段只做工程化，不做功能开发。
3. **发现缺陷先记录，不修复。** 一律写入 `CURRENT_ISSUES.md`，等待项目负责人批准。
4. **不擅自增删依赖。** 需要新依赖 → 先在 `CURRENT_ISSUES.md` 记录并等批准。
5. **不改动他人负责的模块。** 每个任务以 `docs/TASK_TEMPLATE.md` 中「允许修改的文件」为唯一授权范围。

---

## 1. 项目背景

OncoImmunoRAG 是**肿瘤免疫治疗文献的证据分级系统**。读取临床试验 PDF，向量化存入 ChromaDB，检索相关证据片段，调用 LLM 产出结构化 GRADE 评级。

**代码实况 ≠ 产品设想**（务必先读 `CURRENT_ISSUES.md`）：
- 现有能力 = **单篇文献的自动评分卡**。
- **RoB 2 工具不存在**（全仓 0 命中）。
- **结局层 GRADE 不存在**。
- **跨研究综合引擎不可用**（`QueryEngine.query` 不存在，见 ISSUE-008）。
- **Web 端图表能力为 0**。
- `design_gold`（界面上的「金标准设计」）**仅用于显示，从不参与校验**。

不要因为文档提到某能力就假定代码已实现。

---

## 2. 技术栈与启动

| 项 | 值 |
|---|---|
| 语言 | Python **3.12**（`>=3.10` 声明无效，`numpy 2.5.1` 实际要求 `>=3.12`） |
| 依赖管理 | Poetry 2.5.1（`pyproject.toml` + `poetry.lock`） |
| 前端 | Streamlit 1.66（单文件 `app.py`） |
| 向量库 | ChromaDB 0.6.3（`data/vector_db`） |
| 嵌入模型 | `BAAI/bge-large-en-v1.5`（约 1.3 GB，首次运行需联网下载） |
| LLM | OpenAI 兼容端点（`config.yaml` 的 `llm_base_url`，当前为 MiniMax） |
| 推理 | torch 2.13.0 **CPU-only** |

### 启动命令（Windows，当前可用）

```powershell
cd D:\program\OncoImmunoRAG
set -a; . .\.env; set +a
.venv\Scripts\python.exe -m streamlit run app.py
```

访问 **http://localhost:8501**。

⚠️ `poetry run oncoimmuno-rag` **是错的**（无 console script）。`src/cli.py` 是占位符，无可用子命令。

### 环境变量

| 变量 | 优先级 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | **环境变量 > config.yaml** | 缺失则 `get_query_engine()` 返回 `None`，检索与分级全部不可用 |
| `LLM_BASE_URL` | **config.yaml > 环境变量** | 改环境变量**无效** |
| `LLM_MODEL` | **config.yaml > 环境变量** | 同上 |

**凭据纪律**：
- `config.yaml` **已被 Git 跟踪**，绝不可写入密钥。
- 密钥只经环境变量传入；本地可放 `.env`（已 gitignore）。
- **任何情况下不得回显、复制、提交、记录密钥值。**
- 代码不自动加载 `.env`，必须 `set -a; . ./.env; set +a`。

---

## 3. 模块职责

```
app.py                        UI + 业务编排（848 行，耦合度最高）
src/utils/config_loader.py    配置读取 + 数据目录定位（被 5 模块依赖的事实内核）
src/utils/prompts.py          提示词片段（纯字符串）
src/utils/trial_registry.py   手工维护的研究设计覆盖表（参与评分）
src/ingestion/pdf_parser.py   PDF 加载 + 切块
src/ingestion/table_extractor.py  表格抽取
src/ingestion/vector_store.py ChromaDB 写入
src/ingestion/append_pdfs.py 增量入库
src/retrieval/query_engine.py 检索编排（MMR over-fetch → canonical 补召回 → rerank → select_top_k）
src/retrieval/selector.py     rerank 与 top_k 选取
src/retrieval/temp_indexer.py 临时 collection（会话级上传 PDF）
src/retrieval/pubmed_client.py NCBI E-utilities 客户端
src/grading/grade_evaluator.py GRADE 评分（1587 行，零测试覆盖）
src/evaluation/               ⚠️ 空壳包，无任何代码
```

接口契约见 `docs/interfaces.md`，架构图见 `ARCHITECTURE.md`。

---

## 4. 代码风格

- Ruff（`line-length = 100`，`select = ["E","F","I","W","B","UP"]`），配置在 `pyproject.toml`。
- **新增代码必须通过 `ruff check`；存量代码的 248 个存量错误不要求修复**，也不要跑 `ruff --fix` 或 `ruff format` 做全仓格式化。
- 类型标注覆盖率已达 **96%**，新代码必须补全。详细规则见 `docs/coding_standards.md`。

---

## 5. 测试命令

```bash
# 唯一基线命令（必须用 -o addopts="" 绕开失效的覆盖率配置）
.venv/Scripts/python.exe -m pytest tests/ -o addopts="" -p no:cacheprovider -q
```

当前基线：**15 passed**（14 个 `test_table_extractor` + 1 个 `test_parser`）。

⚠️ `pyproject.toml` 中 `--cov=oncoimmuno_rag` **指向不存在的包**，覆盖率门禁静默失效。

⚠️ `tests/` 下的 `build_review_table.py` 与 `eval_batch_grade.py` **不是测试文件**，是脚本，勿当作测试。

---

## 6. 禁止事项

- ❌ 禁止把密钥、凭据、`.env`、真实患者数据提交到仓库。
- ❌ 禁止提交 PDF（已 gitignore）。论文未来由专用服务器提供。
- ❌ 禁止删除/重建 `data/vector_db`（重建需要 1.3 GB 模型 + 全量重算）。
- ❌ 禁止修复 `CURRENT_ISSUES.md` 中已记录的缺陷（除非负责人明确指派该条）。
- ❌ 禁止修改他人任务范围内的文件。
- ❌ 禁止为「代码更优雅」改动可正常运行的实现。
- ❌ 禁止删除 `D:/Python/poetry.lock.before-depfix.bak`（依赖修复的唯一回滚副本）。
- ❌ 禁止 `git add .` / `git add -A` / `git reset` / `git clean` / `git push`（除非明确授权）。
- ❌ 禁止触碰 `src_00/`（陈旧副本，已 gitignore）。

---

## 7. Git 协作流程

```
main → feature 分支 → 本地测试 → PR → 自动测试 → 人工审查 → 合并
```

- 禁止在 `main` 上直接开发。
- 一个 PR 只解决一个明确问题。
- 涉及接口的 PR **必须**同步更新 `docs/interfaces.md` 并补测试。
- 发现无关缺陷 → 记入 `CURRENT_ISSUES.md`，**不顺手修**。

---

## 8. 完成任务前的检查清单

- [ ] 只改了任务授权范围内的文件
- [ ] `pytest tests/ -o addopts="" -q` 通过，且**原有 15 个用例无一变化**
- [ ] `ruff check <改动的文件>` 无新增错误
- [ ] 页面仍能启动（HTTP 200 + AppTest 0 异常）
- [ ] `retrieve()` 与 `grade_extraction()` 仍能返回结果
- [ ] 新增依赖已获批准且写入 `CURRENT_ISSUES.md`
- [ ] 接口变更已同步 `docs/interfaces.md`
- [ ] 未提交任何密钥、PDF、数据库、模型权重
- [ ] 新发现的缺陷已记入 `CURRENT_ISSUES.md`