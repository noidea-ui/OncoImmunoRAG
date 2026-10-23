# docs/cross_platform.md — 跨平台与数据管理

> 核查日期：2026-10-23（Windows 实机）· 面向 WSL/Linux 迁移前的现状盘点。
> **本次不迁移运行环境**，只记录事实与处置建议。

---

## 1. 核查结论速览

| 检查项 | 结果 |
|---|---|
| 源码中的 Windows 专有 API | ✅ **0 处**（无 `winreg`/`ntpath`/`msvcrt`/`fcntl`/`os.startfile`） |
| 源码中的硬编码盘符路径 | ✅ **0 处** |
| 路径处理是否用 `pathlib` | ✅ 全部使用 `Path` |
| 是否依赖 CWD | ✅ 否，全部从 `__file__` 或 `config.yaml` 派生 |
| `.env` 是否自动加载 | ❌ 否，需手工 `set -a; . ./.env; set +a` |
| 行尾策略 | ✅ `.gitattributes` 强制 LF |
| **数据路径烘焙问题** | ❌ **向量库元数据中含他机路径**（ISSUE-022） |
| **脚本可移植性** | ⚠️ 8 个脚本缺依赖，本机无法运行（ISSUE-039） |

**整体判断：源码层面跨平台友好度良好；真正的阻塞点在数据层，不在代码层。**

---

## 2. 路径处理现状

### 统一惯例（5 处一致）

```python
resolve_config_path().parent / "data" / "raw"
```
出现于 `grade_evaluator.py:242`、`append_pdfs.py:90`、`pdf_parser.py:160`、`vector_store.py:42`、`query_engine.py:102`。

### 第 6 种写法（不一致）

```python
Path(__file__).resolve().parents[2] / "data" / "raw"   # table_extractor.py:317,321
```
绕过 `config_loader`，属遗留（ISSUE-044）。

### ⚠️ 一个隐含约束

`config_loader.py:35` 与 `table_extractor.py:317` 都用 `Path(__file__).resolve().parents[2]` 定位项目根 —— 即**假设源码树恰好在仓库根下两层**。

这在当前 `src/` 布局下成立，但意味着：**一旦把项目 install 成包（`package-mode` 改为 `true`），所有数据路径会全部指向 `site-packages`**，静默失败。当前 `package-mode = false`，安全。

### 跨平台结论

源码中无 `\\` 拼接、无盘符、无 CWD 依赖。**把仓库 clone 到 WSL 后，`config.yaml` 和 `data/` 结构相同即可运行** —— 只要数据齐全。

---

## 3. ⚠️ 数据层的跨平台阻塞点

### ISSUE-022 —— 向量库元数据烘焙了他机路径（最高优先级）

`data/vector_db` 的 406 个 chunk 的 `file_path` 字段为：

```
D:\OncoImmunoRAG\data\raw\Balar_2024_LancetOncol_KEYNOTE-057.pdf
```

而本项目实际位于 `D:\program\OncoImmunoRAG`。

- **本机就已失效**（该路径不存在）
- **WSL/Linux 下完全不可用**（`D:\` 语法无法解析）

**影响范围**：任何读取 chunk `file_path` 做溯源、读原文件、展示出处的功能。

**为什么现在没暴露**：现有 UI 用的是 `app.py` 里 `_PAPERS_RAW` 的相对路径拼接，不读 chunk 的 `file_path`。

**未来处置（需批准）**：入库时写相对路径，读取时按项目根拼接。**属于改入库逻辑，本阶段不修。**

### ISSUE-023 —— `data/vector_db_22` 不可读

24 MB，用当前 chromadb 0.6.3 打开抛 `KeyError: '_type'`（老版本构建）。**未删除。** 若要保留，需用匹配的老版 chromadb 导出后再迁移。

---

## 4. 依赖的跨平台差异

| 依赖 | Windows | Linux/WSL | 风险 |
|---|---|---|---|
| `torch` | 2.13.0 **+cpu** | 锁文件中 18 个 CUDA 包会**实际安装** | ⚠️ 安装体积数 GB；无 N 卡驱动时可能失败或回退 |
| `numpy` | 2.5.1（要求 Python ≥3.12） | 同 | ⚠️ **3.10/3.11 装不上**（ISSUE-002） |
| `matplotlib` / `plotly` / `seaborn` | **未安装** | 未声明 | ⚠️ 8 个绘图脚本无法运行（ISSUE-039） |
| `requests` | 2.34.2 | 同 | ✅ 已声明（ISSUE-035 已修） |
| `requests` 之外的传递依赖 | 已随锁文件固定 | 按 marker 正确跳过/安装 | ✅ |

### ⚠️ ISSUE-032 详述

`poetry.lock` 含 18 个 Linux CUDA 包。Windows 上按 marker 正确跳过，Linux 上会**真实安装**。

**含义**：WSL 侧的实际依赖图与 Windows **不完全相同**。若 WSL 无 NVIDIA 环境：
- `torch` 可能装不上 CUDA 轮子而回退 CPU 轮子
- 或因 `nvidia-*` 包缺失导致解析失败

**建议**：迁移前先在 WSL 跑 `nvidia-smi` 确认驱动状态，再决定是否装 CUDA 版 torch。

---

## 5. 环境变量

| 变量 | 优先级 | 陷阱 |
|---|---|---|
| `OPENAI_API_KEY` | 环境变量 **>** `config.yaml` | 缺失 → `get_query_engine()` 静默返回 `None` |
| `LLM_BASE_URL` | **`config.yaml` >** 环境变量 | ⚠️ 在 `.env` 里改它**无效**（ISSUE-043） |
| `LLM_MODEL` | **`config.yaml` >** 环境变量 | 同上 |
| `HF_HOME` | 未使用 | ⚠️ 模型默认下到 `~/.cache/huggingface`，WSL 与 Windows **不共享** |

⚠️ **代码不加载 `.env`**（无 `python-dotenv`）。每次启动必须手工导出。

---

## 6. ⚠️ ISSUE-054 —— 脚本提示用户把凭据写进被跟踪的文件

**新发现，2026-10-23。** 6 个 benchmark 脚本在 Key 缺失时提示：

```
scripts/run_batch_3.py:454        "OPENAI_API_KEY is empty — set it in config.yaml ..."
scripts/run_rerank_ab_holdout.py:433  "OPENAI_API_KEY 未配置 — 请在 config.yaml / env 中设置"
```

而 `config.yaml` **已被 Git 跟踪**，且其第 3–6 行明确写着：

```
# openai_api_key: DO NOT put a credential in this file — it is versioned.
```

**影响**：新成员照着脚本提示操作，会把密钥写进被跟踪的文件 → **`git commit -a` 即泄漏**。脚本提示与配置文件的自我警告**直接矛盾**。

**当前状态**：记录，**未修**（改提示文案需逐个改 6 个脚本文件，且属于业务代码）。

**附带发现**：其中 3 个脚本有 `--dry-run` 逃生阀（`run_full_benchmark`、`run_holdout_grading`、`run_verification_grading`），但 `run_batch_3`、`run_batch_4`、`run_rerank_ab_holdout` **没有** —— 无 Key 时只能中止或改配置文件。

---

## 7. 数据分类与 Git 处置

### ✅ 应进入 Git

| 内容 | 位置 | 现状 |
|---|---|---|
| 源码 | `src/`, `app.py`, `scripts/` | ✅ 已跟踪 |
| 配置 | `config.yaml`, `pyproject.toml`, `poetry.lock` | ✅ 已跟踪 |
| 测试 | `tests/test_*.py` | ✅ 已跟踪 |
| 文档 | `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `docs/`, `CURRENT_ISSUES.md` | ⚠️ **本轮新增，尚未 add** |
| CI 配置 | `.github/` | ⚠️ **本轮新增，尚未 add** |
| 占位文件 | `data/raw/.gitkeep` | ✅ 已跟踪 |

### ✅ 可进入 Git 的小型非敏感样本

| 内容 | 说明 |
|---|---|
| **合成测试文本** | `tests/` 中的 `PROSE`、`TABLE_BLOCK` 等内联样本 —— 已在用，**不需要真实论文** |
| 评测结果 JSON | `tests/results/`、`data/benchmark_results/` 等，**文献级抽取结果，不含患者数据**。⚠️ 但含**已知错误**（ISSUE-014、ISSUE-013） |

⚠️ 评测 JSON 目前**有意不忽略**（`.gitignore` 第 35–41 行注明是刻意选择）。上传会把这些结果固化，建议等 ISSUE-013/014 修复后重跑基线。

### ❌ 不应进入 Git

| 内容 | 数量/体积 | ignore 规则 | 状态 |
|---|---|---|---|
| 论文 PDF | 22 个 / 约 41 MB | `data/raw/*`、`verification/*.pdf`、`holdout/*.pdf` | ✅ 已忽略 |
| ChromaDB | `vector_db` 14 MB | `data/vector_db/*` | ✅ 已忽略 |
| ChromaDB（旧） | `vector_db_22` 24 MB | `data/vector_db_22/*` | ✅ 已忽略 |
| 嵌入模型 | 1.3 GB（不入库） | 由 `HF_HOME` 管理 | ✅ 不在仓库 |
| `.venv/` | 1.3 GB | `.venv/` | ✅ 已忽略 |
| 陈旧源码副本 | `src_00/` | `src_00/`（本轮新增） | ✅ 已忽略 |
| 密钥 | 1 个 | `.env` | ✅ 已忽略 |
| 缓存 | — | `__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`.coverage` | ✅ 已忽略 |

### ⚠️ 有意未忽略的产物（约 4.0 MB / 70 文件）

`result/` 下 7 张 PNG/JPG + 2 个 HTML + 1 个 SVG、3 个 `.xlsx`、多个 benchmark JSON。

`.gitignore` 注释说明这是**刻意的**（等流水线修好后重跑）。**非疏忽，非泄漏**，仅占体积。是否忽略属决策项。

### 🔴 绝不能提交

- `.env` 与任何 API Key
- `config.yaml` 中的凭据（该文件已跟踪）
- 真实患者数据 —— ✅ **核查确认：项目中不存在**。评测 JSON 均为文献级抽取结果（试验名、查询词、文件名、chunk_id、分数、GRADE 评级）
- 22 个论文 PDF（未来由专用服务器提供）

---

## 8. 测试的跨平台依赖

| 测试 | 是否依赖本机路径/数据 | CI 可运行 |
|---|---|---|
| 108 个现有用例 | ❌ **全部不依赖** | ✅ |
| 依赖的真实数据 | 无（合成文本 + mock） | ✅ |
| 真实 PDF 集成测试 | ❌ **尚未建立** | ❌ |

✅ 当前测试套件**完全可移植**：`test_ingestion_pipeline.py` 用内联合成文档而非真实 PDF，这是刻意的设计（见该文件 docstring）。

---

## 9. 迁移到 WSL 的检查清单

- [ ] 确认 Python **3.12**（3.10/3.11 装不上锁文件）
- [ ] `nvidia-smi` 确认 GPU 状态，再决定 torch CUDA 版
- [ ] `poetry install` 后跑 `pytest tests/ -o addopts="" -q`，期望 108 passed + 5 xfailed
- [ ] 手工获取 22 个 PDF（**不入库**，未来由服务器提供）
- [ ] 若使用现有 `data/vector_db`，接受 ISSUE-022 的路径失效；或重建索引
- [ ] 设置 `HF_HOME` 指向与 Windows 共享的目录，避免重下 1.3 GB
- [ ] `set -a; . ./.env; set +a` 导出凭据（代码不自动加载）
- [ ] 提交 `pyproject.toml` + `poetry.lock` 的依赖修复，否则拉到的是坏环境