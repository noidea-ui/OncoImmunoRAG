# Current Project Issues

> 记录日期：2026-10-23（Windows 环境实机核查）
> 原则：**本阶段只记录，不修复。**
> 证据强度标注：**[已验证]** = 本次实际运行/读取确认；**[静态]** = 阅读代码推断，未运行；**[历史]** = 上一轮已实测复现，见 `HANDOFF_TO_WSL.md` §4。
> 环境：Windows 11 / Python 3.12.15 / `.venv` / CPU-only torch 2.13.0。

---

## Environment

### ISSUE-001
**问题：** 项目入口 `app.py` 直接依赖 `streamlit`，但原 `pyproject.toml` 未声明该依赖。

**影响：** 全新环境执行 `poetry install` 后启动即 `ModuleNotFoundError: No module named 'streamlit'`。

**当前状态：** **已修复但未提交。** `pyproject.toml` / `poetry.lock` 已补 `streamlit>=1.60.0,<2.0.0`（实际安装 1.66.0）与 `llama-index-vector-stores-chroma>=0.6.0`，当前工作区可运行。**这两个文件的修改仍在未暂存区，尚未 commit —— WSL 侧 `git pull` 前必须先提交，否则拉不到。**

**建议未来处理：** 在 WSL 迁移前先 commit `pyproject.toml` + `poetry.lock`。

---

### ISSUE-002
**问题：** `pyproject.toml` 声明 `requires-python = ">=3.10,<3.15"`，但锁定的 `numpy 2.5.1` 要求 `>=3.12`。3.10 / 3.11 实际装不上。

**影响：** 新机器若按 README 选 Python 3.10 会解析失败，且错误信息不指向真实原因。

**当前状态：** 不影响当前 Windows 运行（本机 3.12.15）。

**建议未来处理：** WSL 侧固定 Python 3.12，并在 README 写明。

---

### ISSUE-003
**问题：** README 的启动指令 `poetry run oncoimmuno-rag` **无效**。

**影响：** 照 README 操作的任何人都会失败。`pyproject.toml` 设了 `package-mode = false`，且**没有 `[project.scripts]` / `[tool.poetry.scripts]`**，该命令不存在。

**当前状态：** 不影响 Streamlit 启动（正确入口是 `python -m streamlit run app.py`）。

**建议未来处理：** 修正 README 启动章节。

---

### ISSUE-004
**问题：** `src/cli.py` 是占位符，只打印 `"placeholder. Add subcommands for ingest / query / evaluate."`，无任何子命令。

**影响：** 项目没有可用的命令行入口，所有能力只能经由 Streamlit 使用。

**当前状态：** 不影响运行。

**建议未来处理：** 记录为已知功能缺口。

---

### ISSUE-005
**问题：** `config.yaml` 无 `openai_api_key`，且必须由环境变量提供 `OPENAI_API_KEY`。加载配置即产生 `UserWarning`。

**影响：** **这是最硬的运行门控** —— `app.py:117` 的 `get_query_engine()` 在 `if not cfg.effective_openai_api_key(): return None` 处直接返回 `None`，**没有 API Key 就不会构造 `QueryEngine`，向量化/检索/问答整条主链路都不会被初始化**。

**当前状态：** ✅ [已验证-已解除] 用户授权后创建了 `.env`（已被 `.gitignore` 第 49 行忽略，`git check-ignore` 确认）。设置 `OPENAI_API_KEY` 后 `load_config().effective_openai_api_key()` 返回有效值，`base_url` 正确从 `config.yaml` 取到 MiniMax 端点。**Key 现在以明文存在于磁盘 `D:\program\OncoImmunoRAG\.env`（已忽略，不上传）。**

⚠️ **待用户处理：验证完成后务必在 MiniMax 控制台轮换该 Key，并删除本地 `.env`。我不回显也不代为轮换。**

**建议未来处理：** README 补充「需要 `OPENAI_API_KEY`」前置说明。

---

### ISSUE-006
**问题：** 嵌入模型 `BAAI/bge-large-en-v1.5`（约 1.3 GB）需联网下载，本机原本无 `~/.cache/huggingface` 缓存。

**影响：** 首次执行检索必须联网；离线或网络受限时必然失败。

**当前状态：** ✅ [已验证-已下载] 用户授权后已下载完成。`QueryEngine.from_config()` 冷启动耗时 **58.8s**（含加载权重），之后走本地缓存。控制台提示 `unauthenticated requests to the HF Hub`（未设 `HF_TOKEN`，只影响下载限速，不影响功能）。

**建议未来处理：** WSL 迁移时用 `HF_HOME` 指向与 Windows 共享的目录，可避免重复下载 1.3 GB。

---

### ISSUE-007
**问题：** 无 `.env` 文件，且项目**未安装 `python-dotenv`、代码也不自动加载 `.env`**。`.env.example` 已诚实注明这一点。

**影响：** 每次启动都必须手工 `set -a; . ./.env; set +a`，无文档化的一键方式。

**当前状态：** 不影响当前运行。

**建议未来处理：** 启动脚本里显式导出，或引入 dotenv（**不建议现在改**）。

---

## Backend

### ISSUE-008
**问题：** `QueryEngine` **没有 `query()` 方法** —— 该方法误缩进在 `class _NodeWithScore` 内部（`src/retrieval/query_engine.py:435,450`）。

**影响：** [本轮已实测] 在真实环境实例化 `QueryEngine.from_config()` 后枚举公开方法，结果为：
```
['from_config', 'get_rerank_enabled', 'retrieve', 'set_rerank_enabled']
hasattr(eng, "query") == False
```
`app.py:829`、模块级 `query()`(:525)、CLI `main()`(:574) 三处调用**全部不可达** → 跨研究综合功能整体不可用。`src_00/` 有同样缺陷。

**当前状态：** 不影响页面启动与检索/分级；**跨研究综合确定不可用**。**本阶段不修。**

---

### ISSUE-008b
**问题（实测新增）：** `QueryEngine.retrieve()` 返回的是 **llama-index 原生 `NodeWithScore`**，而误缩进的 `_NodeWithScore` 是自定义类 —— 两者同名不同物，极易误判与误修。

**影响：** [已验证] `type(hits[0]) == llama_index.core.schema.NodeWithScore`，其元数据在 `.metadata` 属性上，**不提供 `.get()` 字典访问**。任何按 `hit.get("file_name")` 写的下游代码都会失败（本轮调试脚本首次尝试即踩中）。

**当前状态：** 记录。修复 ISSUE-008 时必须先确认要提升的是哪个类的方法。

---

### ISSUE-009
**问题：** 跨文献守卫**逻辑上不可执行**：提示词要求模型「按文件名归类」，但喂给模型的 `_format_chunks` 输出只有 `[片段 N]`，**不含文件名**。

**影响：** [历史] 跨文献识别实际退化为单篇处理。

**当前状态：** 不影响启动。

---

### ISSUE-010
**问题：** `grade_evaluator.py`（1587 行）与 `query_engine.py`（587 行）**测试覆盖率为 0**。

**影响：** 任何评分/检索改动都没有回归网。

**当前状态：** [已验证] `pytest tests/ -o addopts="" -q` → **15 passed**（14 个 `test_table_extractor` + 1 个 `test_parser`），全部不依赖凭据/网络/模型。这是唯一的测试基线。

**建议未来处理：** 改评分逻辑前必须先补测试。

---

### ISSUE-011
**问题：** `pyproject.toml` 的 `addopts = "--cov=oncoimmuno_rag"`，但**不存在名为 `oncoimmuno_rag` 的包**（包名为 `ingestion`/`retrieval`/…，且 `package-mode = false`）。

**影响：** 默认 `pytest` 覆盖率门禁静默失效（实测 3 条 warning、报告为空）。

**当前状态：** 不影响功能。绕行命令：`.venv/Scripts/python.exe -m pytest tests/ -o addopts="" -q`

**建议未来处理：** 修正 `--cov` 目标。

---

### ISSUE-012
**问题：** [静态] `retrieval/temp_indexer.py` 在 `app.py:60` 被导入，但**不在之前的 import 检查清单中**，也未列入 `pyproject.toml` 的包说明中。

**影响：** 若该模块存在额外依赖，上传 WSL 后可能在 import 期失败。

**当前状态：** [已验证] `app.py` 实际运行时 import 成功。

---

### ISSUE-013
**问题：** 量化 GRADE 阈值**双轨**：提示词用 8.5 / 5.0 / 2.6 / 0，Python 侧用 8.5 / **6.0** / **3.5** / 0。

**影响：** ⚠️ **[本轮已实测复现]** 真实调用 KEYNOTE-057 摘要，分级器在 `grade_reasoning` 末尾输出：
```
[综合评分重算] LLM=7.0 / Python=2.0 (差异=5.00); 以 Python 覆盖。
breakdown: 起始=4.0(single...)
```
即 **LLM 给出 7.0 分（按提示词阈值应属高档），Python 重算得 2.0 分并直接覆盖，最终 `overall_grade` 判为「极低」**。分差达 5.00 分，足以让结论从「高」翻转到「极低」。

**当前状态：** 不影响启动，**严重影响评分可信度**。这是当前所有已落盘评测结果（`tests/results/`、`data/benchmark_results/`）的共同系统性偏差来源。

**建议未来处理：** 属于需要产品决策的问题（以哪套阈值为准），本阶段不修。

---

### ISSUE-014
**问题：** [静态] `effect_size` 在 Python 侧**零校验**，别篇研究（未检索到的文献）的 HR 直接参与当前文献评分。

**影响：** [历史] 已落盘证据 `data/benchmark_results/McGregor-DAD.json:46,47` —— N=886、HR=0.47 实为 EV-302 的数值。

**当前状态：** 不影响启动，**影响结果正确性**。基线数据本身已被污染。

---

### ISSUE-015
**问题：** [静态] `_normalize_risk` 把 `未提及 / 未知 / 未报告 / 不适用 / n/a / ''` **全部塌缩为「中」**。

**影响：** [历史] 「不可评估」被伪装成「确定判断」呈现给临床读者。

**当前状态：** 不影响启动。

---

### ISSUE-016
**问题：** `ruff check .` → **248 errors**（123 可自动修复），`ruff format --check` 有 47 个文件需重排。多数为风格项。

**影响：** 不影响运行。

**当前状态：** 本阶段不处理。

---

### ISSUE-017
**问题：** 控制台编码：Windows 默认 GBK 下，任何打印 emoji 的代码路径会抛 `UnicodeEncodeError`。

**影响：** [已验证] 复现于调试脚本；`app.py` 自身在 Streamlit（UTF-8 网页环境）中不受影响。

**当前状态：** 不影响 Web 界面。WSL/Linux 下不复现。

---

### ISSUE-021b
**问题（本轮实测新增）：** `QueryEngine.retrieve()` 的 **`top_k` 不是参数**，而是构造时固定为 `self._top_k`（默认 **5**）。函数签名只有 `retrieve(question, *, canonical_file=None)`，传 `top_k=` 会抛 `TypeError`。

**影响：** [已验证] 实测 `select_top_k(hits, 12, ...)` 对 retrieve 返回的 5 条结果只能选出 5 条 —— **调用方无法通过该接口扩大候选池**。同时这解释了 ISSUE-020：`top_k` 恒为 5 时单篇占比 100% 是必然结果，而非偶发。

**当前状态：** 不影响启动。记录待后续设计决策。

---

### ISSUE-021c
**问题（本轮实测新增）：** `GRADEResult` 字段类型不一致 —— `risk_of_bias` 实际返回**裸字符串** `"高"`，而非结构化对象。

**影响：** [已验证] `json.loads(r.model_dump_json())["risk_of_bias"]` 得到的是 `"高"`。任何按字段访问子键（如 `risk_of_bias["随机化"]`）的下游代码或图表脚本都会崩。`scripts/` 下 7 个绘图脚本均消费这些 JSON，值得逐个核查。

**当前状态：** 不影响启动，影响下游脚本。

---

## Frontend

### ISSUE-018
**问题：** 界面多处**硬编码宣称「22 篇」**（`app.py:363` 下拉框 help、`app.py:428` 标签、`app.py:450` 标题「库内文献分级（22 篇 PDF 池）」），但**向量库实际只索引了 12 篇 / 406 个 chunk**。

**影响：** [已验证] 用户看到「22 篇池」但 `data/vector_db` 只有 12 个 distinct `file_name`。文案与数据不符，且这 22 篇中有 10 篇在 `verification/` 与 `holdout/` 目录、**从未进入向量库**。

**当前状态：** 不影响启动，**影响对系统能力的判断**。

**建议未来处理：** 文案改为从数据实际统计得出。

---

### ISSUE-019
**问题：** 章节分类器实质失效。

**影响：** [已验证] `data/vector_db` 406 chunks 的 `section_type` 分布为 `unknown=308 (75.8%)`、`secondary=70`、`primary=28`。证据质量分层近乎无效，直接拉低重排与选取结果。

**当前状态：** 不影响启动。

---

### ISSUE-020
**问题：** [静态] `per_file_cap`（单篇结果占比上限）被回填循环**完全绕过**：top_k=5/8/12/25 实测单篇占比均为 **100%**。

**影响：** 多样性约束形同虚设。

**当前状态：** 不影响启动。

---

### ISSUE-021
**问题：** RoB 2 偏倚风险评估工具**全仓 0 命中（完全不存在）**；结局层 GRADE 不存在（`GRADEResult` 的 13 个字段无一是 list）；图表能力为 0（无 OCR / 图像抽取）。

**影响：** 产品设想中的「循证平台」与代码实况差距很大 —— 现有代码实为**单篇文献的自动评分卡**。

**当前状态：** 记录为功能缺口，避免后续误认为已实现。

---

## Database / RAG

### ISSUE-022
**问题：** ⚠️ **向量库元数据中烘焙了另一台机器的 Windows 绝对路径**。`data/vector_db` 的 `file_path` 字段为 `D:\OncoImmunoRAG\data\raw\...`，而本项目实际位于 `D:\program\OncoImmunoRAG`。

**影响：** 该路径在**本机就已失效**，在 WSL/Linux 下更是完全不可用。依赖 `file_path` 做溯源/读文件的功能会失败。

**当前状态：** [已验证] 已从 Chroma collection 直接读出。

**建议未来处理：** 入库时写入**相对路径**，读取时按项目根目录拼接。

---

### ISSUE-023
**问题：** ⚠️ **`data/vector_db_22`（24 MB）无法被当前 chromadb 0.6.3 打开** —— 抛 `KeyError: '_type'`（`chromadb/api/configuration.py:209`），提示该库由更旧版本的 chromadb 构建。

**影响：** 24 MB 数据当前**完全不可用**，也无法用现有依赖读取或迁移。

**当前状态：** [已验证] 打开失败。**不要删除**（用户指令：不动现有数据）——需要时用匹配的老版本 chromadb 读取后重新导出。

---

### ISSUE-024
**问题：** `data/vector_db` 的 406 chunks 覆盖 12 篇文献，而 `_PAPERS_RAW` 声明 22 篇。**入库脚本 `scripts/build_kb_22.py` 产出的库与 `data/vector_db` 不一致。**

**影响：** 界面承诺的 22 篇检索能力在数据层不存在。

**当前状态：** 记录，不重建向量库（用户指令：除非启动必须，不重新生成 ChromaDB）。

---

### ISSUE-025
**问题：** `_vector_db_path()` 通过 `resolve_config_path().parent / "data" / "vector_db"` 定位 —— 依赖 `config.yaml` 位置，**强耦合配置与数据布局**。

**影响：** [静态] WSL 侧若 `config.yaml` 位置或大小写与预期不同（例如放在 `/mnt/d/...`），路径解析会失败或解析到错误位置。

**当前状态：** Windows 下可用。

---

## Data / Papers

### ISSUE-026
**问题：** ⚠️ **全部 22 个源 PDF 都被 `.gitignore` 排除** —— `data/raw/*`、`verification/*.pdf`、`holdout/*.pdf`。

**影响：** **这是 Windows → GitHub → WSL 流程的最大阻塞点。** 我已实测：22 个 PDF 在本机**全部存在**（`missing PDFs: 0`），但一个都不会进 Git。WSL 侧 `git clone` 后 `app.py` 的 22 篇文献列表会全部指向不存在的文件，Tab 1/2/3 的核心功能**直接不可用**。

**当前状态：** [已验证] `git check-ignore` 三条规则全部命中。

**建议未来处理（迁移前必须决策）：**
- 方案 A：另建私有仓库 / 私有 release 存放 PDF，仓库内只保留校验和（SHA256）与获取说明。
- 方案 B：用 Git LFS 存 PDF（`.gitattributes` 已把 `*.pdf` 标为 `binary`，但**没有配置 LFS filter**）。
- 方案 C：脚本化下载（多为 Lancet Oncol / NEJM，需机构订阅）。
- ⚠️ **无论哪种方案：先确认这些文献 PDF 的授权状态允许再分发。不要默认可以直接推到公开 GitHub。**

---

### ISSUE-027
**问题：** `data/raw/.gitkeep` 被跟踪，但 `data/vector_db/.gitkeep` / `data/vector_db_22/.gitkeep` 的反向忽略规则（`!….gitkeep`）**实际不存在** —— 两个目录下没有 `.gitkeep` 文件。

**影响：** 新克隆后 `data/vector_db/` 目录不会存在，入库脚本的 `get_or_create` 可能因父目录缺失报错。

**当前状态：** Windows 下目录已存在，不影响。

---

### ISSUE-028
**问题：** 已落盘的评测结果**含有已知错误数据**，例如 ISSUE-014 的 `McGregor-DAD.json`（HR 张冠李戴）。

**影响：** 这些 JSON 是当前的"基线"，若上传 GitHub 会把错误结论固化并传播。

**当前状态：** 记录。`.gitignore` 第 35–41 行**有意不忽略**评测输出（注释说明是刻意为之，等流水线修好后重跑）。

**建议未来处理：** 修复 ISSUE-014 后重跑基线再上传。

---

## Windows → Linux / WSL Compatibility

### ISSUE-029
**问题：** ⚠️ **`.gitignore` 有 3 处漏网，非忽略文件合计约 4.0 MB / 70 个文件**，会被 `git add` 上传：
- `result/` 下 7 张 PNG/JPG + 2 个 HTML + 1 个 SVG（最大单文件 1.4 MB）
- `data/benchmark_results.xlsx`、`result/benchmark_results_4.xlsx`、`tests/results/review/review_table.xlsx`
- `result/*.json`（批量评测原始输出）
- `HANDOFF_TO_WSL.md`（47 KB，**建议上传**，属交接文档）

**影响：** 仓库体积膨胀、diff 噪声；但**均不含敏感凭据**，风险等级低。

**当前状态：** `.gitignore` 第 35–41 行的注释显示这是**刻意的选择**，非疏忽。

**建议未来处理：** 若希望保持仓库精简，可考虑忽略 `result/*.png`、`*.xlsx`；但**这是决策项，不在本阶段修改范围**。

---

### ISSUE-030
**问题：** ⚠️ **`poetry.lock` 被明确排除在 `.gitignore` 之外（第 15–17 行注释：锁文件必须入库以保证可复现）**，但**迁移前的回滚副本在仓库外**。

**影响：** `D:/Python/poetry.lock.before-depfix.bak` 是 ISSUE-001 那次依赖修复的**唯一回滚副本**，不受 Git 保护。丢失后无法回到 171 包的原始锁定状态。

**当前状态：** 该备份**存在且在仓库外**（不会被误传 GitHub，这点是安全的）。但需确保 WSL 迁移前该文件未被清理。

**建议未来处理：** 另存一份到私有位置。

---

### ISSUE-031
**问题：** `.venv/`（1.3 GB）已在 `.gitignore`，**不应搬运**到 WSL。

**影响：** WSL 侧必须重建环境。

**当前状态：** 已忽略，正确。

**建议未来处理：** WSL 侧用 `pipx install poetry`，不要把 Poetry 装进项目 venv（Windows 侧这样做会额外引入 28 个工具包并牵动 urllib3 等版本）。

---

### ISSUE-032
**问题：** `poetry.lock` 中含 **18 个 Linux CUDA 包**的条目，在 Windows 上按 marker 正确跳过。

**影响：** 迁移到 WSL 后这些包会**实际安装**（torch CUDA 轮子，数 GB），依赖解析结果与 Windows 侧**不完全相同**。若 WSL 侧无 NVIDIA 驱动/环境，会安装失败或回退到 CPU 轮子。

**当前状态：** Windows 侧已验证可安装（torch 2.13.0+cpu，CUDA 不可用）。

**建议未来处理：** WSL 侧先确认 `nvidia-smi` 可用再决定是否装 CUDA 版 torch。

---

### ISSUE-033
**问题：** **凭据卫生状况良好 —— 未发现泄漏。** [已验证]

具体核查结果：
- `config.yaml`（**已被 Git 跟踪**）：**不含任何凭据**，且第 3–6 行有显式警告「DO NOT put a credential in this file — it is versioned」。
- 全仓正则扫描 `sk-*` / `eyJ*`（JWT）/ `api_key=<长串>`：**0 命中**。
- `.env.example` 中所有值均为空或示例。
- 未发现密码、token、连接串。
- 70 个待上传文件中无凭据。

⚠️ **本轮变化：** 磁盘上现在**存在 `.env`**（用户授权我写入以解除 ISSUE-005 门控），其中含明文 API Key。已用 `git check-ignore -v .env` 确认命中 `.gitignore:49`，**不会上传**。但这是本机上的明文凭据文件。

**⚠️ 待你处理：**
1. 在 MiniMax 控制台**轮换该 Key**（记忆中记载 `config.yaml` 曾经也存过明文 key）。我不代为操作、不回显任何 Key、**不假定你已轮换**。
2. 轮换后删除 `D:\program\OncoImmunoRAG\.env`。
3. 若使用 `git add -A` 之类的宽泛命令前，务必再确认一次 `.env` 仍被忽略。

---

### ISSUE-034
**问题：** 未发现**患者隐私数据**。[已验证]

待上传的评测 JSON 均为**文献级抽取结果**（试验名、查询词、检索到的文件名/chunk_id/分数、GRADE 评级），例如 `tests/results/grade_eval/KEYNOTE-057.json` 的字段为 `scenario/title/query/started_at/retrieved_chunks/grade/error`。不含患者姓名、住院号、个体记录。

**当前状态：** 源 PDF 本身是公开发表的临床试验文献，且已被 `.gitignore` 排除（但见 ISSUE-026 的授权问题）。

---

## 协作与工程化（2026-10-23 审计新增）

### ISSUE-035
**问题：** `src/retrieval/pubmed_client.py` **直接 `import requests`**，但 `requests` 不在 `pyproject.toml` 的依赖声明中。

**影响：** 能运行纯粹是因为其他包传递依赖把它带进来了。任何上游包移除该依赖，Tab 2 的 PubMed 检索即崩溃。

**当前状态：** ✅ **已修复。** 已加入 `pyproject.toml`（`requests>=2.31.0,<3.0.0`）。执行 `poetry lock` 后**锁文件仅 content-hash 变化，零个包变动**，现装版本仍为 2.34.2。回归验证：15 passed，11 个模块全部 import 成功。**未提交。**

---

### ISSUE-036
**问题：** **文献目录存在两个独立真相源，且使用两套不同的受控词表。**

- `app.py:70-93` `_PAPERS_RAW`（22 条，含 `design_gold`）—— 仅用于界面显示
- `src/utils/trial_registry.py:69` `TRIAL_REGISTRY`（6 条，含 `study_design`）—— **参与评分，覆盖 LLM 输出**

实测重叠的 6 条中 **5 条字面量不一致**：

| 文件 | `_PAPERS_RAW` | `TRIAL_REGISTRY` |
|---|---|---|
| Zhou_2025_..._RC48-C014.pdf | `单臂` | `单臂试验` |
| McGregor_2024_AnnOncol_DAD.pdf | `单臂` | `单臂试验` |
| Boorjian_2021_..._Nadofaragene.pdf | `单臂` | `单臂试验` |
| Szabados_2022_..._ABACUS.pdf | `单臂` | `单臂试验` |
| Li_2025_..._Cretostimogene-Nivo.pdf | `单臂` | `单臂试验` |
| Pfister_2024_..._VESPER.pdf | `RCT` | `RCT` ← 唯一一致 |

**影响：** 新增文献时必须记得同时改两处，遗漏不会报错，只会静默不一致。

**当前状态：** 不影响启动。**已记录，未修。**

---

### ISSUE-037
**问题：** `design_gold`（界面上的「金标准设计」）**从不参与任何校验**。全仓仅 2 处引用，均在 `app.py`：第 103 行赋值、第 372 行显示。

**影响：** 界面上呈现为「金标准」的设计分类，实际上**没有任何代码将其与 LLM 输出比对**。这是装饰性标签，会给读者错误的可信度暗示。

**当前状态：** 不影响启动，**影响结果可信度**。已记录，未修。

---

### ISSUE-038
**问题：** ⚠️ **`src_00/` 是 `src/` 的陈旧副本，且曾处于已暂存状态。**

已验证差异：缺 `retrieval/pubmed_client.py` 与 `retrieval/temp_indexer.py`；`selector.py` 277 行 vs 现版 502 行（缺整个 evidence-grade rerank 层）；`query_engine.py` 538 行 vs 现版 587 行。

**影响：** 若被提交，会把过时源码树写入 Git 历史；任何代码搜索或 AI 代理会同时命中两份，产生混淆。`ISSUE-008`（`query()` 缺失）在两份中都有。

**当前状态：** ✅ **已处理** —— 2026-10-23 经负责人确认，`git restore --staged src_00/` 已从暂存区移除，磁盘文件保留；并已加入 `.gitignore` 防止再次误提交。

---

### ISSUE-039
**问题：** `scripts/` 下 **8 个绘图脚本当前无法运行** —— `matplotlib`、`plotly`、`seaborn` 均未安装（实测 `ModuleNotFoundError`），且三者都不在 `pyproject.toml` 中声明。

受影响：`plot_confusion_matrix`、`plot_error_sankey`、`plot_grade_traffic_light`、`plot_knowledge_graph_mpl`、`plot_optimization`、`plot_performance_optimization`。

**影响：** `result/` 目录下已有的 PNG/HTML 图表**无法在本环境复现** —— 它们是在另一套环境（装了绘图库）生成的。因此这些图表的产出流程在 Windows 当前基线下**不可复现**。

**当前状态：** 不影响主应用运行。已记录，未修（新增依赖需批准）。

---

### ISSUE-040
**问题：** **全仓 0 处使用 `logging`**，全部使用 `print()`（共 47 处，分布：`app.py` 无、`grade_evaluator` 4、`append_pdfs` 11、`pdf_parser` 6、`table_extractor` 5、`vector_store` 5、`pubmed_client` 2、`query_engine` 11、`config_loader` 1、`trial_registry` 1、`cli` 1）。

**影响：** 测试无法用 `caplog` 捕获输出；长流程失败无法定位；脚本重定向拿不到结构化日志。

**当前状态：** 不影响运行。`docs/coding_standards.md` 第 8 节已规定「新代码不得引入 logging 以免混用」，全仓统一需单独任务。

---

### ISSUE-041
**问题：** **全仓 0 处设置随机种子**（`random.seed` / `np.random.seed` / `seed=` 全部无命中）。

**影响：** 涉及采样的结果不可复现。叠加 LLM 温度由服务端决定这一事实，`scripts/run_batch_*.py` 产出的结果每次运行都可能不同，**当前无法保证「改造前后结果一致」**。

**当前状态：** 不影响运行，但**影响回归验证的可靠性**。已记录。

---

### ISSUE-042
**问题：** `AppConfig` **没有 `llm_base_url` / `llm_model` 属性**，二者存放在 `AppConfig.extras` 字典中（实测 `extras` 键为 `llm_base_url`、`llm_model`）。

**影响：** 直觉写法 `cfg.llm_base_url` 抛 `AttributeError`（本轮实测踩中）。新成员和 AI 代理极易写错。

**当前状态：** 不影响运行。已在 `docs/interfaces.md` 与 `docs/coding_standards.md` 中标注。

---

### ISSUE-043
**问题：** 环境变量优先级**不对称**，且方向与直觉相反：`OPENAI_API_KEY` 是「环境变量 > config.yaml」，但 `llm_base_url` / `llm_model` 是「**config.yaml > 环境变量**」。

**影响：** 修改 `.env` 中的 `LLM_BASE_URL` / `LLM_MODEL` **完全无效**。`.env.example` 已注明这点，但极易被忽略，表现为「改了配置没生效」。

**当前状态：** 不影响运行。已在 `AGENTS.md` 与 `docs/coding_standards.md` 中标注。

---

### ISSUE-044
**问题：** **路径解析惯例重复 5 次**，且存在**第 6 种不一致写法**。

5 处统一用 `resolve_config_path().parent / "data" / X`：`grade_evaluator.py:242`、`append_pdfs.py:90`、`pdf_parser.py:160`、`vector_store.py:42`、`query_engine.py:102`。
第 6 处 `table_extractor.py:317,321` 用 `Path(__file__).resolve().parents[2]`，**绕过 `config_loader`**。

**影响：** 未来若要支持配置数据目录位置，需改 6 处，漏改即行为不一致。

**当前状态：** 不影响运行。已记录。

---

### ISSUE-045
**问题：** **分层倒置**：`retrieval/temp_indexer.py` 导入 `grading/grade_evaluator.py` —— 检索层依赖评分层。

**影响：** 评分层的任何改动都会波及检索层；`grade_evaluator`（1587 行）成为多个方向的共同依赖。

**当前状态：** 不影响运行。已在 `ARCHITECTURE.md` 依赖图中以虚线标注。**不擅自重构。**

---

### ISSUE-046
**问题：** `except Exception` 共 **18 处**（`app.py` 10、`grade_evaluator` 3、`query_engine` 3、`selector` 1、`temp_indexer` 1），**无一处记录堆栈或区分异常类型**。

**影响：** 长流程失败时无法定位到具体环节；`app.py:169-170` 把异常转成 `{"error": "..."}` 字符串塞进结果，调用方无法区分错误类型。

**当前状态：** 不影响运行。已记录。

---

### ISSUE-047
**问题：** `vector_store.build_index()` 的 `reset` 参数**默认值为 `True`**，会调用 `_reset_vector_db(path)` **删除并重建整个 `data/vector_db`**。

**影响：** 任何未显式传 `reset=False` 的调用都会销毁现有索引。误用代价高（需 1.3 GB 模型 + 全量重算）。

**当前状态：** 不影响当前运行。**已在 `docs/interfaces.md` 与 `CONTRIBUTING.md` 中标注为高危副作用。**

---

### ISSUE-048
**问题：** `retrieval.temp_indexer` 的 `TempIndexer.retrieve(question, top_k=5)` **接受 `top_k` 参数**，而 `QueryEngine.retrieve()` **不接受**（`top_k` 是构造参数）。

**影响：** 同一「检索」概念在两个模块有**相反的接口形态**，调用方极易混淆并写错。

**当前状态：** 不影响运行。已在 `docs/interfaces.md` 中并列标注两者的真实签名。

---

| 项目 | 状态 |
|---|---|
| Streamlit 页面启动 | ✅ **可用**，`http://localhost:8501`，HTTP 200，AppTest 0 异常 |
| 3 个主 Tab + 2 个子 Tab 渲染 | ✅ 可用 |
| 文献下拉框 | ✅ 22 个选项，本机 22 个 PDF **全部存在**（missing: 0） |
| 8 个 `src/` 模块 import | ✅ 全部成功 |
| 测试基线 | ✅ **15 passed** |
| 嵌入模型加载 | ✅ 已下载，`QueryEngine.from_config()` 冷启动 58.8s |
| **向量检索 retrieve()** | ✅ **可用** —— 0.53s，MMR over-fetch + canonical routing + rerank + select_top_k 全链路跑通，返回带 `file_name`/`section_type`/`score` 的 hits |
| **LLM 分级 grade_extraction()** | ✅ **可用** —— 单次 15.2s，UI 按钮端到端 10.3s，输出 13 个字段（`overall_grade`/`composite_score`/`effect_size`/`grade_reasoning` 等） |
| UI 端到端点击 | ✅ 点击「评估…」按钮 → 0 异常 → 渲染 GRADE 结果 |
| LLM 问答 / 跨研究综合 | ❌ **不可用**（ISSUE-008 实测确认无 `query()` 方法） |
| CLI 入口 | ❌ 占位符（README 的 `poetry run oncoimmuno-rag` 无效） |
| `data/vector_db_22` | ❌ **损坏不可读**（老版 chromadb 构建，`KeyError: '_type'`） |
| 向量库元数据路径 | ⚠️ 烘焙了 `D:\OncoImmunoRAG\...`（他机路径，本机即失效） |
| 界面「22 篇」宣称 | ⚠️ 与数据不符：向量库实际只有 12 篇 / 406 chunks |
| 章节分类 | ⚠️ 406 chunks 中 `unknown` 占 **75.8%** |
| 评分可信度 | ⚠️ 实测 LLM=7.0 / Python=2.0 覆盖冲突（ISSUE-013） |
| WSL 迁移就绪度 | ❌ **未就绪** —— 22 个 PDF 全部不会随 Git 走（ISSUE-026） |

**结论：这是一个「Web 界面 + 单篇检索 + 单篇自动分级」完整可跑，但「跨研究综合」不可用、且评分与数据存在系统性缺陷的基线版本。**

**已实机验证的核心链路**：`Streamlit → PDF 解析 → 分块 → grade_extraction → MiniMax LLM → GRADE 结果渲染`，端到端 10.3s。

**未修复任何代码。** 本轮唯一的写操作是：创建 `.env`（gitignored）与 `CURRENT_ISSUES.md`。

---

## 测试体系补强（2026-10-23 第三步新增）

### ISSUE-049
**问题（本轮补测发现）：** `retrieval/selector.py` 的 `__all__` **列出了私有常量却漏掉了公共函数** —— 包含 `_DEFAULT_TOP_K`、`_TABLE_SUMMARY_KEEP_SMALL`、`_TABLE_SUMMARY_KEEP_BOUNDARY`、`_CANONICAL_PRIMARY_RESERVE_*` 等，而公共的 `rerank_hits`、`get_rerank_enabled_default`、`set_rerank_enabled_default` **均不在其中**。

**影响：** `from retrieval.selector import *` 拿不到公共的 rerank 相关函数。API 声明与实现不一致。

**当前状态：** 已用 `tests/test_selector.py::test_public_symbols_are_exported_in_dunder_all` 以 **strict xfail** 记录 —— 修复时它会变成 XPASS 并提醒同步文档。**未修**（需改业务代码）。

---

### ISSUE-050
**问题：** `tests/test_config_smoke.py` 与 `tests/test_new_deps_smoke.py` 名为 `test_*.py`，实际**只定义了 `main()` 或模块级 `print`，贡献 0 个测试用例**。

**影响：** 团队看到「有 smoke 测试」会产生虚假的安全感，实际什么都没测。

**当前状态：** ✅ **已修复**（属测试代码，非业务代码）。两个文件改写为真实用例：`test_config_smoke.py` 保留原有断言（chunk_size=1024、chunk_overlap=200、embedding_model），`test_new_deps_smoke.py` 改为 5 个用例并逐项核对 `pyproject.toml` 声明的运行时依赖是否已安装。**未提交。**

---

### ISSUE-051
**问题：** `tests/test_new_deps_smoke.py` 在**模块级** `import sentence_transformers`（进而拉起 torch）。pytest 在收集阶段 import 全部 `test_*.py`，因此**任何缺少 torch 的环境会导致整个测试运行 error，而不是 skip**。

**影响：** CI 或新成员机器上缺 torch 时跑不出任何结果，且报错指向「收集错误」而非「缺依赖」，排查成本高。

**当前状态：** ✅ **已修复** —— 第三方 import 移入测试函数内部，并打 `@pytest.mark.requires_model`，可用 `-m "not requires_model"` 排除。**未提交。**

---

### ISSUE-052
**问题（本轮补测发现）：** `PubMedHit.year` 与 `PubMedDetail.year` 声明为 `year: str`，实际被赋成 **list**：
- `src/retrieval/pubmed_client.py:268` — `hit.year = pubdate.split()[:1] or [""]`
- `src/retrieval/pubmed_client.py:183` — `detail.year = pubdate.get("Year") or (...).split()[:1] or [""]`

`split()[:1]` 返回列表而非元素。

**影响：** 实测 `hit.year == ['2023']`。`PubMedHit.short_label()` 直接把 `year` 插入 f-string，界面会渲染成 `Balar AV ['2023'] (Lancet Oncol)`。任何 `hit.year.startswith(...)` 之类调用会崩溃。

**当前状态：** 已用 `tests/test_pubmed_client_contract.py::test_hit_year_is_a_string` 以 **strict xfail** 记录。**未修**（需改业务代码）。

---

### ISSUE-053
**问题：** `docs/interfaces.md` 初次编写时把 `AppConfig.has_openai_api_key` 记为方法，实际是 **`@property`**。

**影响：** 按文档写 `cfg.has_openai_api_key()` 会抛 `TypeError: 'bool' object is not callable`。

**当前状态：** ✅ **文档已更正**，并新增 `test_has_openai_api_key_is_a_property_not_a_method` 固化。

⚠️ **方法论教训：** 用 AST 批量提取签名时**不会区分 property 与方法**（两者都是 `FunctionDef`）。接口文档的人工复核必须覆盖这一维度。

---

### ISSUE-054
**问题：** 6 个 benchmark 脚本在 API Key 缺失时**提示用户把凭据写进 `config.yaml`**：
```
scripts/run_batch_3.py:454          "OPENAI_API_KEY is empty — set it in config.yaml ..."
scripts/run_rerank_ab_holdout.py:433 "OPENAI_API_KEY 未配置 — 请在 config.yaml / env 中设置"
```
而 `config.yaml` **已被 Git 跟踪**，其第 3–6 行明确写着「DO NOT put a credential in this file — it is versioned」。

**影响：** 新成员照脚本提示操作 → 密钥写进被跟踪文件 → **`git commit -a` 即泄漏**。脚本提示与配置文件自我警告直接矛盾。

**当前状态：** 记录，**未修**（需逐个改 6 个脚本文件，属业务代码）。

**附带：** `run_batch_3`、`run_batch_4`、`run_rerank_ab_holdout` **没有 `--dry-run` 逃生阀**（另外 3 个有），无 Key 时只能中止或改配置文件。

---

### ISSUE-055
**问题：** `.gitignore` 的规则写成 `data/raw/*`（忽略**内容**而非目录本身），所以 `git check-ignore data/raw` **返回"未忽略"**。

**影响：** 任何按目录名做忽略校验的脚本/CI 检查都会**误报失败**。本轮编写 CI 校验时实测踩中。

**当前状态：** ✅ CI 中已改用代表性文件路径（`data/raw/example.pdf` 等）并**本地实测通过**。**未修改 `.gitignore`** —— 目录级忽略会让 `data/raw/.gitkeep` 这类的占位文件跟踪失效。

---

### ISSUE-056
**问题：** `config_loader.py:35` 与 `table_extractor.py:317` 用 `Path(__file__).resolve().parents[2]` 定位项目根，**隐含假设源码树恰好在仓库根下两层**。

**影响：** 若 `pyproject.toml` 的 `package-mode` 改为 `true`，所有数据路径会指向 `site-packages`，**静默失败**（不报错，只是找不到数据）。

**当前状态：** 当前 `package-mode = false`，安全。记录为架构约束。

---

### 补测后的测试基线（2026-10-23）

| 项 | 补测前 | 补测后 |
|---|---|---|
| 用例总数 | 15 | **108 passed + 5 xfailed** |
| `grade_evaluator`（1587 行） | **0 覆盖** | 18 个用例（mock LLM） |
| `query_engine`（587 行） | **0 覆盖** | 10 个用例（接口契约） |
| `selector`（502 行） | **0 覆盖** | 18 个用例 |
| `trial_registry`（207 行） | **0 覆盖** | 11 个用例 |
| `pubmed_client`（299 行） | **0 覆盖** | 13 个用例（mock requests，零网络） |
| 流水线集成 | 无 | 12 个用例（合成文档，零 PDF 依赖） |
| 原有 15 个用例 | 15 passed | **15 passed，行为无变化** |

5 个 xfail 全部是**已知缺陷的可执行记录**：ISSUE-008、ISSUE-015、ISSUE-019、ISSUE-049、ISSUE-052。修复对应缺陷时会转为 XPASS 并提示同步文档。