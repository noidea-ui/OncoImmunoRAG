# CONTRIBUTING.md — 新成员上手指南

本指南的目标：**让第一次接触本项目的人，能独立完成一个最小开发任务并提交 PR。**

预计耗时：环境搭建 15 分钟，通读文档 30 分钟，最小任务 30 分钟。

---

## 0. 开始之前必读

| 文档 | 何时读 |
|---|---|
| [`AGENTS.md`](../AGENTS.md) | **先读这个。** 硬约束与禁止事项 |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | 想理解整体结构时 |
| [`docs/interfaces.md`](interfaces.md) | 动手改任何公共接口前 |
| [`docs/coding_standards.md`](coding_standards.md) | 写第一行代码前 |
| [`docs/testing.md`](testing.md) | 写测试时 |
| [`CURRENT_ISSUES.md`](../CURRENT_ISSUES.md) | **动手前必读** —— 避免重复修复已知缺陷 |
| [`docs/TASK_TEMPLATE.md`](TASK_TEMPLATE.md) | 领任务时 |

---

## 1. 环境搭建

### 前置

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | **3.12.x** | ⚠️ `pyproject.toml` 声明 `>=3.10`，但锁定的 `numpy 2.5.1` 要求 `>=3.12`，**3.10/3.11 装不上** |
| Poetry | 2.x | 建议用 `pipx install poetry`，不要装进项目 venv |
| Git | 任意 | |
| 磁盘 | ≥5 GB | 嵌入模型 1.3 GB + 依赖 |

### 不需要

Node.js、Conda、CUDA、Docker、数据库服务。

### 步骤

```bash
git clone <repo-url>
cd OncoImmunoRAG

# 创建虚拟环境（Linux/WSL）
python3.12 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
# python -m venv .venv
# .venv\Scripts\Activate.ps1

# 安装依赖
poetry install

# 验证
.venv/bin/python -m pytest tests/ -o addopts="" -q    # 期望：15 passed
```

⚠️ `poetry install` 后必须用 `-o addopts=""` 跑测试 —— 默认的 `--cov=oncoimmuno_rag` 指向**不存在的包**，覆盖率门禁会静默失效。

### 凭据

```bash
cp .env.example .env      # Windows: copy .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY
```

⚠️ **绝对不要把密钥写进 `config.yaml`** —— 该文件**已被 Git 跟踪**。
⚠️ **不要提交 `.env`** —— 已在 `.gitignore` 中。

优先级提醒：`OPENAI_API_KEY` 环境变量优先于 `config.yaml`；但 `llm_base_url` / `llm_model` **反过来**，`config.yaml` 优先于环境变量。

---

## 2. 运行项目

```bash
set -a; . ./.env; set +a          # 加载密钥
.venv/bin/python -m streamlit run app.py
```

打开 **http://localhost:8501**。

| 现象 | 含义 |
|---|---|
| 页面正常但检索/评分无反应 | 没有 `OPENAI_API_KEY`。`get_query_engine()` 静默返回 `None` |
| 首次操作很慢（~60s） | 正在下载 1.3 GB 嵌入模型 `BAAI/bge-large-en-v1.5`，只需一次 |
| 文献下拉框有 22 项但都报错 | 22 个 PDF **不在 Git 中**，需另行获取（见下） |

⚠️ **`poetry run oncoimmuno-rag` 是错的** —— `src/cli.py` 是占位符，无可用命令。

### 论文数据

**22 个 PDF 被 `.gitignore` 排除，不会随仓库分发。** 分布：

| 目录 | 数量 | 覆盖 Tab |
|---|---|---|
| `data/raw/` | 12 | Tab 1（库内检索） |
| `verification/` | 5 | Tab 1/3 |
| `holdout/` | 5 | Tab 1/3、holdout 基准 |

未来由专用服务器提供。**不要把这些 PDF 提交到 GitHub。**

---

## 3. Git 协作流程

```
main ──→ feature/<任务ID>-<简述> ──→ 本地测试 ──→ PR ──→ CI ──→ 审查 ──→ 合并
```

### 硬性规则

1. **禁止在 `main` 上直接开发。**
2. 一个 PR 只解决**一个**明确问题。
3. **禁止未经批准的大范围格式化或重构**（尤其 `ruff format` 全仓、`git add .`、`git reset`、`git push`）。
4. **发现无关缺陷 → 记入 `CURRENT_ISSUES.md`，不顺手修。**
5. 涉及接口的 PR **必须**同步更新 `docs/interfaces.md` 并补测试。

### 分支命名

```
feature/T-012-selector-contract-tests
fix/T-013-config-loader-raise-flag
docs/T-014-interfaces-update
```

### 提交信息

```
<type>: <简述> (<任务ID>)

<为什么这样改，一到三行>

任务ID: T-012
```

`type` ∈ `feat` / `fix` / `test` / `docs` / `refactor` / `chore`

---

## 4. 领取任务

1. 认领 `docs/TASK_TEMPLATE.md` 中的一个任务，**先复制一份到 `docs/tasks/T-xxx.md`** 填写。
2. 确认「允许修改的文件」清单 —— **超出清单即越权**。
3. 若发现接口不足以完成任务，**先提接口变更建议**，不要自行改公共签名。

---

## 5. 修改模块

### 先看契约

改任何公共函数前，查 [`docs/interfaces.md`](interfaces.md) 的「调用方清单」，确认影响面。

### 最容易踩的三个坑

| 坑 | 后果 |
|---|---|
| `QueryEngine.retrieve(q, top_k=10)` | **抛 `TypeError`**。`top_k` 是构造参数，不是方法参数 |
| `hit.get("file_name")` | **失败**。元数据在 `hit.metadata["file_name"]` |
| `hit.get("risk_of_bias")` 之类按对象访问 | `risk_of_bias` 是**裸字符串** `"高"`，不是字典 |

### 高危副作用

| 调用 | 风险 |
|---|---|
| `vector_store.build_index()` | `reset=True` 是**默认值**，会**删除并重建** `data/vector_db` |
| `clear_temp_collection()` | 删除临时 collection |

### 绝对禁止

- ❌ 修复 `CURRENT_ISSUES.md` 中已记录的缺陷（除非该缺陷就是你的任务）
- ❌ 改业务逻辑、改阈值、改模型、改数据库
- ❌ 重写 `grade_evaluator.py` 的评分规则
- ❌ 触碰 `src_00/`（陈旧副本，已 gitignore）
- ❌ 提交密钥、PDF、数据库、模型权重

---

## 6. 运行测试

```bash
# 基线（唯一必跑）
.venv/bin/python -m pytest tests/ -o addopts="" -p no:cacheprovider -q

# 单个文件
.venv/bin/python -m pytest tests/test_selector.py -o addopts="" -v
```

**回归红线：原有 15 个用例必须全部通过且行为不变。** 详见 `docs/testing.md`。

---

## 7. 提交代码

```bash
git status --short              # 先看清楚，不要用 git add .
git add <明确的文件路径>          # 逐个添加
git diff --cached                # 提交前 review
git commit -m "..."
git push -u origin feature/T-012-xxx
```

⚠️ **禁止 `git add .` / `git add -A`** —— 会误提交 `result/`、`data/` 等本不该入库的内容。

提交前确认：
```bash
git diff --cached --name-only    # 逐行确认没有密钥、PDF、数据库
```

---

## 8. 发起 Pull Request

PR 描述必须包含：

```markdown
## 目的
解决什么问题（关联任务 ID / Issue）

## 变更范围
- 改了哪些文件
- 为什么改这些

## 测试结果
- 命令 + 实际输出（**必须粘贴真实输出，不得只写"通过"**）
- 若有测试未运行，必须说明原因

## 兼容性影响
- 是否改动公共接口？是 → 必须已更新 docs/interfaces.md

## 未解决问题
- 本次没解决的、已知遗留的问题
```

⚠️ **没跑的测试必须明说。** 「全部通过」与「跑了一部分」要区分清楚。

---

## 9. 处理冲突

```bash
git fetch origin
git rebase origin/main
# 解决冲突后
git rebase --continue
```

| 冲突场景 | 处理 |
|---|---|
| 业务代码冲突 | **停下，找负责人。** 不要自行取舍 |
| `poetry.lock` 冲突 | **不要手改。** 重新 `poetry lock` 并 review diff |
| `docs/interfaces.md` 冲突 | 合并双方条目，保留更新的签名 |

⚠️ 涉及 `grade_evaluator.py` 评分规则、`prompts.py` 提示词的冲突，**必须人工协商**，这类改动影响所有已落盘的评测结果。

---

## 10. 更新文档

| 你改了什么 | 必须同步更新 |
|---|---|
| 公共函数签名 | `docs/interfaces.md` |
| 新增模块 | `ARCHITECTURE.md` 的依赖图与职责表 |
| 新增依赖 | `pyproject.toml` + `CURRENT_ISSUES.md`（需批准） |
| 新发现缺陷 | `CURRENT_ISSUES.md` |
| 新的测试入口 | `docs/testing.md` |

---

## 11. 最小上手任务（照着做即可完成第一次 PR）

**任务：为 `retrieval/selector.py` 的 `select_top_k` 补单元测试**

1. 建分支：
   ```bash
   git switch -c feature/T-100-selector-contract-tests
   ```
2. **不要修改 `selector.py`** —— 该模块无测试，纯函数，可直接测。
3. 新建 `tests/test_selector.py`，测试：
   - `canonical_file` 指定的文献的 chunk 被强制提升到 top_k 内
   - 传入的 hits 数少于 `top_k` 时返回全部，不报错
   - 空输入返回空列表
4. 跑测试：
   ```bash
   .venv/bin/python -m pytest tests/ -o addopts="" -q
   ```
   期望：`17 passed`（原 15 + 新 2）
5. 更新 `docs/testing.md` 的测试清单。
6. 提交并开 PR，描述贴上真实输出。

这个任务完成后，你就熟悉了：环境 → 分支 → 读契约 → 写测试 → 验证 → 提 PR 的完整流程。