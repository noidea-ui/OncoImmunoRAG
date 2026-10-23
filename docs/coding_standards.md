# docs/coding_standards.md — 代码规范

> 目标：在**不改动现有业务逻辑**的前提下统一风格。
> **核心原则：规范只约束新增/修改的代码，存量代码不要求整改。**

---

## 0. 工具选型（不引入新依赖）

| 用途 | 工具 | 现状 |
|---|---|---|
| 检查 + 格式化 | **Ruff**（已在 dev 组） | 已配置，**存量 248 errors 不要求修复** |
| 测试 | **pytest**（已在 dev 组） | 基线 15 passed |
| 类型检查 | **不引入 mypy/pyright** | 类型标注覆盖率已 96%，运行时无强制检查 |

⚠️ **不要为了满足格式检查而改变业务逻辑。** `ruff --fix` 可能重写逻辑，必须逐条 review。

---

## 1. Ruff 配置（已在 `pyproject.toml`）

```toml
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "W", "B", "UP"]
```

| 要求 | 范围 |
|---|---|
| 改动的文件 `ruff check` 零错误 | ✅ 必须 |
| 全仓 `ruff check` 零错误 | ❌ 不要求（存量 248） |
| `ruff format` 全仓格式化 | ❌ **禁止**（47 文件需重排，会污染 diff） |

---

## 2. 命名

| 元素 | 规范 | 示例 |
|---|---|---|
| 模块 / 函数 / 变量 | `snake_case` | `extract_table_summaries` |
| 类 | `PascalCase` | `GRADEResult`、`QueryEngine` |
| 常量 | `UPPER_SNAKE` | `_DEFAULT_TOP_K`、`_DEFAULT_COLLECTION_NAME` |
| 私有 | 前缀 `_` | `_retrieve_canonical_chunks` |
| 测试文件 | `test_<模块>.py` | `test_selector.py` |
| 测试函数 | `test_<行为>` | `test_select_top_k_promotes_canonical` |

⚠️ **不要重命名现有公共符号**（会破坏 `docs/interfaces.md` 的调用方清单）。

---

## 3. 导入顺序（ruff `I` 规则）

标准顺序：**stdlib → 第三方 → 本地**，组间空一行。

```python
import sys
from pathlib import Path

from llama_index.core import Document

from utils.config_loader import AppConfig, load_config
```

本地包以 `ingestion` / `retrieval` / `grading` / `utils` 开头。

---

## 4. 类型标注

- **新增函数必须补全**参数与返回值标注（存量已达 96%）。
- 跨模块边界的返回值**必须**标注。
- 可选参数用 `X | None = None`，不要用可变默认值。
- ⚠️ **`Literal` 联合类型要保持精确**：`study_design` 只接受 `['RCT','观察性','系统评价','Meta分析','单臂试验']`。新增取值前先查 `docs/interfaces.md`。

---

## 5. 路径处理

**现有两种写法并存，均为遗留，勿模仿：**

```python
resolve_config_path().parent / "data" / "raw"     # 出现 5 处
Path(__file__).resolve().parents[2] / "data"      # table_extractor 独有
```

| 规则 | 说明 |
|---|---|
| 新代码用 `pathlib.Path` | 禁止裸字符串拼接路径 |
| **禁止硬编码绝对路径** | 不写 `C:\` `D:\` `/home/` `/mnt/` |
| 优先 `resolve_config_path()` 派生 | 保持与现有 5 处一致 |
| Windows/Linux 兼容 | 禁止 `\` 作分隔符；禁止依赖盘符 |

⚠️ **已知跨平台问题**：`data/vector_db` 的 chunk 元数据中 `file_path` 已烘焙 `D:\OncoImmunoRAG\...`（他机路径，本机即失效）→ ISSUE-022。**入库代码未来应写相对路径**，但改动需单独批准。

---

## 6. 配置读取

```python
cfg = load_config()                       # 默认读 <项目根>/config.yaml
key = cfg.effective_openai_api_key()      # 内部处理 env > yaml 优先级
```

| 规则 | 说明 |
|---|---|
| 优先 `load_config()`，**不要自己读 yaml** | 否则会绕过优先级逻辑 |
| `llm_base_url` / `llm_model` 在 `cfg.extras` 里 | ⚠️ `AppConfig` **没有**这两个属性 |
| **禁止把密钥写进 `config.yaml`** | 该文件已被 Git 跟踪 |
| 密钥只走环境变量 | 本地可放 `.env`（已 gitignore，代码不自动加载） |

---

## 7. 异常处理

| 规则 | 说明 |
|---|---|
| **禁止裸 `except:`** | 现有代码已无裸 except，保持 |
| 捕获具体异常类型 | `except ValueError:` 优于 `except Exception:` |
| 必须捕获 `Exception` 时**要记录** | 现有 18 处均未记录 —— 新代码不得模仿 |
| 重新抛出时用 `raise ... from exc` | 保留因果链 |
| **不要静默吞异常** | 现有 `pdf_parser` 的 `except Exception: txt = ""` 是遗留问题 |

---

## 8. 日志

⚠️ **现状：全仓 0 处使用 `logging`，全部 `print()`（47 处）。** 这是已知工程缺陷。

**新代码规则**：

| 场景 | 做法 |
|---|---|
| 可诊断的失败路径 | 用 `print()` 与周边代码保持一致（**本阶段不引入 logging**，避免大范围改动） |
| 引入 logging | 需先批准并全仓统一，**不得只在新模块用** |
| 禁止 | 在库代码里 `print()` 到 stdout 污染 CLI 输出（现存问题） |

---

## 9. 随机性与可复现

⚠️ **现状：全仓 0 处设置随机种子。**

| 规则 | 说明 |
|---|---|
| 新增涉及采样/排序打乱的代码**必须**接受 `seed` 参数 | 默认值固定，不用 `None` |
| 产出落盘结果时**必须**把 seed 写进结果元数据 | 否则结果不可复现 |
| 现有脚本调用 LLM 的**天然不确定性**：LLM 温度由服务端决定，`run_batch_*` 类脚本结果每次可能不同 —— 需要可比性时应固定并记录输入 |

---

## 10. 资源释放

```python
with PdfReader(path) as r: ...
client.close()      # chromadb / requests session
```

| 资源 | 规则 |
|---|---|
| Chroma client | 进程级复用，不在函数内反复创建 |
| HTTP session | `requests` 使用长连接 |
| 临时 collection | `TempIndexer` 必须 `close()`；退出前 `clear_temp_collection()` |
| 嵌入模型 | 通过 `@st.cache_resource` 复用，**不要每次重建**（冷启动 58.8s） |

---

## 11. 注释与文档字符串

| 规则 | 说明 |
|---|---|
| 注释语言 | **中文**（与现有代码一致） |
| docstring 语言 | **英文**（与现有代码一致，模块级 docstring 尤其如此） |
| 注释解释**为什么**，不解释**做了什么** | 代码自解释的不写 |
| 文档字符串用 reST 风格 | `Args:` / `Returns:` / `Raises:` |
| 保留关键的**设计决策注释** | 如 `query_engine.py` 中 canonical routing 的成因说明 —— 不要在重构时删掉 |

⚠️ 当前存在一处文档与代码不符：`query_engine.py` 模块 docstring 宣称 `from retrieval.query_engine import QueryEngine, query` 可用，但 **`query` 实际不存在** → ISSUE-008。修复该缺陷时应同步修正 docstring。

---

## 12. 提交前自检

```powershell
.venv\Scripts\python.exe -m ruff check <改动的文件>
.venv\Scripts\python.exe -m pytest tests/ -o addopts="" -p no:cacheprovider -q
```

- [ ] 改动文件 ruff 零错误
- [ ] pytest 全部通过，**原有 15 个用例无一变化**
- [ ] 未新增 `logging` 混用、未新增硬编码路径
- [ ] 新函数有完整类型标注
- [ ] 未提交密钥、PDF、数据库、模型权重