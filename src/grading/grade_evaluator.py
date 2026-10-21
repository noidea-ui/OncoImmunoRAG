"""GRADE evidence-grading module for OncoImmunoRAG.

Given the top-K literature chunks retrieved for a clinical question, ask an
LLM to extract the GRADE evidence-assessment fields into a structured
:class:`GRADEResult`, then run a small deterministic rule-engine over the
extracted fields to compute :attr:`GRADEResult.overall_grade`.

The module is intentionally self-contained — :func:`grade_extraction` is a
pure function over a list of strings, so callers can feed it text retrieved
from any source (Chroma store, PubMed dump, pasted excerpts, …). The CLI
self-test wires it to the existing OncoImmunoRAG vector store.

Typical usage::

    poetry run python -m grading.grade_evaluator
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from pydantic import BaseModel, Field

from utils.config_loader import AppConfig, load_config, resolve_config_path
from utils.prompts import (
    discussion_section_rule,
    statistical_significance_rules,
)
from utils.trial_registry import (
    canonical_sample_size,
    resolve_study_design,
)

# Heavy third-party imports are deferred to the functions that need them so
# the pure rule engine (``GRADEResult`` / ``apply_downgrade_rules``) can be
# imported in offline / test contexts where ``chromadb`` / ``llama_index``
# aren't installed.
if TYPE_CHECKING:  # pragma: no cover — only for type checkers
    from openai import OpenAI

# --- enums (mirrored exactly in the GRADEResult model) ----------------------

RiskLevel = Literal["高", "中", "低"]
Confounding = Literal["是", "否", "未报告"]
# ``单臂试验`` (single-arm trial) is a real oncology study design that
# doesn't fit any of the GRADE canonical buckets; we treat it as
# observational-equivalent in the downgrade rule engine.
StudyDesign = Literal["RCT", "观察性", "系统评价", "Meta分析", "单臂试验"]
OverallGrade = Literal["高", "中", "低", "极低"]

# Designs the LLM sometimes returns that should collapse to the closest
# canonical bucket before rule-engine evaluation. Order matters for the
# substring-match fallback below: single-arm markers are listed *before* the
# generic "open-label" / dose-escalation tokens so they win when both appear.
_DESIGN_ALIASES: dict[str, StudyDesign] = {
    "rct": "RCT",
    "随机对照试验": "RCT",
    "随机对照": "RCT",
    # Randomised-control evidence markers (from the spec: only these +
    # a comparator arm qualify for "RCT").
    "randomized controlled": "RCT",
    "randomised controlled": "RCT",
    "randomized": "RCT",
    "randomised": "RCT",
    "double-blind": "RCT",
    "placebo-controlled": "RCT",
    "phase 3": "RCT",
    "phase iii": "RCT",
    "随机双盲": "RCT",
    "随机安慰剂对照": "RCT",
    "观察性": "观察性",
    "观察性研究": "观察性",
    "队列研究": "观察性",
    "病例对照": "观察性",
    "横断面研究": "观察性",
    "系统评价": "系统评价",
    "系统综述": "系统评价",
    "meta分析": "Meta分析",
    "meta-analysis": "Meta分析",
    # Single-arm canonical aliases (Chinese + English / mixed).
    "单臂试验": "单臂试验",
    "单臂研究": "单臂试验",
    "单组研究": "单臂试验",
    "single-arm": "单臂试验",
    "single arm": "单臂试验",
    "single-arm trial": "单臂试验",
    "single-arm study": "单臂试验",
    "monotherapy": "单臂试验",
    "single-arm, open-label": "单臂试验",
    # Phase I/II design markers with no internal control.
    "phase 1": "单臂试验",
    "phase i": "单臂试验",
    "phase 2": "单臂试验",
    "phase ii": "单臂试验",
    "phase i/ii": "单臂试验",
    "phase ib/ii": "单臂试验",
    "phase 1/2": "单臂试验",
    "phase 1b/2": "单臂试验",
    # Open-label without a randomised comparator + dose-escalation vocab.
    "open-label, multicentre": "单臂试验",
    "open-label multicentre": "单臂试验",
    "open-label": "单臂试验",
    "dose-escalation": "单臂试验",
    "dose-expansion": "单臂试验",
    "dose-finding": "单臂试验",
}

# Tokens that ``_normalize_design`` checks *before* the RCT / observational
# aliases, so a substring containing any of them collapses to "单臂试验"
# regardless of other conflicting cues. Mirrors the single-arm branch of
# the prompt's strong-constraint block.
_SINGLE_ARM_DESIGN_TOKENS: tuple[str, ...] = (
    "single-arm",
    "single arm",
    "single-arm trial",
    "single-arm study",
    "monotherapy",
    "dose-escalation",
    "dose-expansion",
    "dose-finding",
    "phase i/ii",
    "phase ib/ii",
    "phase 1/2",
    "phase 1b/2",
    "phase 1",
    "phase 2",
    "phase i",
    "phase ii",
    "open-label, multicentre",
    "open-label multicentre",
    "open-label",
    "单臂试验",
    "单臂研究",
    "单组研究",
)


def _normalize_design(value: object) -> StudyDesign:
    """Map LLM output to one of the canonical study designs.

    Defaults to ``"观察性"`` when the LLM emits something unrecognised —
    conservative fallback that pushes the rule engine towards a lower grade
    rather than assuming the strongest design.

    The substring fallback iterates single-arm markers *first* so that
    text containing "single-arm" / "phase 1" / "phase 2" never collapses
    up to "RCT" even when it also contains the word "randomized" or
    "phase 3". The prompt's strong constraints already block that case at
    extraction time; this order makes the post-fix defence-in-depth.
    """
    s = str(value or "").strip().lower()
    if s in _DESIGN_ALIASES:
        return _DESIGN_ALIASES[s]
    # Substring match — single-arm markers first (defence-in-depth).
    for k in _SINGLE_ARM_DESIGN_TOKENS:
        if k in s:
            return "单臂试验"
    for k, v in _DESIGN_ALIASES.items():
        if k in s:
            return v
    return "观察性"

# Canonical ordering used by the downgrade rule engine.
_GRADE_ORDER: tuple[OverallGrade, ...] = ("高", "中", "低", "极低")

# When the LLM says "未报告" / "未知" for one of the 5 risk domains (which
# only allow 高/中/低), we fall back to "中" — unknown is best modelled as
# an uncertain middle grade rather than the optimistic default.
_RISK_FALLBACKS = {
    "高": "高",
    "中": "中",
    "低": "低",
    "未提及": "中",
    "未知": "中",
    "未报告": "中",
    "不适用": "中",
    "n/a": "中",
    "na": "中",
    "": "中",
}


def _normalize_risk(value: object) -> str:
    """Map LLM output to one of {"高", "中", "低"}, defaulting to "中"."""
    s = str(value or "").strip().lower()
    # The LLM occasionally writes "中风险" / "低风险" etc.
    for k in _RISK_FALLBACKS:
        if s == k or s == k.lower():
            return _RISK_FALLBACKS[k]
    # Try prefix match for things like "中风险" / "高 (high)".
    for k in ("高", "中", "低"):
        if k in str(value or ""):
            return k
    return "中"

# --- pydantic model ---------------------------------------------------------


class GRADEResult(BaseModel):
    """Structured GRADE evidence profile for one body of literature."""

    risk_of_bias: RiskLevel = Field(description="偏倚风险")
    inconsistency: RiskLevel = Field(description="不一致性")
    indirectness: RiskLevel = Field(description="间接性")
    imprecision: RiskLevel = Field(description="不精确性")
    publication_bias: RiskLevel = Field(description="发表偏倚")
    confounding_by_subsequent_therapy: Confounding = Field(
        description="后续治疗混杂"
    )
    study_design: StudyDesign = Field(description="研究设计")
    sample_size: int = Field(description="样本量")
    effect_size: str = Field(description='效应量，如 HR=0.47, 95% CI 0.38-0.58')
    overall_grade: OverallGrade = Field(description="最终证据等级")
    grade_reasoning: str = Field(description="分级依据")
    # ---- composite-score fields (added by 2026-09 spec change) ---------
    # Defaults keep backward compatibility: old JSON payloads that lack the
    # new fields still validate, falling back to score 0.0 + a placeholder
    # breakdown. ``_compute_composite_score_breakdown`` mirrors the
    # algorithm in the system prompt so reviewers can audit whether the
    # LLM's number matches an independent recomputation.
    composite_score: float = Field(
        default=0.0,
        description="综合评分(0.0-10.0,精确到 0.1)",
    )
    score_breakdown: str = Field(
        default="未计算",
        description="综合评分拆解,逐项列出加减分数值",
    )

    model_config = {"extra": "forbid"}


# --- config + LLM factory (shared pattern with retrieval.query_engine) ------


def _vector_db_path() -> Path:
    return resolve_config_path().parent / "data" / "vector_db"


def _require_api_key(cfg: AppConfig) -> str:
    key = cfg.effective_openai_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set and config.yaml has no openai_api_key. "
            "Set the OPENAI_API_KEY environment variable before running this "
            "module, e.g.:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  poetry run python -m grading.grade_evaluator"
        )
    return key


def _build_llm(cfg: AppConfig) -> "OpenAI":
    """Raw OpenAI SDK client (not llama-index wrapper, see query_engine)."""
    from openai import OpenAI  # lazy — keeps the offline regrade path importable

    api_key = _require_api_key(cfg)
    base_url = cfg.extras.get("llm_base_url") or os.environ.get(
        "LLM_BASE_URL", "https://api.openai.com/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)


def _llm_model_name(cfg: AppConfig) -> str:
    return cfg.extras.get("llm_model") or os.environ.get("LLM_MODEL", "gpt-4o-mini")


def _build_embed_model(cfg):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=cfg.embedding_model)


def _open_index(cfg: AppConfig):
    import chromadb
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    db_path = _vector_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {db_path}. Run "
            "`poetry run python -m ingestion.vector_store` first."
        )
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection("oncoimmuno_rag")
    except Exception as exc:
        raise RuntimeError(
            f"Collection 'oncoimmuno_rag' missing from {db_path}. "
            "Re-run the ingestion step."
        ) from exc
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=_build_embed_model(cfg)
    )


# --- LLM extraction ---------------------------------------------------------


# Strong-constraint prompt block that prevents the LLM from mislabelling a
# single-arm / early-phase oncology trial as an RCT. Earlier benchmark
# showed the model defaulting to "RCT" whenever the abstract mentioned
# "open-label" or "multicentre" — this block is the counterweight. It is
# appended to the system prompt so the model sees it on every call.
_STUDY_DESIGN_RULES = """\
【研究设计分类强约束（关键，违反将被判错）】
1. 默认归类为 "观察性"。
2. 当且仅当文本中**同时**满足以下全部要素时，才允许判定为 "RCT"：
   - 含 "randomized" / "randomisation" / "randomly assigned" / "随机"
   - 含 "placebo-controlled" / "double-blind" / "active comparator" / 明确对照组
   - 若以期数描述，须明确为 phase 3
3. 若文本中**出现任一**下列标记且【不存在对照组】，严禁输出 "RCT" 或
   "系统评价"，必须输出 "单臂试验"：
   - "single-arm" / "single arm" / "single group" / "monotherapy"
   - "phase 1" / "phase i" / "phase 2" / "phase ii" / "phase 1/2" / "phase i/ii"
   - "open-label, multicentre" 或 仅 "open-label" 且为 I/II 期
   - "dose-escalation" / "dose-expansion" / "dose-finding"
   - 多个剂量队列（dose-level cohorts）而只有一个给药组
4. "系统评价" 仅当文本明确指出 systematic review / meta-analysis / PRISMA
   时才可判定；其余一律不输出 "系统评价"。
5. study_design 只能输出以下五个值之一："RCT" | "观察性" | "系统评价"
   | "Meta分析" | "单臂试验"；不要输出其他取值。
"""


_CROSS_PAPER_PROMPT_GUARD = """\
【跨文献隔离规则（关键：严禁跨文献提取）】
1. 当上下文片段中包含多个不同文件名的文献时，必须先把片段按文件名归类。
2. 提取 sample_size / effect_size 时，**只能**从与目标文献（见上方明示的
   canonical_file_name）**同名**的片段中提取。其它片段里出现的 HR / N /
   ORR 仅作为参考上下文，**禁止**复制到目标字段。
3. 如果目标文献的片段里没有任何 N=… / patients / 例数 表达，且其上下文
   未提供足够信息，sample_size 必须输出 0（留待人工或后处理兜底），
   **严禁**用别的试验的数字填充。
4. effect_size 同理：只允许使用目标文献片段中报告的 HR / ORR / pCR / OS
   等指标；找不到时填 "未提及"。
5. 在 grade_reasoning 中明确写出："本字段数据来源于 <canonical_file_name>"
   以便审计。
"""


# Composite-score rules — appended at the very end of the system prompt so
# the model sees the concrete scoring algorithm right before producing JSON.
# The LLM is asked to *compute* the score per these rules; Python mirrors
# the algorithm in :func:`_compute_composite_score_breakdown` so reviewers
# can diff the LLM-emitted score against an independent recomputation.
#
# Numerical anchors are pinned so Python's reimplementation can match the
# LLM one-for-one:
#   * 5 域 维度 → 低=0, 中=-1.0, 高=-1.5 (严重), 极严重触发仅 A 规则
#     (B 规则 "OS-strict-missed + imprecision=高 → -3.0" 已删除 2026-09;
#      OS 阴性现在仅通过 imprecision=高(-1.5) + P-bonus -0.5 体现)
#   * 升级分仅 单臂/观察性 适用 (RCT 一律 0)
#   * 效应量按 HR/RR 量化：极强<0.25 → +3.0, 大<0.5 → +1.5, 中等<0.8 → +0.5
#   * HR ≥ 1 或 无 HR → 升级 0 (unknown)
_COMPOSITE_SCORE_RULES = """\
【综合评分规则（含数值化阈值，便于重现计算）】

# 1) 起始分（按研究设计）
- RCT / 系统评价 / Meta分析 = 10.0
- 单臂试验 / 观察性       = 4.0
- 未知                    = 5.0（兜底中位）

# 2) 降级分（每个 5 域维度按其等级扣分；同一维度不重复扣）
| 维度等级 | 扣分 |
|:---:|:---:|
| 低   | 0    |
| 中   | -1.0 |
| 高   | -1.5（=严重）|
| 极严重 | -3.0（仅 (A) 触发）|

触发「极严重」档（不依赖 5 域的字面取值，由客观条件判定）：
  (A) ``confounding_by_subsequent_therapy=是`` 且 该研究为 RCT → 视为极严重 -3.0
      （即 base RCT=10 一次最多掉到 7；与 confound=是 相关的常规 -1.5 不再叠加）

# 3) 升级分（仅 单臂/观察性；RCT 一律 0）
| HR / RR 区间 | 效应量 | 加分 |
|:---:|:---:|:---:|
| HR ≥ 1.0 或 无可解析 HR | unknown / harmful | 0 |
| 0.8 ≤ HR < 1.0 | 小 | 0 |
| 0.5 ≤ HR < 0.8 | 中等 | +0.5 |
| 0.25 ≤ HR < 0.5 | 大 | +1.5 |
| HR < 0.25 | 极大/极强 | +3.0 |

（HR > 1 表示对干预不利方向，升级分不可触发）

# 4) P值附加分（一次只命中一档）
- P < 0.001 且 效应量大或极大 → +1.0
- P < 0.05  且 效应量中等      → +0.5
- P ≥ 0.05                    → 0
- OS 预设严格 alpha 未达        → -0.5（与以上三档互斥，命中时取此项）

# 5) 综合计算公式
  综合分数 = 起始分 - 总降级分 + 总升级分 + P值附加分
  （不得为负，最低 0.0；不得高于 10.0）

# 6) 封顶规则
  最终分数 = min(综合分数, 起始分 - 总降级分 + 2.0)
  （防止升级/P值附件 把分数推得过高）

# 7) 等级映射
- 8.5 ≤ 分数 ≤ 10.0 → 高
- 5.0 ≤ 分数 <  8.5 → 中
- 2.6 ≤ 分数 <  5.0 → 低
- 0.0 ≤ 分数 <  2.6 → 极低

# 8) 输出格式（JSON）
- ``composite_score``: 浮点数，保留 1 位小数（0.0 - 10.0）
- ``score_breakdown``: 字符串，必须按上述 6 步逐项列出，例如：
  ``起始=10.0(RCT); 降级:risk_of_bias=中→-1.0,imprecision=中→-1.0;
   升级:N/A(RCT); P值附加:P<0.001且效应量大,+1.0;
   综合=8.5; 封顶:8.5 ≤ max(8.0+2.0=10.0)→不封; 最终=8.5``
"""


# The LLM is asked to produce every GRADE field *except* ``overall_grade``
# (computed deterministically by :func:`apply_downgrade_rules`) so the model
# cannot short-circuit the rule engine.
_EXTRACTION_SYSTEM = (
    "你是医学循证专家，熟悉 GRADE 证据分级框架。"
    "请严格基于给定的文献片段内容，提取 GRADE 评估字段。"
    "若文献未提及某项，请按字段允许的占位值填写并说明。"
    "必须只返回合法 JSON，不要包含任何额外文字、Markdown 代码块或解释。\n\n"
    + statistical_significance_rules()
    + "\n\n"
    + discussion_section_rule()
    + "\n\n"
    + _STUDY_DESIGN_RULES
    + "\n\n"
    + _CROSS_PAPER_PROMPT_GUARD
    + "\n\n"
    + _COMPOSITE_SCORE_RULES
)

_EXTRACTION_INSTRUCTION = """\
返回以下字段（注意：不要填写 overall_grade，由规则引擎计算）：

- risk_of_bias        : "高" | "中" | "低"
- inconsistency       : "高" | "中" | "低"
- indirectness        : "高" | "中" | "低"
- imprecision         : "高" | "中" | "低"
- publication_bias    : "高" | "中" | "低"
- confounding_by_subsequent_therapy : "是" | "否" | "未报告"
- study_design        : "RCT" | "观察性" | "系统评价" | "Meta分析" | "单臂试验"
- sample_size         : 整数（若未提及填 0；不要写成字符串、不要带 "N=" 前缀）
- effect_size         : 字符串，如 "HR=0.47, 95% CI 0.38-0.58"；未提及填 "未提及"
- composite_score     : 浮点数(0.0-10.0)，按【综合评分规则】算出综合分数(精确到 0.1)
- score_breakdown     : 字符串，逐项列出起始分 / 降级分 / 升级分 / P 值附加分 / 封顶后分数；
                        例如 "起始=10.0; 降级:risk_of_bias=高→-1.5,imprecision=中→0;
                        升级:N/A(RCT); P值附加:P<0.001,+1.0; 封顶后=9.5"
- grade_reasoning     : 简要中文说明提取依据与不确定性

文献片段（共 {n_chunks} 段）\
"""


def _format_chunks(chunks: Sequence[str]) -> str:
    """Join chunks with stable separators so the LLM can reference positions."""
    body = "\n\n".join(
        f"[片段 {i + 1}]\n{c.strip()}" for i, c in enumerate(chunks) if c and c.strip()
    )
    return body or "(无内容)"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_to_dict(raw: str) -> dict:
    """Tolerate LLM responses that wrap JSON in prose / code fences."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    match = _JSON_OBJECT_RE.search(text)
    if match:
        text = match.group(0)
    return json.loads(text)


# Patterns tried (in order) by ``_extract_sample_size_from_chunks`` when the
# LLM extractor returns 0 / blank. These mirror how trial papers describe
# their enrolled cohort — "N=21", "n = 33 patients", "21例", etc.
_SAMPLE_SIZE_CHUNK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[Nn]\s*=\s*(\d[\d,]*)"),
    re.compile(r"\bn\s*\(?\s*(\d[\d,]*)\s*\)?\s*(?:patients|subjects|participants|cases)", re.I),
    # "N evaluable/total/enrolled subjects" — covers "33 evaluable subjects".
    re.compile(
        r"\b(\d[\d,]*)\s+(?:evaluable|total|enrolled|included)\s+"
        r"(?:patients|subjects|participants|cases)\b",
        re.I,
    ),
    re.compile(r"\b(\d[\d,]*)\s+(?:patients|subjects|participants)\b", re.I),
    re.compile(r"enrolled\s+(\d[\d,]*)\s+(?:patients|subjects|participants)", re.I),
    re.compile(r"(\d[\d,]*)\s*例(?:患者|受试者)?", ),
)


def _extract_sample_size_from_chunks(chunks: Sequence[str]) -> int:
    """Return the largest ``n=``-style integer found in ``chunks`` (0 if none).

    Used as a second-pass fallback when the LLM extractor emits ``0`` (or a
    non-numeric string) for ``sample_size`` — without this the rule engine's
    ``sample_size < 300 → −1`` penalty silently no-ops, which is what caused
    the recent McGregor-DAD / ABACUS under-call.

    A match is only counted when the integer is ≥ 5 — that filters out
    artefact hits like "n=1 cohort" while keeping the typical single-arm /
    phase 2 enrolled-N range (usually 10–250).
    """
    best = 0
    for chunk in chunks or ():
        if not chunk:
            continue
        for pat in _SAMPLE_SIZE_CHUNK_PATTERNS:
            for m in pat.finditer(chunk):
                try:
                    val = int(m.group(1).replace(",", ""))
                except (ValueError, AttributeError, IndexError):
                    continue
                if val >= 5:
                    best = max(best, val)
    return best


def _coerce_sample_size(value: object) -> int:
    """Robustly coerce an LLM-emitted sample_size to ``int``.

    Accepts: int, float, or a string in any common phrasing:

    * ``"21"``                  → 21
    * ``"N=21"``                → 21
    * ``"N = 21 patients"``     → 21
    * ``"250 patients"``        → 250
    * ``"approximately 250"``   → 250  (first integer wins)
    * ``""`` / ``"未提及"``      → 0
    * anything unparseable      → 0
    """
    if isinstance(value, bool):  # bool is a subclass of int — exclude early.
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if value is None:
        return 0
    s = str(value).strip()
    if not s:
        return 0
    # Skip non-numeric placeholders the LLM likes to insert.
    lower = s.lower()
    if lower in {"n/a", "na", "未提及", "未报告", "未知", "none", "null"}:
        return 0
    # First integer (with optional thousands commas) in the string wins.
    m = re.search(r"\d[\d,]*", s)
    if not m:
        return 0
    try:
        return max(0, int(m.group(0).replace(",", "")))
    except ValueError:
        return 0


# --- downgrade rule engine --------------------------------------------------


def apply_downgrade_rules(result: GRADEResult) -> GRADEResult:
    """Apply deterministic downgrades and overwrite ``overall_grade``.

    Rules (each fires at most once, in the order listed):

    0. **Baseline assignment** (GRADE convention, not a "downgrade"):

       * ``study_design ∈ {"RCT", "系统评价", "Meta分析"}`` → 基线 "高"
       * ``study_design ∈ {"观察性", "单臂试验"}``           → 基线 "低"

       The "低" baseline already encodes the GRADE convention that
       observational evidence starts two levels below RCT. The rules
       below stack on top of that baseline, so observational designs
       can still be downgraded further (e.g. sample-size < 300).

    1. **Non-RCT penalty (only for designs that started at "高")**:
       ``study_design ∉ {"RCT"}`` *and* baseline is "高" → −1.
       Single-arm and observational designs **must NOT** take this hit
       again — they already paid for it in the baseline.

    2. ``sample_size < 300``                                  → −1
    3. any of ``risk_of_bias/inconsistency/indirectness/
       imprecision/publication_bias`` == "高"               → −1 per domain
    4. ``confounding_by_subsequent_therapy == "是"``          → −1

    System reviews and meta-analyses are treated as RCT-equivalent
    (start at "高") unless their constituent primary studies are flagged
    by rules 2–4.
    """
    # GRADE base by study design. Single-arm trials don't have an
    # internal control group, so they share the observational base.
    _OBSERVATIONAL_DESIGNS = {"观察性", "单臂试验"}
    _RCT_LIKE = {"RCT", "系统评价", "Meta分析"}

    if result.study_design in _OBSERVATIONAL_DESIGNS:
        idx = _GRADE_ORDER.index("低")
        baseline_note = (
            f"基线等级 by study_design={result.study_design}: 低 "
            f"(GRADE: 观察性设计默认 -2,不再叠加非 RCT 惩罚)"
        )
    elif result.study_design in _RCT_LIKE:
        idx = _GRADE_ORDER.index("高")
        baseline_note = f"基线等级 by study_design={result.study_design}: 高"
    else:
        # Unknown design → start mid ("中") so unknown does not silently
        # inherit the strongest default.
        idx = _GRADE_ORDER.index("中")
        baseline_note = (
            f"基线等级 by study_design={result.study_design}: 中 "
            f"(未知设计,默认中位)"
        )

    notes: list[str] = [baseline_note]

    # Rule 1: non-RCT penalty ONLY applies when the baseline is "高".
    # Observational / single-arm designs already pay for their non-RCT
    # status via the baseline ("低"); applying rule 1 on top would
    # double-count and over-penalise (the previous bug).
    if result.study_design != "RCT" and _GRADE_ORDER[idx] == "高":
        idx += 1
        notes.append(f"非 RCT (study_design={result.study_design}) → −1")
    elif result.study_design != "RCT":
        notes.append(
            f"非 RCT (study_design={result.study_design}) → 跳过 "
            f"(已在基线扣过)"
        )

    if result.sample_size and result.sample_size < 300:
        idx += 1
        notes.append(f"样本量 {result.sample_size} < 300 → −1")
    elif not result.sample_size:
        # Trace the gap rather than silently skip — if this fires the
        # caller (or the chunk-fallback path) needs attention.
        notes.append(
            "样本量未知 (sample_size=0) → 跳过 < 300 降级 "
            "(无法判定 imprecision)"
        )

    for domain in (
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
    ):
        if getattr(result, domain) == "高":
            idx += 1
            notes.append(f"{domain}=高 → −1")

    if result.confounding_by_subsequent_therapy == "是":
        idx += 1
        notes.append("confounding_by_subsequent_therapy=是 → −1")

    idx = min(idx, len(_GRADE_ORDER) - 1)
    new_grade = _GRADE_ORDER[idx]
    notes.append(f"最终等级 = {new_grade}")

    new_reasoning = (
        (result.grade_reasoning or "").strip()
        + "\n\n[降级规则引擎]\n  - "
        + "\n  - ".join(notes)
    ).strip()

    return result.model_copy(
        update={"overall_grade": new_grade, "grade_reasoning": new_reasoning}
    )


# --- composite-score engine (deterministic Python mirror of system-prompt rules) ---

# Match an HR / hazard-ratio / 风险比 token, capturing the leading scalar. The
# optional "=" / "[" / 中文前缀 are tolerated so LLM-emitted effect_size
# strings ("HR=0.78, 95% CI …", "hazard ratio 0·69", "风险比 0.78 …") all
# extract cleanly. ``_HR_RE`` is purposely the *first* HR-like number in
# the string — for KEYNOTE-361 that's PFS=0.78 (LLM choice); for SWOG-1605
# (CR率-only) it returns None.
_HR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:HR|hr|hazard\s*ratio|hazards?\s*ratio)\s*[:=]?\s*\[?\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"风险比\s*[:=]?\s*\[?\s*(\d+(?:\.\d+)?)"),  # 风险比 0.78
    re.compile(r"\b(?:HR|hazard\s*ratio)\s+(\d+(?:\.\d+)?)", re.I),
)

# Match a one-sided P-value, tolerating "p=0.0033", "P<0.001", "P值<0.001"
# (Chinese "值" trailing the letter), "p < 0.0001", etc.
# Note: leading \b is optional — Chinese chars adjacent to "P" (e.g. "P值")
# don't form a regex word boundary, so the optional prefix lets us match
# both ASCII and CJK-suffixed phrasings.
_PVALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lt", re.compile(r"(?:\b|[一-鿿])[Pp]\s*[值v]?\s*<\s*0?\.(\d+)", re.I)),
    ("le", re.compile(r"(?:\b|[一-鿿])[Pp]\s*[值v]?\s*<=\s*(\d+(?:\.\d+)?)", re.I)),
    ("eq", re.compile(r"(?:\b|[一-鿿])[Pp]\s*[值v]?\s*=\s*(\d+(?:\.\d+)?)", re.I)),
    ("eq", re.compile(r"(?:\b|[一-鿿])[Pp]\s*[值v]?\s*[=:]\s*0?\.(\d+)", re.I)),
    ("gt", re.compile(r"(?:\b|[一-鿿])[Pp]\s*[值v]?\s*>\s*(\d+(?:\.\d+)?)", re.I)),
)


def _extract_hr(text: str) -> float | None:
    """Parse the leading HR (or hazard ratio) from ``text``.

    Returns ``None`` when no HR-shaped token is present (e.g. ``"CR率 27%
    (95% CI 17-38%)"`` has no HR). Used by :func:`_compute_composite_score`.
    """
    if not text:
        return None
    for pat in _HR_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            v = float(m.group(1))
        except (ValueError, IndexError):
            continue
        # Plausible-HR guard: HR should be in (0, 5]. Refuse HR=10 / 20
        # which the LLM occasionally emits from "p<0.001" through greedy
        # numeric matching.
        if 0.0 < v <= 5.0:
            return v
    return None


def _extract_pvalue(text: str) -> float | None:
    """Parse the leading P-value from ``text``.

    Returns ``None`` when no P-shaped token is present. ``"<0.001"`` and
    ``"=0.04"`` both resolve to floats (0.001 / 0.04).
    """
    if not text:
        return None
    for kind, pat in _PVALUE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        raw = m.group(1)
        try:
            if kind == "lt":
                v = float("0." + raw)
            else:
                v = float(raw)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
    return None


# Wrapper used by the RCT floor rule + the composite scorer: scans both
# ``effect_size`` (preferred) and ``grade_reasoning`` (fallback) so a missing
# P in the LLM's effect_size still gets caught from reasoning text like
# "P<0.001 达到统计学显著".
def _extract_pvalue_composite(
    effect_size: str | None, reasoning: str | None = None
) -> float | None:
    """Return the leading P across effect_size + reasoning, or None."""
    p = _extract_pvalue(effect_size)
    if p is not None:
        return p
    return _extract_pvalue(reasoning)


def _classify_effect(hr: float | None) -> tuple[str, float]:
    """Map a parsed HR to (bucket, upgrade_points).

    Bucket mapping (mirrors system-prompt rules):
      * HR ≥ 1.0 or HR is None → "unknown_or_harmful" → 0
      * 0.8  ≤ HR < 1.0  → "small"          → 0
      * 0.5  ≤ HR < 0.8  → "medium"         → +0.5
      * 0.25 ≤ HR < 0.5  → "large"          → +1.5
      * HR < 0.25        → "extremely_large" → +3.0
    """
    if hr is None or hr >= 1.0:
        return "unknown_or_harmful", 0.0
    if hr < 0.25:
        return "extremely_large", 3.0
    if hr < 0.5:
        return "large", 1.5
    if hr < 0.8:
        return "medium", 0.5
    return "small", 0.0


def _detect_strict_threshold_failure(reasoning: str | None) -> bool:
    """Detect "预设严格阈值未达" markers in the LLM's ``grade_reasoning``.

    Different LLMs word this very differently across runs ("未达预设严格阈值",
    "未达到预设的严格界值", "未跨越统计学显著性界值", etc.). We layer
    matchers so the detector stays robust to phrasing drift:

    1. **Phrase dictionary** — common Chinese phrasings observed in
       verification runs.
    2. **Regex on (未达|未跨越) … (阈值|界值|α|alpha)** — catches variants
       the dictionary misses.
    3. **Numeric p>α signature** — when the reasoning text shows a p-value
       inequality like "p=0.0033 > 0.0019", the threshold was clearly
       *not* crossed.
    """
    if not reasoning:
        return False
    text = reasoning.replace(" ", "").replace("　", "")

    markers = (
        "未达预设严格阈值",
        "未达预设的严格阈值",
        "未达到预设的严格界值",
        "未达到预设的严格阈值",
        "未达到预设的严格显著性界值",
        "未达到预设的严格统计学显著性界值",
        "未完全达到预设的严格显著性界值",
        "未跨越预设严格阈值",
        "未跨越预设的严格阈值",
        "未跨越预设界值",
        "未达严格阈值",
        "未过预设严格阈值",
        "OS阴性",
        "OS未达",
        "未达到预设alpha",
        "未达到预设α",
        "未达到预设的alpha",
        "未达预设α",
        "未跨越预设α",
    )
    if any(m.replace(" ", "") in text for m in markers):
        return True
    # Catch "未达到...统计...显著性界值" / "未跨越...界值" families.
    secondary = (
        "未达到预设的统计学显著性界值",
        "未达到统计学显著性界值",
        "未跨越统计学显著性界值",
        "未达到预设界值",
        "未跨越预设界值",
        "统计学显著性界值未达",
        "未达统计学界值",
    )
    if any(s in text for s in secondary):
        return True
    # Regex fallback: 未达|未跨越 + 一定距离内的 阈值|界值|α|alpha
    loose = re.compile(
        r"未[达跨越]?(?:到|过)?[^\n]{0,30}?(?:阈值|界值|α|alpha|Alpha|ALPHA)",
    )
    if loose.search(text):
        return True
    # Numeric signature: a p>α inequality anywhere in reasoning.
    p_vs_alpha = re.compile(
        r"[Pp]\s*[=<>]\s*0?\.?\d+\s*[<>≥≤]\s*0?\.?\d+"
    )
    if p_vs_alpha.search(text):
        return True
    return False


def _compute_composite_score_breakdown(
    result: GRADEResult,
    *,
    strict_threshold_not_met: bool | None = None,
) -> tuple[float, str, dict]:
    """Compute ``composite_score`` deterministically per the rules in :data:`_COMPOSITE_SCORE_RULES`.

    Implements rules 1-7 strictly — same algorithm the LLM is asked to follow
    in its system prompt. Used by :func:`grade_extraction` to *override* the
    LLM-emitted ``composite_score`` so the final value is reproducible across
    re-runs (the LLM tends to drift on borderline inputs like SWOG-S1605's
    "CR率 only, no HR").

    Args:
        result: A :class:`GRADEResult` (post-pydantic-validation, before
            ``apply_downgrade_rules``). All fields are read but not mutated.
        strict_threshold_not_met: When non-None, force the "OS preset-strict
            alpha missed" arm of rule 4. When ``None``, infer from the LLM's
            :attr:`GRADEResult.grade_reasoning` via
            :func:`_detect_strict_threshold_failure`.

    Returns:
        3-tuple ``(score, breakdown_str, debug_dict)``:

        * ``score`` — float in [0.0, 10.0], rounded to 1 decimal.
        * ``breakdown_str`` — human-readable summary matching the format the
          system prompt requests from the LLM.
        * ``debug_dict`` — structured fields (``"baseline"``,
          ``"downgrade_total"``, ``"upgrade_total"``, ``"p_bonus"``,
          ``"raw_score"``, ``"cap"``, ``"cap_applied"``, ``"effect_bucket"``,
          ``"hr"``, ``"p"``) for tests / inspection.
    """
    # ----- 1) baseline by study design -------------------------------------
    if result.study_design in {"RCT", "系统评价", "Meta分析"}:
        baseline = 10.0
        base_label = "RCT-family"
    elif result.study_design in {"单臂试验", "观察性"}:
        baseline = 4.0
        base_label = "single-arm/observational"
    else:
        baseline = 5.0
        base_label = "unknown"

    # ----- 2) downgrade -----------------------------------------------------
    downgrade_total = 0.0
    downgrade_notes: list[str] = []

    # 5 risk domains
    domain_deltas = {"低": 0.0, "中": -1.0, "高": -1.5}
    for domain in (
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
    ):
        v = getattr(result, domain)
        delta = domain_deltas.get(v)
        if delta is None:
            # Defensive: if the LLM emitted an unrecognised token, fall back
            # to "中" so we don't silently inherit 0.
            delta = -1.0
            v = f"{v}(→中)"
        if delta < 0:
            downgrade_total += delta
            downgrade_notes.append(f"{domain}={v}→{delta:+.1f}")

    # Trigger (A): confounding_by_subsequent_therapy=是 in an RCT-family
    # design → 极严重 -3.0 (REPLACES, not stacks with, the常规 -1.5 dedup).
    # For non-RCT designs the regular confounding penalty doesn't apply in
    # this rule set (it was already captured in the prior 5-domain scoring).
    if (
        result.confounding_by_subsequent_therapy == "是"
        and result.study_design in {"RCT", "系统评价", "Meta分析"}
        and not any(n.startswith("极严重") for n in downgrade_notes)
    ):
        # Subtract an *extra* -1.5 on top of the 5-domain dedup so the net
        # RCT-confounding penalty lands at -3.0 from baseline.
        downgrade_total += -1.5
        downgrade_notes.append("极严重[A:confounding=是 & RCT]→-1.5")

    # ---- Trigger (B) REMOVED per 2026-09 spec ----
    # The previous "OS-strict-missed + imprecision=高 → extra -1.5" rule has
    # been removed: the OS-negative penalty is now expressed *only* through
    # (i) imprecision being set to "高" by the RCT-floor exemption (which
    # deducts -1.5 on its own), and (ii) the -0.5 P-bonus arm when strict
    # threshold is missed. Keeping an explicit stacking penalty would
    # double-count the same OS-nega evidence.

    # ----- 3) upgrade (single-arm / observational only) ---------------------
    hr = _extract_hr(result.effect_size)
    effect_bucket, effect_upgrade = _classify_effect(hr)

    if result.study_design in {"单臂试验", "观察性"}:
        upgrade_total = effect_upgrade
        upgrade_notes: list[str] = []
        if effect_upgrade > 0:
            upgrade_notes.append(f"效应量={effect_bucket}(HR={hr})→+{effect_upgrade:.1f}")
        else:
            upgrade_notes.append(f"效应量={effect_bucket}(HR={hr},未达升级阈值)")
    else:
        upgrade_total = 0.0
        upgrade_notes = ["N/A(RCT)"]

    # ----- 4) P-value bonus (mutually exclusive) ----------------------------
    # Strict-threshold failure detection is reused here — the *only* place
    # where it's still applied is the P-bonus arm (P-bonus -0.5 below).
    # The trigger (B) "extra -1.5" was removed in 2026-09.
    inferred_strict = _detect_strict_threshold_failure(result.grade_reasoning)
    p = _extract_pvalue_composite(result.effect_size, result.grade_reasoning)
    p_bonus = 0.0
    p_notes: list[str] = []

    if inferred_strict:
        # OS preset-strict-alpha missed → -0.5 (overrides any positive arm).
        p_bonus = -0.5
        p_notes.append("OS预设严格阈值未达→-0.5")
    elif p is not None and p < 0.001 and effect_bucket in ("large", "extremely_large"):
        p_bonus = 1.0
        label_cn = "大" if effect_bucket == "large" else "极大"
        p_notes.append(f"P<0.001且效应量{label_cn}→+1.0")
    elif p is not None and p < 0.05 and effect_bucket == "medium":
        p_bonus = 0.5
        p_notes.append("P<0.05且效应量中等→+0.5")
    else:
        if p is None:
            p_notes.append("无可解析P值→+0")
        elif p >= 0.05:
            p_notes.append(f"P={p}≥0.05→+0")
        else:
            p_notes.append(f"P={p}(效应量{effect_bucket}不满足附加档)→+0")

    # ----- 5) composite -----------------------------------------------------
    raw_score = baseline + downgrade_total + upgrade_total + p_bonus

    # ----- 6) cap -----------------------------------------------------------
    cap = baseline + downgrade_total + 2.0
    if raw_score > cap:
        final_score = cap
        cap_applied = True
    else:
        final_score = raw_score
        cap_applied = False

    # ----- clamp to [0, 10] -------------------------------------------------
    final_score = max(0.0, min(10.0, final_score))
    final_score = round(final_score, 1)

    # ----- 7) assemble breakdown --------------------------------------------
    parts: list[str] = [f"起始={baseline:.1f}({base_label})"]
    if downgrade_notes:
        parts.append("降级:" + ",".join(downgrade_notes))
    if upgrade_notes:
        parts.append("升级:" + ",".join(upgrade_notes))
    if p_notes:
        parts.append("P值附加:" + "; ".join(p_notes))
    if cap_applied:
        parts.append(f"封顶:综合={round(raw_score, 1)}→上限={round(cap, 1)}")
    else:
        parts.append(f"封顶:综合={round(raw_score, 1)}≤上限={round(cap, 1)}→不封")
    parts.append(f"最终={final_score:.1f}")

    debug = {
        "baseline": baseline,
        "downgrade_total": downgrade_total,
        "upgrade_total": upgrade_total,
        "p_bonus": p_bonus,
        "raw_score": raw_score,
        "cap": cap,
        "cap_applied": cap_applied,
        "effect_bucket": effect_bucket,
        "hr": hr,
        "p": p,
    }
    return final_score, "; ".join(parts), debug


def _grade_to_bucket(score: float) -> str:
    """Map a 0.0-10.0 composite score to the GRADE 4-level bucket.

    Rev. 2026-09 (tightened to 8.5/6.0/3.5/0):
      * ≥ 8.5 → 高
      * 6.0 - 8.4 → 中
      * 3.5 - 5.9 → 低
      * 0 - 3.4  → 极低

    Edge cases resolve to the lower band (e.g. exactly 8.5 → 高, exactly 6.0
    → 中). Note the combined effect with :func:`_apply_rct_floor`:
    qualifying positive RCTs (N≥300 AND P<0.05 AND no severe evidence)
    land at ≥ 9.5 after the floor's leniency cap, comfortably inside "高".
    """
    if score >= 8.5:
        return "高"
    if score >= 6.0:
        return "中"
    if score >= 3.5:
        return "低"
    return "极低"


# Pattern bank used by :func:`_has_severe_evidence` to detect when a
# qualifying RCT must NOT receive the floor-rule leniency. The list is
# intentionally narrow (only patterns we have *positive* evidence cause):
# dropout/LTFU >30%, data-integrity issues, protocol violations, fraud.
# Adding more markers is a one-line change.
_SEVERE_EVIDENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"脱落率[^\n]{0,40}?超过[^\n]{0,5}?30\s*%"),
    re.compile(r"脱落率[^\n]{0,40}?>\s*30\s*%"),
    re.compile(r"脱落率[^\n]{0,40}?>=\s*30\s*%"),
    re.compile(r"withdraw[^\n]{0,40}?>\s*30\s*%", re.I),
    re.compile(r"drop[\- ]?out[^\n]{0,40}?>\s*30\s*%", re.I),
    re.compile(r"drop[\- ]?out[^\n]{0,40}?\b[3-9]\d\s*%", re.I),
    re.compile(r"LTFU[^\n]{0,40}?>\s*30\s*%", re.I),
    re.compile(r"重大方案偏离"),
    re.compile(r"重大违背方案"),
    re.compile(r"重大违反gcp"),
    re.compile(r"fraud"),
    re.compile(r"数据完整性(受损|问题|存疑)"),
    re.compile(r"数据造假"),
    re.compile(r"主要终点定义发生变化"),
    re.compile(r"event\s*rate\s*drift"),
)


def _has_severe_evidence(reasoning: str | None) -> bool:
    """Return True when ``reasoning`` carries clear severe-evidence markers.

    Used by :func:`_apply_rct_floor` to decide whether to skip the leniency
    cap. Markers come from :data:`_SEVERE_EVIDENCE_PATTERNS`.
    """
    if not reasoning:
        return False
    text = reasoning.replace(" ", "").replace("　", "")
    return any(p.search(text) for p in _SEVERE_EVIDENCE_PATTERNS)


def _has_multiple_primary_endpoints(reasoning: str | None, effect_size: str | None) -> bool:
    """Return True when the trial appears to have multiple primary endpoints.

    Uses three signals:
      1. Explicit "dual primary" / "co-primary" markers (EN/ZH).
      2. Both PFS and OS appear in the combined reasoning+effect_size text.
      3. Multiple distinct alpha / significance thresholds are mentioned
         (typical of dual-primary trials that split α across endpoints).

    Drives the RCT-floor exemption clause: trials with multiple primary
    endpoints where one (typically OS) fails are not eligible for the
    RCT-floor leniency cap.
    """
    if not reasoning and not effect_size:
        return False
    haystack = (
        (reasoning or "") + " " + (effect_size or "")
    ).replace(" ", "").replace("　", "").lower()

    # (1) explicit "dual primary" markers
    if any(
        m in haystack
        for m in (
            "双主终点",
            "双主要终点",
            "dualprimary",
            "co-primary",
            "coprimaryendpoint",
        )
    ):
        return True

    # (2) both PFS and OS mentioned
    pfs_signals = ("pfs", "progression-free", "无进展生存", "dfs", "无复发生存", "disease-free", "recurrence-free")
    os_signals = ("overall survival", "总生存", "总生存期", " os ", " os,", " os。", " os(", "os =" , "os<", "os>", "os ")
    has_pfs = any(s.lower() in haystack for s in pfs_signals)
    has_os = any(s in haystack for s in os_signals)
    if has_pfs and has_os:
        return True

    # (3) multiple alpha / p-boundary values mentioned (typical of split-α
    #     design — e.g. KEYNOTE-361's PFS α=0.0019 + OS α=0.0142, both
    #     mentioned as "alpha=0.0019" / "alpha=0.0142" or with their
    #     values placed near "alpha"/"预设" / "边界" / "界值" keywords).
    # We capture any "<small decimal>" appearing in alpha-boundary contexts
    # and require ≥ 2 distinct values.
    alpha_anchored = re.findall(
        r"(?:alpha|α|边界|界值|阈值|预设|p\s*值边界|p\s*value\s*boundary|p\s*值界值)"
        r"[^0-9.]{0,40}"
        r"(0\.\d{2,5})",
        haystack,
        flags=re.I,
    )
    # Also count plain alpha-boundary decimals mentioned with no keyword
    # prefix, but only when 4+ digits precision is used (0.0XXX).
    alpha_loose = re.findall(r"0\.0\d{2,4}", haystack)
    candidates = set(alpha_anchored + alpha_loose)
    # Multi-endpoint design usually has 2+ distinct thresholds between
    # 0.0001 and 0.05 — typical GRADE p-boundary range. Threshold like
    # 0.5+ are CI/Hazard ratios, not alpha boundaries, so we exclude them.
    alpha_like = {
        v for v in candidates
        if v.startswith("0.0") and 0.0001 <= float(v) <= 0.05
    }
    if len(alpha_like) >= 2:
        return True
    return False


def _has_rct_floor_exemption(reasoning: str | None, effect_size: str | None) -> bool:
    """Returns True when the RCT-floor leniency must be **skipped**.

    Combination of two checks:
      1. The trial exposes **multiple primary endpoints**
         (via :func:`_has_multiple_primary_endpoints`).
      2. At least one endpoint was not reached / showed a negative trend
         (via the existing strict-threshold detector
         :func:`_detect_strict_threshold_failure`).

    When True, :func:`_apply_rct_floor` skips its leniency cap and instead
    forces ``imprecision="高"`` so the failure is reflected in the score.
    """
    if not _has_multiple_primary_endpoints(reasoning, effect_size):
        return False
    return _detect_strict_threshold_failure(reasoning)


def _apply_rct_floor(result: GRADEResult) -> tuple[GRADEResult, str | None]:
    """Apply the **RCT 评分下限规则** leniency **or its exemption**.

    Default floor path (RCT-family, N≥300, P<0.05, no severe evidence):
    cap ``imprecision`` and ``risk_of_bias`` to "低" so the composite
    score reaches ≥ 9.5 ("高" bucket).

    Exemption path (multi-primary-endpoint trial where at least one key
    endpoint failed): **no cap**; instead bump ``imprecision`` to "高"
    (unless already "高") so the failure is reflected in the score with
    its full −1.5 deduction.

    Returns:
        (possibly mutated result, trace note or None). The trace note
        starts with ``[RCT 评分下限]`` (floor applied) or
        ``[RCT-floor 豁免]`` (exemption applied) so reviewers can audit.
    """
    if result.study_design not in {"RCT", "系统评价", "Meta分析"}:
        return result, None
    if not result.sample_size or result.sample_size < 300:
        return result, None
    p = _extract_pvalue_composite(result.effect_size, result.grade_reasoning)
    if p is None or p >= 0.05:
        return result, None
    if _has_severe_evidence(result.grade_reasoning):
        return result, None

    # ---- exemption path: multi-primary + at least one failed ----
    if _has_rct_floor_exemption(result.grade_reasoning, result.effect_size):
        if result.imprecision == "高":
            note = (
                f"[RCT-floor 豁免] 多主终点 + 关键终点未达, "
                f"imprecision 维持=高 (正常扣分); floor 规则不适用"
            )
            return result, note
        new_result = result.model_copy(update={"imprecision": "高"})
        note = (
            f"[RCT-floor 豁免] 多主终点 + 关键终点未达 → "
            f"imprecision:{result.imprecision}→高 (正常 −1.5); "
            f"RCT-floor 上限规则不适用"
        )
        return new_result, note

    # ---- default floor path: cap imprecision / risk_of_bias to 低 ----
    new_imprecision = result.imprecision
    new_risk = result.risk_of_bias
    capped_fields: list[str] = []
    if result.imprecision in ("中", "高"):
        new_imprecision = "低"
        capped_fields.append(f"imprecision:{result.imprecision}→低")
    if result.risk_of_bias in ("中", "高"):
        new_risk = "低"
        capped_fields.append(f"risk_of_bias:{result.risk_of_bias}→低")

    if not capped_fields:
        return result, None

    note = (
        f"[RCT 评分下限] N={result.sample_size}≥300 & 主要终点P={p}<0.05, "
        f"无严重证据 → {', '.join(capped_fields)}"
    )
    return (
        result.model_copy(
            update={"imprecision": new_imprecision, "risk_of_bias": new_risk}
        ),
        note,
    )


# --- top-level entry point --------------------------------------------------


def grade_extraction(
    chunks: Sequence[str],
    *,
    llm: "OpenAI" | None = None,
    cfg: AppConfig | None = None,
    canonical_file_name: str | None = None,
    chunk_file_names: Sequence[str] | None = None,
) -> GRADEResult:
    """Run LLM extraction over ``chunks`` then apply downgrade rules.

    Args:
        chunks: Top-K literature excerpts (plain text).
        llm: Pre-built LLM. When ``None``, one is constructed from
            ``cfg`` (default ``load_config()``).
        cfg: App config. Ignored when ``llm`` is provided.
        canonical_file_name: The **primary** source PDF this scenario is
            about. When present, it's used as the lookup key into
            :mod:`utils.trial_registry`. Pass ``""`` to disable the
            registry for this call.
        chunk_file_names: Optional per-chunk ``file_name`` list (parallel
            to ``chunks``). Reserved for future per-chunk overrides —
            currently informational only.

    Returns:
        A fully populated :class:`GRADEResult` whose ``overall_grade`` and
        ``grade_reasoning`` reflect the rule-engine output (with any
        registry-driven overrides noted in the reasoning).
    """
    cfg = cfg or load_config()
    if llm is None:
        llm = _build_llm(cfg)

    if not chunks:
        raise ValueError("grade_extraction requires at least one chunk")

    prompt = (
        _EXTRACTION_INSTRUCTION.format(n_chunks=len(chunks))
        + "\n\n"
        + _format_chunks(chunks)
    )

    # ``complete`` returns a ``CompletionResponse`` with a ``.text`` field.
    response = llm.chat.completions.create(
        model=cfg.extras.get("llm_model") or os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    raw = _coerce_to_dict(response.choices[0].message.content or "")

    # The LLM must NOT set overall_grade; if it does, drop it and let the
    # rule engine recompute.
    raw.pop("overall_grade", None)

    # Cross-paper sanity check (telemetry): if the retrieved chunks span
    # more than one file_name, the prompt above will have already warned
    # the LLM. Surface this fact in the saved reasoning so reviewers can
    # audit cross-paper contamination quickly. (Pure-logic check — does
    # not block grading.)
    cross_paper_note: str | None = None
    if chunk_file_names:
        unique_sources = {fn for fn in chunk_file_names if fn}
        if len(unique_sources) > 1:
            cross_paper_note = (
                f"[跨文献检索] 本场景 chunks 命中 {len(unique_sources)} 个不同 "
                f"file_name: {sorted(unique_sources)}; 数据仅来自 "
                f"canonical_file_name={canonical_file_name!r} 的片段"
            )

    # Defensive coercion: sample_size may come back as int / float / str.
    # Funnel everything through ``_coerce_sample_size`` so the rule engine's
    # ``sample_size < 300 → −1`` check can fire reliably.
    llm_sample_size = _coerce_sample_size(raw.get("sample_size"))
    raw["sample_size"] = llm_sample_size

    # Defensive coercion: the 5 risk domains only accept 高/中/低. When the
    # LLM says "未报告" / "未知" / "n/a", fall back to "中".
    for domain in (
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
    ):
        if domain in raw:
            raw[domain] = _normalize_risk(raw[domain])

    # Defensive coercion: study_design must match the Literal. Funnel
    # unrecognised values through ``_normalize_design``.
    if "study_design" in raw:
        raw["study_design"] = _normalize_design(raw["study_design"])

    # Registry override: consult the trial registry using the canonical
    # file_name the caller declared for this scenario (not the noisy
    # set of retrieved-chunk file_names, which may include unrelated
    # papers that happen to mention the trial).
    override_note: str | None = None
    if canonical_file_name:
        resolved, source = resolve_study_design(
            canonical_file_name, raw.get("study_design", "")
        )
        if source == "registry" and resolved != raw.get("study_design"):
            override_note = (
                f"[试验注册表覆盖] canonical_file_name={canonical_file_name!r} "
                f"将 LLM 输出 study_design={raw.get('study_design')!r} "
                f"替换为 {resolved!r}"
            )
            raw["study_design"] = resolved

    # Fallback: if the LLM emitted 0 / blank for sample_size — or returned a
    # value that turned out to be unparseable — try to recover the enrolled
    # cohort size from the raw chunks via a regex sweep. Without this the
    # rule engine's ``sample_size < 300 → −1`` line never fires and
    # single-arm trials (McGregor-DAD, ABACUS, …) silently inherit the
    # stronger baseline grade.
    sample_size_fallback_note: str | None = None
    if not raw.get("sample_size") and chunks:
        recovered = _extract_sample_size_from_chunks(chunks)
        if recovered:
            raw["sample_size"] = recovered
            sample_size_fallback_note = (
                f"[样本量兜底] LLM 未返回有效样本量, "
                f"从 chunks 正则匹配到 N={recovered}"
            )

    # Final-authority registry override: when the trial registry has a
    # known enrolled cohort size for ``canonical_file_name``, that value
    # wins over both the LLM output AND any chunk-regex recovery. This is
    # what closes the "retriever pulled EV-302 chunks for a
    # McGregor-DAD query, LLM copied 886" failure mode definitively.
    registry_n_note: str | None = None
    if canonical_file_name:
        registry_n = canonical_sample_size(canonical_file_name)
        if registry_n and registry_n != raw.get("sample_size"):
            registry_n_note = (
                f"[样本量注册表覆盖] canonical_file_name={canonical_file_name!r} "
                f"将 sample_size={raw.get('sample_size')} 替换为 {registry_n}"
            )
            raw["sample_size"] = registry_n

    # Validate against the pydantic schema. ``overall_grade`` is required by
    # the schema, so seed it with "高" before model validation — the rule
    # engine will overwrite it. The composite-score fields are also required
    # by the schema (no default behaviour unless the model declares one) —
    # fall back to 0.0 + a placeholder so an LLM that forgot the new fields
    # still validates. Reviewers can spot the gap via the "未提供" prefix.
    raw.setdefault("overall_grade", "高")
    raw.setdefault("composite_score", 0.0)
    raw.setdefault("score_breakdown", "未提供(LLM 漏返该字段)")

    result = GRADEResult.model_validate(raw)

    # ---- Composite-score override (rule engine mirror) ----
    # The LLM is *asked* to compute composite_score per the system-prompt
    # rules, but it tends to drift (e.g. SWOG-S1605's "效应量=中等→+0.5"
    # when the paper has no HR at all). Python recomputes deterministically
    # and overrides the LLM's value — log a trace note when they disagree so
    # reviewers can audit. The LLM's score is preserved on first line of
    # the trace and Python's replaces it on the next line.
    composite_note: str | None = None
    try:
        llm_score_raw = raw.get("composite_score", 0.0)
        llm_breakdown = raw.get("score_breakdown", "未提供") or "未提供"
        py_score, py_breakdown, py_debug = _compute_composite_score_breakdown(result)
        diff = abs(float(llm_score_raw) - float(py_score))
        if diff >= 0.5 or llm_breakdown == "未提供(LLM 漏返该字段)":
            composite_note = (
                f"[综合评分重算] LLM={llm_score_raw} / Python={py_score} "
                f"(差异={diff:.2f}); 以 Python 覆盖。breakdown: {py_breakdown}"
            )
        else:
            composite_note = f"[综合评分校验] LLM={llm_score_raw} ≈ Python={py_score} (一致)"
        # Always override the displayed value with the deterministic Python
        # one — this is the entire point of having a rule-engine mirror.
        result = result.model_copy(
            update={
                "composite_score": py_score,
                "score_breakdown": py_breakdown,
            }
        )
    except Exception as exc:  # pragma: no cover — never let scorer break grading
        composite_note = f"[综合评分重算失败] {type(exc).__name__}: {exc}"

    # ---- RCT 评分下限规则 ----
    # Big-N (≥ 300) RCTs whose primary endpoint hits P<0.05 — without
    # counter-evidence (dropout >30%, fraud, …) — get their imprecision /
    # risk_of_bias capped at "低". This lifts them out of the
    # "Penalty-defaults-to-中" failure mode the LLM often exhibits when the
    # abstract doesn't explicitly defend the trial's precision. After the
    # cap, we recompute composite_score once more so the displayed value
    # matches the (now-relaxed) risk domains.
    rct_floor_note: str | None = None
    pre_floor_design = result.study_design
    pre_floor_size = result.sample_size
    try:
        capped_result, rct_floor_note = _apply_rct_floor(result)
        if rct_floor_note:
            result = capped_result
            # Recompute composite_score with the relaxed domains; the new
            # value usually lands at ≥ 9.5 (i.e. "高" bucket per the new
            # 8.5/6.0/3.5/0 mapping).
            py_score2, py_breakdown2, _ = _compute_composite_score_breakdown(result)
            result = result.model_copy(
                update={
                    "composite_score": py_score2,
                    "score_breakdown": py_breakdown2,
                }
            )
            composite_note = (
                f"{rct_floor_note}; [综合评分重算(应用 RCT 下限)] "
                f"score={py_score2:.1f}; breakdown: {py_breakdown2}"
            )
    except Exception as exc:  # pragma: no cover
        rct_floor_note = f"[RCT 评分下限失败] {type(exc).__name__}: {exc}"

    # Stash any rule-engine pre-notes (registry override / sample-size
    # fallback / cross-paper warning) so they appear in ``grade_reasoning``
    # for human review.
    extra_notes: list[str] = []
    if cross_paper_note:
        extra_notes.append(cross_paper_note)
    if override_note:
        extra_notes.append(override_note)
    if sample_size_fallback_note:
        extra_notes.append(sample_size_fallback_note)
    if registry_n_note:
        extra_notes.append(registry_n_note)
    if composite_note:
        extra_notes.append(composite_note)
    if extra_notes:
        result = result.model_copy(
            update={
                "grade_reasoning": (
                    (result.grade_reasoning or "").rstrip()
                    + "\n"
                    + "\n".join(extra_notes)
                ).strip()
            }
        )

    # Run the legacy rule engine for its *trace* (the [降级规则引擎] block
    # is useful for reviewers to compare the two scoring approaches), but
    # the final overall_grade is overwritten below to follow the
    # composite_score bucket.
    result = apply_downgrade_rules(result)

    # ---- 强制 overall_grade 服从 bucket ----
    bucket = _grade_to_bucket(result.composite_score)
    if result.overall_grade != bucket:
        bucket_note = (
            f"[等级映射] composite_score={result.composite_score:.1f} "
            f"→ overall_grade={bucket} "
            f"(阈值 8.5/6.0/3.5/0; 规则引擎此前输出 '{result.overall_grade}')"
        )
        result = result.model_copy(
            update={
                "overall_grade": bucket,
                "grade_reasoning": (
                    (result.grade_reasoning or "").rstrip()
                    + "\n"
                    + bucket_note
                ),
            }
        )

    return result


# --- CLI self-test ----------------------------------------------------------


_TEST_QUERY = "enfortumab vedotin pembrolizumab EV-302 urothelial cancer advanced"
_DEFAULT_TOP_K = 5


def _retrieve_chunks(query: str, top_k: int, cfg: AppConfig) -> list[str]:
    index = _open_index(cfg)
    retriever = index.as_retriever(similarity_top_k=top_k)
    hits = retriever.retrieve(query)
    return [h.node.get_content() for h in hits]


def main(
    query: str = _TEST_QUERY,
    top_k: int = _DEFAULT_TOP_K,
    *,
    cfg: AppConfig | None = None,
) -> GRADEResult:
    cfg = cfg or load_config()
    if not cfg.effective_openai_api_key():
        warnings.warn(
            "OPENAI_API_KEY is not set; set it (or openai_api_key in config.yaml) "
            "before running the LLM stage.",
            stacklevel=2,
        )
    chunks = _retrieve_chunks(query, top_k, cfg)
    if not chunks:
        raise RuntimeError("No chunks retrieved — is the vector store populated?")
    # ``cfg`` and ``llm`` are rebuilt here so the LLM model name stays in
    # sync with config (avoids passing a now-stale cached ``cfg`` into
    # ``grade_extraction``).
    result = grade_extraction(chunks, llm=_build_llm(cfg), cfg=cfg)
    print("=" * 72)
    print(f"GRADE evidence profile (top-{top_k} chunks for: {query!r})")
    print("=" * 72)
    print(result.model_dump_json(indent=2, ensure_ascii=False))
    return result


__all__ = [
    "Confounding",
    "GRADEResult",
    "OverallGrade",
    "RiskLevel",
    "StudyDesign",
    "_DESIGN_ALIASES",
    "_normalize_design",
    "_normalize_risk",
    "_compute_composite_score_breakdown",
    "_extract_hr",
    "_extract_pvalue",
    "_grade_to_bucket",
    "apply_downgrade_rules",
    "grade_extraction",
    "main",
]


if __name__ == "__main__":
    main()