"""Shared prompt fragments for the OncoImmunoRAG project.

Keep cross-module prompt wording here so the retrieval, grading, and any
future evaluation prompts don't drift apart.
"""

from __future__ import annotations


_STATISTICAL_SIGNIFICANCE_RULES = """\
【统计学显著性判断规则】
1. 默认显著性阈值为双侧 P < 0.05。
2. 若原文明确标注了校正后的显著性界值（例如 "prespecified significance threshold of alpha = 0.001" 或 "interim alpha spending"），
   必须以该研究预设的 alpha 阈值作为唯一判定标准。
3. 若 P 值未达到预设的严格阈值，必须准确表述为：
   "由于预设 alpha 分配较严格（如 alpha=0.001），当前 P=0.004 未达到预设的统计学显著性界值"，
   严禁笼统概括为"该差异不显著"或"P 值未达统计学显著"。
"""


_DISCUSSION_SECTION_RULE = """\
【Discussion 段处理规则】
讨论（Discussion）段落中引用的其他试验的数据，即使你在回答里把它们写在 A 区
"本研究的直接结论" 之下，也必须按 B 区规则处理：用"根据该文献引用的 XX 研究……"
开头表述，且句末【严禁】附加 [citation: 文件名] —— 因为这些数据是其他试验的结果，
不是当前文献自身报告的结论。
仅当数据来自该文献的 Results / Conclusions / Abstract 段时，才允许写在 A 区并
附 [citation: 文件名]。
"""


def statistical_significance_rules() -> str:
    """Return the markdown block on statistical-significance judgment rules."""
    return _STATISTICAL_SIGNIFICANCE_RULES


def discussion_section_rule() -> str:
    """Return the markdown block on Discussion-section citation handling.

    Discussion paragraphs often quote other trials' numbers — the LLM tends
    to surface those under the "A. direct conclusions" heading even though
    they are not the source paper's own conclusions. This block tells the
    model to treat every Discussion-sourced datum as an indirect citation,
    regardless of which A/B heading it sits under.
    """
    return _DISCUSSION_SECTION_RULE


__all__ = ["statistical_significance_rules", "discussion_section_rule"]