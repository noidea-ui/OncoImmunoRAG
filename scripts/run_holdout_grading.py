"""Independent GRADE grading for the 5 holdout papers.

Mirrors :mod:`scripts.run_verification_grading` but reads from
``holdout/`` and writes to ``holdout/results/``. Used as an independent
test set — these 5 papers are NEITHER in ``data/raw/`` (used to build the
vector store) NOR in ``verification/`` (the previous 5-paper batch).

Typical use::

    poetry run python scripts/run_holdout_grading.py
    poetry run python scripts/run_holdout_grading.py --slug IMvigor211 DANUBE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Bootstrap ``src`` onto sys.path (mirrors the other scripts/ entries).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llama_index.core import Document  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402

from grading.grade_evaluator import GRADEResult, grade_extraction  # noqa: E402
from utils.config_loader import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Paper catalogue: the 5 holdout PDFs and the slugs we expose them under.
# Files are intentionally identified by PII-style DOI numbers; readers
# can map each slug to a publication via the metadata below.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutPaper:
    """One PDF under ``holdout/`` plus its short slug + title."""

    slug: str
    title: str
    pdf_path: Path
    canonical_file: str  # used as the registry key in grade_extraction


_HOLDOUT_PAPERS: tuple[HoldoutPaper, ...] = (
    HoldoutPaper(
        slug="IMvigor211",
        title=(
            "IMvigor211: Atezolizumab vs Chemotherapy in Platinum-Treated mUC "
            "(Lancet 2018)"
        ),
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S014067361733297X-main.pdf",
        canonical_file="1-s2.0-S014067361733297X-main.pdf",
    ),
    HoldoutPaper(
        slug="PIVOT-10",
        title=(
            "PIVOT-10: Bempegaldesleukin + Nivolumab in 1L Advanced/Metastatic UC "
            "(Urol Oncol 2024)"
        ),
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1078143924006719-main.pdf",
        canonical_file="1-s2.0-S1078143924006719-main.pdf",
    ),
    HoldoutPaper(
        slug="CheckMate-275",
        title=(
            "CheckMate 275: Nivolumab Monotherapy in Metastatic UC after Platinum "
            "(Lancet Oncol 2017)"
        ),
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1470204517300657-main.pdf",
        canonical_file="1-s2.0-S1470204517300657-main.pdf",
    ),
    HoldoutPaper(
        slug="DANUBE",
        title=(
            "DANUBE: Durvalumab ± Tremelimumab vs Chemotherapy in 1L mUC "
            "(Lancet Oncol 2020)"
        ),
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1470204520305416-main.pdf",
        canonical_file="1-s2.0-S1470204520305416-main.pdf",
    ),
    HoldoutPaper(
        slug="Mamede-MetaAnalysis",
        title=(
            "Adjuvant Immunotherapy in HR-MIUC: Updated Meta-Analysis of 3 RCTs "
            "(Clin Genitourin Cancer 2025)"
        ),
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1558767324002581-main.pdf",
        canonical_file="1-s2.0-S1558767324002581-main.pdf",
    ),
)


# ---------------------------------------------------------------------------
# PDF parsing (identical to run_verification_grading)
# ---------------------------------------------------------------------------


def _load_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        chunks: list[str] = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text:
                chunks.append(page_text)
        return "\n\n".join(chunks)
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to read holdout PDFs — install with `pip install pypdf`"
        ) from exc


def _split_into_chunks(
    text: str, *, chunk_size: int, chunk_overlap: int
) -> list[str]:
    if not text.strip():
        return []
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(
        [Document(text=text, metadata={"file_name": "<holdout>"})],
        show_progress=False,
    )
    return [n.get_content() for n in nodes]


# ---------------------------------------------------------------------------
# Per-paper grading
# ---------------------------------------------------------------------------


def _grade_one(paper: HoldoutPaper, *, cfg, top_k: int, dry_run: bool) -> dict:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    full_text = _load_pdf_text(paper.pdf_path)
    chunks = _split_into_chunks(
        full_text,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError(
            f"No chunks extracted from {paper.pdf_path.name} — PDF parsing failed?"
        )

    used_chunks = chunks[: max(top_k, 1)]

    if dry_run:
        return {
            "slug": paper.slug,
            "title": paper.title,
            "pdf": paper.pdf_path.name,
            "canonical_file": paper.canonical_file,
            "started_at": started_at,
            "duration_s": round(time.monotonic() - started, 3),
            "dry_run": True,
            "n_chunks_total": len(chunks),
            "n_chunks_used": len(used_chunks),
            "first_chunk_preview": used_chunks[0][:240] if used_chunks else "",
            "grade": None,
            "error": None,
        }

    chunk_file_names = [paper.canonical_file] * len(used_chunks)
    try:
        result: GRADEResult = grade_extraction(
            used_chunks,
            canonical_file_name=paper.canonical_file,
            chunk_file_names=chunk_file_names,
        )
        grade_dict: dict | None = json.loads(result.model_dump_json())
        error: str | None = None
    except Exception as exc:
        grade_dict = None
        error = f"{type(exc).__name__}: {exc}"

    return {
        "slug": paper.slug,
        "title": paper.title,
        "pdf": paper.pdf_path.name,
        "canonical_file": paper.canonical_file,
        "started_at": started_at,
        "duration_s": round(time.monotonic() - started, 3),
        "dry_run": False,
        "n_chunks_total": len(chunks),
        "n_chunks_used": len(used_chunks),
        "grade": grade_dict,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_console_summary(rows: list[dict]) -> None:
    bar = "=" * 100
    print()
    print(bar)
    print(f"{'Holdout GRADE grading summary · 5 papers':^100}")
    print(bar)
    print(
        f"{'slug':<22} {'grade':<4} {'comp':>5} {'bucket':<5} {'design':<10} {'N':>6} "
        f"{'duration':>8}  status"
    )
    print("-" * 100)
    for row in rows:
        g = row.get("grade") or {}
        score = g.get("composite_score", 0) if g else 0
        if score >= 8.5: bucket = "高"
        elif score >= 6.0: bucket = "中"
        elif score >= 3.5: bucket = "低"
        else: bucket = "极低"
        status = (
            "OK" if not row.get("error")
            else f"ERR: {row['error'][:36]}"
        )
        n = g.get("sample_size", 0) if g else 0
        dur = f"{row.get('duration_s', 0):.2f}s"
        ov = g.get("overall_grade") or "—"
        design = g.get("study_design") or "—"
        print(
            f"{row['slug']:<22} {ov:<4} {score:>5.1f} {bucket:<5} {design:<10} {n:>6} "
            f"{dur:>8}  {status}"
        )
    print(bar)
    n_ok = sum(1 for r in rows if not r.get("error"))
    n_err = len(rows) - n_ok
    print(f"  · Total   : {len(rows)}")
    print(f"  · OK      : {n_ok}")
    print(f"  · Errors  : {n_err}")
    print(bar)


def _write_csv(rows: list[dict], path: Path) -> None:
    cols = [
        "slug", "title", "pdf", "canonical_file",
        "overall_grade", "study_design", "sample_size",
        "risk_of_bias", "inconsistency", "indirectness", "imprecision",
        "publication_bias", "confounding_by_subsequent_therapy",
        "effect_size", "composite_score", "duration_s", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            g = row.get("grade") or {}
            vals = []
            for c in cols:
                if c in ("slug", "title", "pdf", "canonical_file",
                         "composite_score", "duration_s", "error"):
                    v = row.get(c, "")
                else:
                    v = g.get(c, "")
                v = "" if v is None else str(v)
                v = v.replace('"', '""')
                vals.append(f'"{v}"')
            fh.write(",".join(vals) + "\n")


def _write_markdown(rows: list[dict], generated_at: str, path: Path) -> None:
    lines: list[str] = [
        "# Holdout GRADE grading · 5 论文 (独立验证集)",
        "",
        f"_Generated at: {generated_at}_",
        "",
        "| Slug | 等级 | composite | 设计 | N | Effect size | 时长 | 状态 |",
        "|---|:---:|:---:|:---|---:|---|---:|---|",
    ]
    for row in rows:
        g = row.get("grade") or {}
        score = g.get("composite_score", 0)
        if score >= 8.5: bucket = "高"
        elif score >= 6.0: bucket = "中"
        elif score >= 3.5: bucket = "低"
        else: bucket = "极低"
        status = "✅" if not row.get("error") else f"❌ {row['error'][:60]}"
        effect = (g.get("effect_size") or "—").replace("|", "\\|")
        title = row["title"][:80] + ("…" if len(row["title"]) > 80 else "")
        lines.append(
            f"| **{row['slug']}** "
            f"| **{g.get('overall_grade') or 'ERR'}** "
            f"| {score:.1f} ({bucket}) "
            f"| {g.get('study_design') or '—'} "
            f"| {g.get('sample_size', 0) or '—'} "
            f"| `{effect}` "
            f"| {row.get('duration_s', 0):.2f}s "
            f"| {status} |"
        )
    lines += [
        "",
        f"- 论文总数 **{len(rows)}** · 成功 **{sum(1 for r in rows if not r.get('error'))}** · 失败 **{sum(1 for r in rows if r.get('error'))}**",
        "",
        "## 各论文明细",
        "",
    ]
    for row in rows:
        g = row.get("grade") or {}
        lines += [
            f"### {row['slug']}",
            f"- 标题：{row['title']}",
            f"- PDF：`{row['pdf']}`",
            f"- 等级：**{g.get('overall_grade', '—')}** · composite={g.get('composite_score', 0)} → bucket={bucket} · 设计：**{g.get('study_design', '—')}** · N = `{g.get('sample_size', 0)}`",
            f"- 偏倚 / 不一致 / 间接 / 不精确 / 发表偏倚：{g.get('risk_of_bias', '—')} / {g.get('inconsistency', '—')} / {g.get('indirectness', '—')} / {g.get('imprecision', '—')} / {g.get('publication_bias', '—')}",
            f"- 后续治疗混杂：{g.get('confounding_by_subsequent_therapy', '—')}",
            f"- 效应量：`{g.get('effect_size', '—')}`",
            "",
            "<details><summary>score_breakdown（点击展开）</summary>",
            "",
            "```",
            (g.get("score_breakdown") or "(空)").strip(),
            "```",
            "",
            "</details>",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slug", nargs="*",
        help="只跑指定 slug（默认全部 5）",
    )
    p.add_argument(
        "--top-k", type=int, default=12,
        help="每篇 PDF 喂给 LLM 的 chunk 数（默认 12）",
    )
    p.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "holdout" / "results"),
        help="结果输出目录（默认 holdout/results）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="跳过 LLM 调用，只跑 PDF 解析+chunk",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config()

    if not args.dry_run and not cfg.effective_openai_api_key():
        print(
            "[holdout-grading] OPENAI_API_KEY 未配置 — 用 --dry-run 跳过 LLM，"
            "或在 config.yaml / env 配置 key",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(_HOLDOUT_PAPERS)
    if args.slug:
        wanted = set(args.slug)
        selected = [p for p in _HOLDOUT_PAPERS if p.slug in wanted]
        missing = wanted - {p.slug for p in selected}
        if missing:
            print(
                f"[holdout-grading] 未知 slug：{sorted(missing)} · 可选值："
                f"{[p.slug for p in _HOLDOUT_PAPERS]}",
                file=sys.stderr,
            )
            return 2

    print("=" * 100)
    print(f"{'holdout/5 论文 GRADE 分级 · ' + ('LIVE LLM' if not args.dry_run else 'DRY RUN'):^100}")
    print("=" * 100)
    for p in selected:
        flag = p.pdf_path.is_file()
        print(f"  · {p.slug:<22}  {'✓' if flag else '✗'}  {p.pdf_path.name}")

    rows: list[dict] = []
    for paper in selected:
        print(f"\n>>> 正在处理 {paper.slug} ...")
        try:
            row = _grade_one(paper, cfg=cfg, top_k=args.top_k, dry_run=args.dry_run)
        except Exception as exc:
            row = {
                "slug": paper.slug,
                "title": paper.title,
                "pdf": paper.pdf_path.name,
                "canonical_file": paper.canonical_file,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "duration_s": 0.0,
                "n_chunks_total": 0,
                "n_chunks_used": 0,
                "grade": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        (out_dir / f"{paper.slug}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(row)

    generated_at = datetime.now(timezone.utc).isoformat()
    csv_path = out_dir / "_summary.csv"
    md_path = out_dir / "_summary.md"

    _write_csv(rows, csv_path)
    _write_markdown(rows, generated_at, md_path)

    _print_console_summary(rows)
    print(f"\n输出：")
    print(f"  · 每篇 JSON   → {out_dir}/<slug>.json")
    print(f"  · 汇总 CSV    → {csv_path}")
    print(f"  · 汇总 Markdown → {md_path}")

    return 0 if not any(r.get("error") for r in rows) else 3


__all__ = ["HoldoutPaper", "_HOLDOUT_PAPERS", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
