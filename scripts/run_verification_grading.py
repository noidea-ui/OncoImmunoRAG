"""GRADE grading over the five verification papers in ``verification/``.

Unlike :mod:`scripts.run_batch_4` (which grades the 12 canonical scenarios
already ingested into the vector store), this script reads NEW PDFs from
``verification/`` directly, chunks them, and runs
:func:`grading.grade_evaluator.grade_extraction` once per paper. Each
``canonical_file_name`` is set to the paper's slug so the trial registry
override path still fires — the registry has no entries for the
verification set, but the call stays structurally identical to production.

Outputs:

* ``verification/results/<slug>.json``  — full GRADEResult + retrievals
* ``verification/results/_summary.csv`` — one-row-per-paper table
* ``verification/results/_summary.md``  — Markdown overview
* stdout                              — human-readable summary

Typical use::

    poetry run python scripts/run_verification_grading.py
    poetry run python scripts/run_verification_grading.py --slug KEYNOTE-361 JAVELIN-B100
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
# Paper catalogue: the 5 verification PDFs and the slugs we expose them under
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationPaper:
    """One PDF under ``verification/`` plus its short slug + title."""

    slug: str
    title: str
    pdf_path: Path
    canonical_file: str  # used as the registry key in grade_extraction


_VERIFICATION_PAPERS: tuple[VerificationPaper, ...] = (
    VerificationPaper(
        slug="KEYNOTE-361",
        title=(
            "KEYNOTE-361: Pembrolizumab + Chemotherapy vs Chemotherapy 1L "
            "Advanced Urothelial Carcinoma (Lancet Oncol 2021)"
        ),
        pdf_path=_PROJECT_ROOT / "verification" / "1-s2.0-S1470204521001522-main.pdf",
        canonical_file="1-s2.0-S1470204521001522-main.pdf",
    ),
    VerificationPaper(
        slug="BOND-003-CohortC",
        title=(
            "BOND-003 Cohort C: Intravesical Cretostimogene in BCG-Unresponsive "
            "HR-NMIBC with CIS — Single-Arm Phase 3 (Lancet Oncol 2026)"
        ),
        pdf_path=_PROJECT_ROOT / "verification" / "1-s2.0-S1470204526001944-main.pdf",
        canonical_file="1-s2.0-S1470204526001944-main.pdf",
    ),
    VerificationPaper(
        slug="JAVELIN-B100",
        title=(
            "JAVELIN Bladder 100: Avelumab Maintenance Therapy for Advanced "
            "or Metastatic Urothelial Carcinoma (NEJM 2020)"
        ),
        pdf_path=_PROJECT_ROOT / "verification" / "NEJMoa2002788.pdf",
        canonical_file="NEJMoa2002788.pdf",
    ),
    VerificationPaper(
        slug="CheckMate-274",
        title=(
            "CheckMate 274: Adjuvant Nivolumab vs Placebo in Muscle-Invasive "
            "Urothelial Carcinoma (NEJM 2021)"
        ),
        pdf_path=_PROJECT_ROOT / "verification" / "NEJMoa2034442.pdf",
        canonical_file="NEJMoa2034442.pdf",
    ),
    VerificationPaper(
        slug="SWOG-S1605",
        title=(
            "SWOG S1605: Phase 2 Atezolizumab in BCG-Unresponsive HR-NMIBC "
            "(2024)"
        ),
        pdf_path=_PROJECT_ROOT / "verification" / "nihms-1939489.pdf",
        canonical_file="nihms-1939489.pdf",
    ),
)


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------


def _load_pdf_text(pdf_path: Path) -> str:
    """Read every page from ``pdf_path`` and concatenate the extracted text.

    Uses :mod:`pypdf` directly (rather than ``SimpleDirectoryReader`` /
    ``PDFReader``) to avoid the file-extractor registration dance and to
    keep this script runnable without the ``llama-index-readers-file``
    package. Surfaces a clean error if the file is missing or unreadable.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        import pypdf  # local import — keeps module importable without pypdf
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
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required to read verification PDFs — install with "
            "`pip install pypdf`"
        ) from exc


def _split_into_chunks(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Chunk ``text`` with the same :class:`SentenceSplitter` as ingestion.

    Each emitted string is fed straight to the LLM grader, so we preserve
    whitespace + paragraph structure (the LLM tolerates it and SectionSplitter
    splits at sentence boundaries that the rule-engine ignores).
    """
    if not text.strip():
        return []
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(
        [Document(text=text, metadata={"file_name": "<verification>"})],
        show_progress=False,
    )
    return [n.get_content() for n in nodes]


# ---------------------------------------------------------------------------
# Per-paper grading
# ---------------------------------------------------------------------------


def _grade_one(paper: VerificationPaper, *, cfg, top_k: int, dry_run: bool) -> dict:
    """Run :func:`grade_extraction` on one paper. Returns a JSON-serialisable dict."""
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    full_text = _load_pdf_text(paper.pdf_path)
    chunks = _split_into_chunks(
        full_text,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    # The grader hard-asserts ``len(chunks) > 0``; if the PDF text came out
    # empty, surface a clear error rather than letting the LLM step crash.
    if not chunks:
        raise RuntimeError(
            f"No chunks extracted from {paper.pdf_path.name} — PDF parsing failed?"
        )

    # Use up to ``top_k`` chunks (default 12 to cover abstract + results +
    # conclusions + endpoints — these trials report 1–3 primary endpoints
    # plus safety; 12 chunks is enough to give the LLM full coverage without
    # blowing the context window). Order matters: chunks come out of
    # SectionSplitter in document order, which is exactly what the LLM needs.
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

    # All chunks here come from the SAME paper, so the "cross-paper guard"
    # in the grading prompt is a no-op. Pass canonical_file_name so a future
    # registry entry for this paper still triggers overrides.
    chunk_file_names = [paper.canonical_file] * len(used_chunks)
    try:
        result: GRADEResult = grade_extraction(
            used_chunks,
            canonical_file_name=paper.canonical_file,
            chunk_file_names=chunk_file_names,
        )
        grade_dict: dict | None = json.loads(result.model_dump_json())
        error: str | None = None
    except Exception as exc:  # pragma: no cover — propagated to caller
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
    print(f"{'Verification GRADE grading summary · 5 papers':^100}")
    print(bar)
    print(
        f"{'slug':<18} {'grade':<6} {'design':<10} {'N':>6} "
        f"{'duration':>8}  status"
    )
    print("-" * 100)
    for row in rows:
        g = row.get("grade") or {}
        status = (
            "OK" if not row.get("error")
            else f"ERR: {row['error'][:36]}"
        )
        n = g.get("sample_size", 0) if g else 0
        dur = f"{row.get('duration_s', 0):.2f}s"
        print(
            f"{row['slug']:<18} "
            f"{(g.get('overall_grade') or 'ERR'):<6} "
            f"{(g.get('study_design') or '—'):<10} "
            f"{n:>6}  "
            f"{dur:>8}  "
            f"{status}"
        )
    print(bar)
    n_ok = sum(1 for r in rows if not r.get("error"))
    n_err = len(rows) - n_ok
    print(f"  · Total   : {len(rows)}")
    print(f"  · OK      : {n_ok}")
    print(f"  · Errors  : {n_err}")
    print(bar)


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write a per-paper CSV (one row per paper, GRADE fields as columns)."""
    cols = [
        "slug",
        "title",
        "pdf",
        "canonical_file",
        "overall_grade",
        "study_design",
        "sample_size",
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
        "confounding_by_subsequent_therapy",
        "effect_size",
        "duration_s",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            g = row.get("grade") or {}
            vals = []
            for c in cols:
                if c in ("slug", "title", "pdf", "canonical_file", "duration_s", "error"):
                    v = row.get(c, "")
                else:
                    v = g.get(c, "")
                # CSV-safe quoting
                v = "" if v is None else str(v)
                v = v.replace('"', '""')
                vals.append(f'"{v}"')
            fh.write(",".join(vals) + "\n")


def _write_markdown(rows: list[dict], generated_at: str, path: Path) -> None:
    """Markdown summary — readable in the terminal and on GitHub."""
    lines: list[str] = [
        "# Verification GRADE grading · 5 论文",
        "",
        f"_Generated at: {generated_at}_",
        "",
        "| Slug | 等级 | 研究设计 | N | Effect size | 时长 | 状态 |",
        "|---|:---:|:---|---:|---|---:|---|",
    ]
    for row in rows:
        g = row.get("grade") or {}
        status = "✅" if not row.get("error") else f"❌ {row['error'][:60]}"
        effect = (g.get("effect_size") or "—").replace("|", "\\|")
        title = row["title"][:80] + ("…" if len(row["title"]) > 80 else "")
        lines.append(
            f"| **{row['slug']}** "
            f"| **{g.get('overall_grade') or 'ERR'}** "
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
            f"- 等级：**{g.get('overall_grade', '—')}** · 设计：**{g.get('study_design', '—')}** · N = `{g.get('sample_size', 0)}`",
            f"- 偏倚 / 不一致 / 间接 / 不精确 / 发表偏倚：{g.get('risk_of_bias', '—')} / {g.get('inconsistency', '—')} / {g.get('indirectness', '—')} / {g.get('imprecision', '—')} / {g.get('publication_bias', '—')}",
            f"- 后续治疗混杂：{g.get('confounding_by_subsequent_therapy', '—')}",
            f"- 效应量：`{g.get('effect_size', '—')}`",
            "",
            "<details><summary>grade_reasoning（点击展开）</summary>",
            "",
            "```",
            (g.get("grade_reasoning") or "(空)").strip(),
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
        "--slug",
        nargs="*",
        help="只跑指定 slug（可多次给，默认全部 5 个）",
    )
    p.add_argument(
        "--top-k", type=int, default=12,
        help="每篇 PDF 喂给 LLM 的 chunk 数（默认 12）",
    )
    p.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "verification" / "results"),
        help="结果输出目录（默认 verification/results）",
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
            "[verify-grading] OPENAI_API_KEY 未配置 — 用 --dry-run 跳过 LLM，"
            "或在 config.yaml / env 配置 key",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(_VERIFICATION_PAPERS)
    if args.slug:
        wanted = set(args.slug)
        selected = [p for p in _VERIFICATION_PAPERS if p.slug in wanted]
        missing = wanted - {p.slug for p in selected}
        if missing:
            print(
                f"[verify-grading] 未知 slug：{sorted(missing)} · 可选值："
                f"{[p.slug for p in _VERIFICATION_PAPERS]}",
                file=sys.stderr,
            )
            return 2

    print("=" * 100)
    print(f"{'verification/5 论文 GRADE 分级 · ' + ('LIVE LLM' if not args.dry_run else 'DRY RUN'):^100}")
    print("=" * 100)
    for p in selected:
        flag = p.pdf_path.is_file()
        print(f"  · {p.slug:<18}  {'✓' if flag else '✗'}  {p.pdf_path.name}")

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
        # Persist
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


__all__ = [
    "VerificationPaper",
    "_VERIFICATION_PAPERS",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
