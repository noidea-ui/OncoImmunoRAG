"""Streamlit front-end for OncoImmunoRAG.

Three-tab interface:
  * Tab 1 · 库内文献分级   — read PDF from disk, GRADE score card
  * Tab 2 · PubMed 检索    — search NCBI, fetch details, grade in temp coll
  * Tab 3 · 手动输入 / PDF  — PMID lookup OR PDF upload, temp collection

The app is intentionally a thin UI layer — it does not modify any of the
existing modules in ``src/`` other than the new ``retrieval/pubmed_client``
and ``retrieval/temp_indexer``.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Streamlit must come before our own modules so set_page_config is first.
# ---------------------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="OncoImmunoRAG · 证据智能分级 (v3 三 Tab 版)",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from llama_index.core import Document  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402

from grading.grade_evaluator import GRADEResult, grade_extraction  # noqa: E402
from retrieval.pubmed_client import (  # noqa: E402
    PubMedDetail,
    PubMedHit,
    fetch_details,
    search_pubmed,
)
from retrieval.query_engine import QueryEngine, QueryResult  # noqa: E402
from retrieval.selector import set_rerank_enabled_default  # noqa: E402
from retrieval.temp_indexer import (  # noqa: E402
    clear_temp_collection,
    index_temp_document,
)
from utils.config_loader import load_config  # noqa: E402


# ===========================================================================
# 1. Paper catalogue (the 22-PDF universe)
# ===========================================================================
_PAPERS_RAW = [
    ("EV-302",       "NEJMoa2312117.pdf",                                                    "RCT",    "data/raw",    "NEJM 2024"),
    ("NIAGARA",      "Powles_2024_NEJM_NIAGARA-Durvalumab.pdf",                              "RCT",    "data/raw",    "NEJM 2024"),
    ("IMvigor011",   "Powles_2025_NEJM_IMvigor011-ctDNA.pdf",                                 "RCT",    "data/raw",    "NEJM 2025"),
    ("IMvigor010",   "Powles_2021_Nature_IMvigor010-ctDNA.pdf",                               "RCT",    "data/raw",    "Nature 2021"),
    ("VESPER",       "Pfister_2024_LancetOncol_VESPER.pdf",                                  "RCT",    "data/raw",    "Lancet Oncol 2024"),
    ("IMvigor130",   "Galsky_2024_CellRepMed_IMvigor130-ChemoImmuno.pdf",                    "RCT",    "data/raw",    "Cell Rep Med 2024"),
    ("KEYNOTE-057",  "Balar_2024_LancetOncol_KEYNOTE-057.pdf",                               "单臂",   "data/raw",    "Lancet Oncol 2024"),
    ("RC48-C014",    "Zhou_2025_AnnOncol_RC48-C014.pdf",                                     "单臂",   "data/raw",    "Ann Oncol 2025"),
    ("Nadofaragene", "Boorjian_2021_LancetOncol_Nadofaragene.pdf",                          "单臂",   "data/raw",    "Lancet Oncol 2021"),
    ("CORE-001",     "Li_2025_NatMed_Cretostimogene-Nivo.pdf",                              "单臂",   "data/raw",    "Nat Med 2025"),
    ("McGregor-DAD", "McGregor_2024_AnnOncol_DAD.pdf",                                      "单臂",   "data/raw",    "Ann Oncol 2024"),
    ("ABACUS",       "Szabados_2022_EurUrol_ABACUS.pdf",                                   "单臂",   "data/raw",    "Eur Urol 2022"),
    ("KEYNOTE-361",       "1-s2.0-S1470204521001522-main.pdf",  "RCT",  "verification", "Lancet Oncol 2021"),
    ("BOND-003 Cohort C", "1-s2.0-S1470204526001944-main.pdf",  "单臂", "verification", "Lancet Oncol 2026"),
    ("JAVELIN Bladder 100","NEJMoa2002788.pdf",                "RCT",  "verification", "NEJM 2020"),
    ("CheckMate 274",     "NEJMoa2034442.pdf",                "RCT",  "verification", "NEJM 2021"),
    ("SWOG S1605",        "nihms-1939489.pdf",                "单臂", "verification", "Eur Urol 2024"),
    ("IMvigor211",        "1-s2.0-S014067361733297X-main.pdf",  "RCT",  "holdout", "Lancet 2018"),
    ("PIVOT-10",          "1-s2.0-S1078143924006719-main.pdf",  "单臂", "holdout", "Urol Oncol 2024"),
    ("CheckMate 275",     "1-s2.0-S1470204517300657-main.pdf",  "单臂", "holdout", "Lancet Oncol 2017"),
    ("DANUBE",            "1-s2.0-S1470204520305416-main.pdf",  "RCT",  "holdout", "Lancet Oncol 2020"),
    ("Mamede-Meta",       "1-s2.0-S1558767324002581-main.pdf",  "Meta", "holdout", "Clin GU Cancer 2025"),
]


def _all_papers():
    out = []
    for display, file_name, design, folder, journal_y in _PAPERS_RAW:
        out.append({
            "display": display,
            "file_name": file_name,
            "file_path": str(_PROJECT_ROOT / folder / file_name),
            "design_gold": design,
            "source": folder,
            "journal_year": journal_y,
        })
    return out


# ===========================================================================
# 2. Cached resource loaders
# ===========================================================================
@st.cache_resource(show_spinner=False)
def get_query_engine() -> Optional[QueryEngine]:
    try:
        cfg = load_config()
        if not cfg.effective_openai_api_key():
            return None
        set_rerank_enabled_default(False)
        return QueryEngine.from_config(cfg)
    except Exception as exc:  # pragma: no cover
        st.error(f"QueryEngine 初始化失败: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_pdf_text(file_path_str: str) -> str:
    import pypdf
    reader = pypdf.PdfReader(file_path_str)
    chunks = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks)


@st.cache_data(show_spinner=False)
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> tuple:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(
        [Document(text=text, metadata={"file_name": "<demo>"})],
        show_progress=False,
    )
    return tuple(n.get_content() for n in nodes)


@st.cache_data(show_spinner=False)
def grade_paper(paper_index: int, top_k: int = 12) -> dict:
    papers = _all_papers()
    paper = papers[paper_index]
    cfg = load_config()
    full_text = load_pdf_text(paper["file_path"])
    chunks = chunk_text(full_text, cfg.chunk_size, cfg.chunk_overlap)
    used_chunks = list(chunks[: max(top_k, 1)])
    chunk_file_names = [paper["file_name"]] * len(used_chunks)

    try:
        result: GRADEResult = grade_extraction(
            used_chunks,
            cfg=cfg,
            canonical_file_name=paper["file_name"],
            chunk_file_names=chunk_file_names,
        )
        grade_dict = json.loads(result.model_dump_json())
    except Exception as exc:
        grade_dict = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "paper": paper,
        "grade": grade_dict,
        "n_chunks_total": len(chunks),
        "n_chunks_used": len(used_chunks),
    }


# ===========================================================================
# 3. New helpers — temp-collection grading
# ===========================================================================
def _grade_with_temp_collection(
    text: str,
    *,
    file_name: str,
    top_k: int = 8,
) -> dict:
    """Index ``text`` into a temp Chroma collection, then run GRADE."""
    # Always start clean — no leftover temp collections from previous runs.
    clear_temp_collection()
    cfg = load_config()
    try:
        temp_res = index_temp_document(
            text=text,
            metadata={
                "file_name": file_name,
                "source": "external_temp",
                "section_type": "primary",
            },
            file_name=file_name,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        used_chunks = temp_res.chunks[: max(top_k, 1)]
        if not used_chunks:
            return {"error": "临时 collection 中无可评分的文本（摘要可能为空）"}
        result: GRADEResult = grade_extraction(
            used_chunks,
            cfg=cfg,
            canonical_file_name=file_name,
            chunk_file_names=[file_name] * len(used_chunks),
        )
        grade_dict = json.loads(result.model_dump_json())
        return {
            "grade": grade_dict,
            "n_chunks_used": len(used_chunks),
            "collection": temp_res.collection_name,
            "file_name": file_name,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        # Always clean up the temp collection, success or failure.
        clear_temp_collection()


# ===========================================================================
# 4. Display helpers
# ===========================================================================
_BUCKET_COLOR = {
    "高":   "#28a745",
    "中":   "#ffc107",
    "低":   "#fd7e14",
    "极低": "#dc3545",
}


def _bucket_color_for(grade_str: str) -> str:
    return _BUCKET_COLOR.get(grade_str, "#6c757d")


def _render_score_pill(score: float, grade: str) -> str:
    color = _bucket_color_for(grade)
    return (
        f'<span style="background:{color};color:white;padding:6px 14px;'
        f'border-radius:18px;font-weight:600;font-size:18px;">'
        f'{score:.1f} / 10</span>'
    )


def _render_grade_card(grade: dict, paper_meta: Optional[dict] = None):
    """Render the canonical GRADE-result card. ``paper_meta`` is shown only
    when present (otherwise Tab 2/3 may pass a synthetic metadata dict)."""
    overall = grade.get("overall_grade", "—")
    color = _bucket_color_for(overall)
    score = grade.get("composite_score", 0.0)
    design = grade.get("study_design", "—")
    n = grade.get("sample_size", 0)
    effect = grade.get("effect_size", "—")
    pmid_or_path = (paper_meta or {}).get("file_name", "—")
    source_label = (paper_meta or {}).get("display") or (paper_meta or {}).get("source", "—")

    st.markdown(
        f"""
        <div style="border-left:6px solid {color}; padding:12px 16px;
                    background:#f8f9fb; border-radius:6px; margin:8px 0 12px 0;">
          <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
            <span style="background:{color};color:white;padding:6px 14px;
                         border-radius:18px;font-weight:600;font-size:18px;">
              {score:.1f} / 10
            </span>
            <span style="font-size:18px;font-weight:700;color:{color};">
              证据等级: {overall}
            </span>
            <span style="background:#e8eef7;color:#1e408b;padding:4px 12px;
                         border-radius:14px;font-size:13px;">
              研究设计: {design}
            </span>
            <span style="background:#e8eef7;color:#1e408b;padding:4px 17px;
                         border-radius:14px;font-size:13px;">
              N = {n}
            </span>
            <span style="background:#e8eef7;color:#1e408b;padding:4px 12px;
                         border-radius:14px;font-size:13px;">
              效应量: {effect}
            </span>
          </div>
          <div style="margin-top:6px;font-size:12px;color:#6b7280;">
            来源: <code style="background:#eef;padding:1px 4px;border-radius:3px">{pmid_or_path}</code>
            &nbsp;·&nbsp; 通道: {source_label}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# 5. Session-state defaults
# ===========================================================================
def _init_session():
    defaults = {
        "messages": [],
        "scored_count": 0,
        "score_sum": 0.0,
        "active_tab": "tab1",          # 'tab1' | 'tab2' | 'tab3'
        "selected_paper": "EV-302",
        "pubmed_results": [],            # list[PubMedHit]
        "pubmed_query": "",
        "temp_grade": None,              # last Tab2/3 grade dict
        "temp_source": None,             # last Tab2/3 source label
        "temp_chunks_used": 0,
        "temp_collection": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ===========================================================================
# 6. Sidebar
# ===========================================================================
papers = _all_papers()
paper_display_to_idx = {p["display"]: i for i, p in enumerate(papers)}

with st.sidebar:
    st.markdown("## 🧬 OncoImmunoRAG")
    st.caption("v3 · 三 Tab 版（库内 / PubMed / 手动）")

    # Tab selector (mirrors the st.tabs choice)
    tab_labels = {
        "tab1": "📚 库内文献",
        "tab2": "🔍 PubMed 检索",
        "tab3": "📥 手动 / PDF",
    }
    sel = st.radio(
        "工作面板",
        options=list(tab_labels.keys()),
        format_func=lambda k: tab_labels[k],
        index=list(tab_labels.keys()).index(st.session_state.active_tab),
        help="切换 3 个工作面板",
    )
    st.session_state.active_tab = sel

    st.divider()

    # Tab-1 specific: paper dropdown lives in the sidebar so it survives
    # across reruns triggered by other tabs.
    if sel == "tab1":
        display_names = [p["display"] for p in papers]
        try:
            cur_idx = display_names.index(st.session_state.selected_paper)
        except ValueError:
            cur_idx = 0
        chosen = st.selectbox(
            "🎯 目标文献（库内）",
            options=display_names,
            index=cur_idx,
            help="22 篇入库文献中选择一篇进行评估。",
        )
        st.session_state.selected_paper = chosen
        p = papers[paper_display_to_idx[chosen]]
        st.markdown(
            f"""
            <div style="background:#f0f4fa; padding:10px 12px; border-radius:6px;
                        font-size:12px; line-height:1.7;">
              <b>期刊</b>: {p["journal_year"]}<br>
              <b>金标准设计</b>: {p["design_gold"]}<br>
              <b>来源</b>: {p["source"]}<br>
              <b>文件名</b>: <code style="font-size:10px">{p["file_name"]}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # System config status panel
    st.markdown("### ⚙️ 系统配置状态")
    cfg_data = [
        ("目标文献优先路由", "🟢 ON", "#d4edda"),
        ("Rerank 证据加权层", "🔴 OFF", "#f8d7da"),
        ("量化评分 10 分制", "🟢 ON", "#d4edda"),
        ("RCT-floor 豁免", "🟢 ON", "#d4edda"),
        ("注册表强制覆写", "🟢 ON", "#d4edda"),
        ("PubMed E-utilities", "🟢 ON (NCBI)", "#d4edda"),
        ("临时 collection 隔离", "🟢 ON (UUID)", "#d4edda"),
    ]
    for label, status, bg in cfg_data:
        st.markdown(
            f'<div style="background:{bg}; padding:4px 8px; border-radius:4px;'
            f' margin:2px 0; font-size:13px;">'
            f'<b>{label}</b>: {status}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    avg = (
        st.session_state.score_sum / st.session_state.scored_count
        if st.session_state.scored_count else 0.0
    )
    st.markdown("### 📊 本次会话统计")
    st.metric("已评估文献数", st.session_state.scored_count)
    st.metric("累计平均得分", f"{avg:.2f} / 10")
    if st.button("🗑 清空对话 / 临时 collection", use_container_width=True):
        st.session_state.messages = []
        st.session_state.scored_count = 0
        st.session_state.score_sum = 0.0
        st.session_state.pubmed_results = []
        st.session_state.temp_grade = None
        st.session_state.temp_source = None
        clear_temp_collection()
        st.rerun()


# ===========================================================================
# 7. Page header
# ===========================================================================
st.markdown(
    "<h1 style='margin-bottom:0'>🧬 尿路上皮癌免疫治疗证据智能分级系统</h1>"
    "<p style='color:#555; font-size:15px; margin-top:0'>"
    "RAG · 10 分制量化 GRADE · "
    "<span style='color:#28a745'>RCT-floor 豁免</span> · "
    "<span style='color:#1e408b'>22 篇库内</span> · "
    "<span style='color:#c0392b'>+ PubMed 接入 · 临时 collection 隔离</span>"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()


# ===========================================================================
# 8. Three tabs
# ===========================================================================
tab1, tab2, tab3 = st.tabs([
    "📚 Tab 1 · 库内文献分级",
    "🔍 Tab 2 · PubMed 检索与分级",
    "📥 Tab 3 · 手动输入 / PDF",
])


# ===========================================================================
# TAB 1 — 库内文献分级
# ===========================================================================
with tab1:
    st.markdown("#### 📚 库内文献分级（22 篇 PDF 池）")
    st.caption(
        "直接读取本地 PDF → 切块 → LLM 抽取 → 确定性 Python 规则引擎 → 10 分制评分卡。"
    )

    # Demo buttons
    demo_buttons = [
        "评估 BOND-003 的 GRADE 等级",
        "评估 KEYNOTE-361 的 GRADE 等级",
        "评估 JAVELIN Bladder 100 的 GRADE 等级",
    ]
    cols = st.columns(len(demo_buttons))
    for col, q in zip(cols, demo_buttons):
        with col:
            if st.button(q, use_container_width=True, key=f"tab1_demo_{q}"):
                st.session_state.messages.append({"role": "user", "content": q, "tab": "tab1"})
                st.rerun()

    # Chat history for this tab
    for msg in st.session_state.messages:
        if msg.get("tab") != "tab1":
            continue
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                for blk in msg["blocks"]:
                    kind = blk.get("kind")
                    if kind == "text":
                        st.markdown(blk["text"])
                    elif kind == "grade_card":
                        _render_grade_card(blk["grade"], blk["paper"])
                        if blk["grade"].get("grade_reasoning"):
                            with st.expander("🧠 GRADE reasoning / score_breakdown"):
                                st.markdown("**grade_reasoning:**")
                                st.code(blk["grade"]["grade_reasoning"], language="text")
                                if blk["grade"].get("score_breakdown"):
                                    st.markdown("**score_breakdown:**")
                                    st.code(blk["grade"]["score_breakdown"], language="text")
                    elif kind == "error":
                        st.error(blk["text"])
            else:
                st.markdown(msg["content"])

    prompt = st.chat_input(
        placeholder="📄 选定文献后, 提问如 '评估这篇的 GRADE 等级' / '讨论部分有哪些外部研究'",
        key="tab1_input",
    )
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "tab": "tab1"})
        st.rerun()

    # Process the last user message for this tab
    tab1_msgs = [m for m in st.session_state.messages if m.get("tab") == "tab1"]
    if tab1_msgs and tab1_msgs[-1]["role"] == "user":
        idx = paper_display_to_idx[st.session_state.selected_paper]
        paper = papers[idx]
        with st.spinner(
            f"📄 正在评估 {paper['display']} · 读取 PDF → 切块 → "
            f"LLM 抽取 → 量化 GRADE 评分 ..."
        ):
            payload = grade_paper(idx, top_k=12)
        grade = payload["grade"]
        paper_meta = payload["paper"]

        if "error" in grade:
            blocks = [{"kind": "error", "text": "LLM 调用失败: " + grade["error"]}]
        else:
            score = grade.get("composite_score", 0.0)
            if score > 0:
                st.session_state.scored_count += 1
                st.session_state.score_sum += float(score)

            text_md = (
                f"### 📄 {paper_meta['display']} — GRADE 评估结果\n\n"
                f"已对 **{paper_meta['display']}** (期刊: {paper_meta['journal_year']}) "
                f"完成量化 GRADE 评分。"
            )
            blocks = [
                {"kind": "text", "text": text_md},
                {
                    "kind": "grade_card",
                    "grade": grade,
                    "paper": paper_meta,
                },
            ]
        st.session_state.messages.append(
            {"role": "assistant", "blocks": blocks, "tab": "tab1"}
        )
        st.rerun()


# ===========================================================================
# TAB 2 — PubMed 检索与分级
# ===========================================================================
with tab2:
    st.markdown("#### 🔍 PubMed 检索与分级")
    st.caption(
        "通过 NCBI E-utilities 检索 PubMed → 选择文献 → 拉取摘要 → 写入临时 collection → GRADE 评分。"
        "**临时 collection 与主知识库物理隔离。**"
    )

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            "临床问题 / PubMed 检索词",
            value=st.session_state.pubmed_query,
            placeholder="例: PD-1 inhibitor BCG non-respond NMIBC urothelial",
            key="tab2_query_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        do_search = st.button("🔍 检索 PubMed", use_container_width=True, type="primary")

    if do_search:
        st.session_state.pubmed_query = query
        if not query.strip():
            st.warning("请先输入检索词")
        else:
            with st.spinner("🌐 正在检索 NCBI PubMed ..."):
                try:
                    hits = search_pubmed(query, max_results=10)
                except Exception as exc:
                    hits = []
                    st.error(f"PubMed 检索失败: {exc}")
            st.session_state.pubmed_results = hits
            if not hits:
                st.info("未命中任何文献，请换检索词或放宽时间范围。")

    # Results list
    hits = st.session_state.pubmed_results
    if hits:
        st.markdown(f"##### 命中 **{len(hits)}** 篇 (Title/Abstract · 2021-2026 · English)")
        for i, hit in enumerate(hits, start=1):
            col_info, col_btn2 = st.columns([6, 1])
            with col_info:
                st.markdown(
                    f"<div style='font-size:14px;'>"
                    f"<b>{i}. {hit.title}</b><br>"
                    f"<span style='color:#555;font-size:12px;'>"
                    f"{', '.join(hit.authors[:3])}{' et al.' if len(hit.authors) > 3 else ''} · "
                    f"<i>{hit.journal}</i> · {hit.year} · PMID: {hit.pmid}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
            with col_btn2:
                if st.button(
                    "评估该文献",
                    key=f"tab2_grade_{hit.pmid}",
                    use_container_width=True,
                ):
                    with st.spinner(f"📥 拉取 PMID {hit.pmid} 摘要并评分 ..."):
                        try:
                            detail = fetch_details(hit.pmid)
                        except Exception as exc:
                            st.error(f"efetch 失败: {exc}")
                            detail = None
                    if detail is not None:
                        if not detail.abstract or len(detail.abstract.strip()) < 100:
                            st.warning(
                                "⚠️ 摘要信息不足（< 100 字符），建议在 **Tab 3** 上传 PDF 全文。"
                            )
                        with st.spinner(
                            "🧮 写入临时 collection → 评分 ..."
                        ):
                            file_name = f"PMID_{detail.pmid}.pdf"
                            res = _grade_with_temp_collection(
                                text=detail.abstract,
                                file_name=file_name,
                                top_k=8,
                            )
                        st.session_state.temp_grade = res
                        st.session_state.temp_source = {
                            "display": f"{hit.short_label()} (PMID {hit.pmid})",
                            "file_name": file_name,
                            "pmid": hit.pmid,
                            "journal": hit.journal,
                            "year": hit.year,
                            "title": hit.title,
                            "source": "PubMed / 临时 collection",
                        }
                        st.session_state.temp_chunks_used = res.get("n_chunks_used", 0)
                        st.session_state.temp_collection = res.get("collection", "")
                        # Persist in messages so it shows in chat history
                        score = (res.get("grade") or {}).get("composite_score", 0.0)
                        if score and score > 0:
                            st.session_state.scored_count += 1
                            st.session_state.score_sum += float(score)
                        st.rerun()

    # Score-card panel for the latest temp assessment
    if st.session_state.temp_grade:
        st.divider()
        grade = st.session_state.temp_grade.get("grade", {})
        if "error" in st.session_state.temp_grade:
            st.error("评分失败: " + st.session_state.temp_grade["error"])
        elif grade:
            st.markdown(
                f"##### 🧮 评分结果 · 临时 collection "
                f"(`{st.session_state.temp_collection}` · "
                f"使用 **{st.session_state.temp_chunks_used}** 个临时 chunk)"
            )
            _render_grade_card(grade, st.session_state.temp_source)
            if grade.get("grade_reasoning"):
                with st.expander("🧠 GRADE reasoning / score_breakdown"):
                    st.markdown("**grade_reasoning:**")
                    st.code(grade["grade_reasoning"], language="text")
                    if grade.get("score_breakdown"):
                        st.markdown("**score_breakdown:**")
                        st.code(grade["score_breakdown"], language="text")


# ===========================================================================
# TAB 3 — 手动输入 / PDF 上传
# ===========================================================================
with tab3:
    st.markdown("#### 📥 手动输入 / PDF 上传")
    st.caption(
        "直接输入 PMID（自动从 PubMed 拉取）或上传 PDF。**两者均写入临时 collection，"
        "不污染主知识库。**"
    )

    subtab_a, subtab_b = st.tabs(["🔢 输入 PMID", "📎 上传 PDF"])

    # --- 3a. PMID lookup ---
    with subtab_a:
        with st.form("tab3_pmid_form", clear_on_submit=False):
            pmid = st.text_input(
                "PubMed ID",
                placeholder="例: 37174581",
                key="tab3_pmid_input",
            )
            submitted = st.form_submit_button(
                "📥 拉取并评分", use_container_width=True, type="primary"
            )
        if submitted:
            if not pmid.strip():
                st.warning("请先填写 PMID")
            elif not re.match(r"^\d{1,9}$", pmid.strip()):
                st.error("PMID 必须为纯数字（1-9 位）")
            else:
                with st.spinner(f"🌐 拉取 PMID {pmid} ..."):
                    try:
                        detail = fetch_details(pmid.strip())
                    except Exception as exc:
                        detail = None
                        st.error(f"efetch 失败: {exc}")
                if detail is not None:
                    if not detail.abstract or len(detail.abstract.strip()) < 100:
                        st.warning(
                            "⚠️ 摘要信息不足（< 100 字符），建议切换到 📎 上传 PDF 子页面。"
                        )
                    with st.spinner("🧮 写入临时 collection → 评分 ..."):
                        file_name = f"PMID_{detail.pmid}.pdf"
                        res = _grade_with_temp_collection(
                            text=detail.abstract,
                            file_name=file_name,
                            top_k=8,
                        )
                    st.session_state.temp_grade = res
                    st.session_state.temp_source = {
                        "display": f"PMID {detail.pmid} · {detail.journal} ({detail.year})",
                        "file_name": file_name,
                        "pmid": detail.pmid,
                        "journal": detail.journal,
                        "year": detail.year,
                        "title": detail.title,
                        "source": "PubMed / 临时 collection",
                    }
                    st.session_state.temp_chunks_used = res.get("n_chunks_used", 0)
                    st.session_state.temp_collection = res.get("collection", "")
                    score = (res.get("grade") or {}).get("composite_score", 0.0)
                    if score and score > 0:
                        st.session_state.scored_count += 1
                        st.session_state.score_sum += float(score)
                    st.rerun()

    # --- 3b. PDF upload ---
    with subtab_b:
        uploaded = st.file_uploader(
            "上传 PDF 文件",
            type=["pdf"],
            accept_multiple_files=False,
            key="tab3_pdf_upload",
        )
        if uploaded is not None:
            st.info(
                f"已选文件: **{uploaded.name}** · "
                f"{uploaded.size // 1024} KB"
            )
            if st.button(
                "📥 解析并评分", key="tab3_pdf_grade_btn", type="primary",
                use_container_width=True,
            ):
                # Save to a temp file so pypdf can open it
                with st.spinner("📄 解析 PDF → 切块 → 评分 ..."):
                    tmp_dir = Path(st.session_state.get("tab3_tmpdir") or
                                   (_PROJECT_ROOT / ".cache"))
                    tmp_dir.mkdir(exist_ok=True)
                    tmp_pdf = tmp_dir / uploaded.name
                    tmp_pdf.write_bytes(uploaded.getbuffer())
                    st.session_state.tab3_tmpdir = str(tmp_dir)

                    try:
                        import pypdf
                        reader = pypdf.PdfReader(str(tmp_pdf))
                        full_text_parts = []
                        for page in reader.pages:
                            try:
                                full_text_parts.append(page.extract_text() or "")
                            except Exception:
                                pass
                        full_text = "\n\n".join(p for p in full_text_parts if p)
                        if not full_text.strip():
                            st.error("PDF 内容为空或无法解析")
                        else:
                            file_name = uploaded.name
                            res = _grade_with_temp_collection(
                                text=full_text,
                                file_name=file_name,
                                top_k=12,
                            )
                            st.session_state.temp_grade = res
                            st.session_state.temp_source = {
                                "display": f"📎 {uploaded.name}",
                                "file_name": file_name,
                                "source": "上传 PDF / 临时 collection",
                            }
                            st.session_state.temp_chunks_used = res.get("n_chunks_used", 0)
                            st.session_state.temp_collection = res.get("collection", "")
                            score = (res.get("grade") or {}).get("composite_score", 0.0)
                            if score and score > 0:
                                st.session_state.scored_count += 1
                                st.session_state.score_sum += float(score)
                            st.rerun()
                    except Exception as exc:
                        st.error(f"PDF 解析失败: {exc}")

    # Show last temp-grade (covers BOTH sub-tabs)
    if st.session_state.temp_grade:
        st.divider()
        grade = st.session_state.temp_grade.get("grade", {})
        if "error" in st.session_state.temp_grade:
            st.error("评分失败: " + st.session_state.temp_grade["error"])
        elif grade:
            st.markdown(
                f"##### 🧮 评分结果 · 临时 collection "
                f"(`{st.session_state.temp_collection}` · "
                f"使用 **{st.session_state.temp_chunks_used}** 个临时 chunk)"
            )
            _render_grade_card(grade, st.session_state.temp_source)
            if grade.get("grade_reasoning"):
                with st.expander("🧠 GRADE reasoning / score_breakdown"):
                    st.markdown("**grade_reasoning:**")
                    st.code(grade["grade_reasoning"], language="text")
                    if grade.get("score_breakdown"):
                        st.markdown("**score_breakdown:**")
                        st.code(grade["score_breakdown"], language="text")


# ===========================================================================
# 9. Legacy global Q&A panel (preserved from v2, behind a checkbox)
# ===========================================================================
with st.expander("🌐 高级：跨论文全局问答 (旧版功能)", expanded=False):
    st.caption(
        "使用现有 QueryEngine 跨库检索并直接 LLM 综合回答。"
        "需要 OPENAI_API_KEY / LLM_MODEL 已配置。"
    )
    engine = get_query_engine()
    if engine is None:
        st.error("⚠️ 全局问答需要配置 OPENAI_API_KEY / LLM_MODEL。")
    else:
        ga_query = st.text_input(
            "跨论文问答", placeholder="例: PD-1/PD-L1 在晚期 UC 的 RCT 证据"
        )
        if st.button("🌐 全局问答", key="legacy_global_qa"):
            if not ga_query.strip():
                st.warning("请输入问题")
            else:
                with st.spinner("🔍 检索文献 · MMR 去冗 · LLM 综合回答 ..."):
                    try:
                        qr: QueryResult = engine.query(ga_query)
                    except Exception as exc:
                        qr = None
                        err = f"{type(exc).__name__}: {exc}"
                if qr is None:
                    st.error(f"检索失败: {err}")
                else:
                    st.markdown(qr.answer or "(LLM 未返回正文)")
                    if qr.sources:
                        with st.expander("📚 检索来源"):
                            for src in qr.sources:
                                p_match = next(
                                    (p for p in papers if p["file_name"] == src), None
                                )
                                if p_match:
                                    st.markdown(
                                        f"- `{src}` · **{p_match['display']}** · "
                                        f"{p_match['journal_year']}"
                                    )
                                else:
                                    st.markdown(f"- `{src}`")