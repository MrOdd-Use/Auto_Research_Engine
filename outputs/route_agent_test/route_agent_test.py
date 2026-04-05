"""
Route_Agent 实战测试脚本（含人工断点版）
Query: AI对就业市场的影响

断点位置：
  BP1  planner 输出大纲后 — 人工审查/修改 section 列表
  BP2  researcher 全部完成后 — 结构化查看所有检索文本
  BP3  writer + claim 完成后 — 审查草稿与 Claim 核验报告
  BP4  每轮 reviewer 和 reviser 完成后 — 追加意见或放行
  BP5  最终 publisher 输出后 — 查看完整最终文档

输出文件（均在 outputs/route_agent_test/<timestamp>/）：
  report_v1_before_scrap_rerun.md
  report_v2_after_scrap_rerun_{section}.md
  report_v3_before_logic_challenge.md
  report_v4_after_logic_revision.md
  route_decisions.log
  workflow.log
  operation_log.jsonl        (精简版)
  operation_log_full.jsonl   (完整版)
  terminal_output.log        (终端完整输出)
"""

from __future__ import annotations

import atexit
import asyncio
import copy
import json
import logging
import os
import random
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from gpt_researcher.utils.enum import Tone
from multi_agents.agents import ChiefEditorAgent
from multi_agents.route_agent import RoutedLLMInvoker, run_live_preflight, set_global_invoker

# ── 输出目录 ─────────────────────────────────────────────────────────────────
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
OUT_DIR  = _PROJECT_ROOT / "outputs" / "route_agent_test" / _RUN_TS
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTE_LOG    = OUT_DIR / "route_decisions.log"
WORKFLOW_LOG = OUT_DIR / "workflow.log"

# ── 终端输出同步写入文件 ──────────────────────────────────────────────────────
_TERMINAL_LOG = OUT_DIR / "terminal_output.log"

class _TeeWriter:
    """同时写入原始 stream 和文件，去除 ANSI 色彩码。"""
    def __init__(self, stream, filepath: Path) -> None:
        self._stream = stream
        self._file = open(filepath, "a", encoding="utf-8", errors="replace")
        atexit.register(self._file.close)

    def write(self, data: str) -> int:
        self._stream.write(data)
        clean = re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", data)
        self._file.write(clean)
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)

sys.stdout = _TeeWriter(sys.stdout, _TERMINAL_LOG)
sys.stderr = _TeeWriter(sys.stderr, _TERMINAL_LOG)

# ── 日志器 ───────────────────────────────────────────────────────────────────
def _make_file_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger

route_log    = _make_file_logger(ROUTE_LOG,    "route_decisions")
workflow_log = _make_file_logger(WORKFLOW_LOG, "workflow")

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s [WORKFLOW] %(message)s", datefmt="%H:%M:%S"))
workflow_log.addHandler(_console)

# ── operation_log ────────────────────────────────────────────────────────────
_OP_LOG_SLIM = open(OUT_DIR / "operation_log.jsonl",      "a", encoding="utf-8")
_OP_LOG_FULL = open(OUT_DIR / "operation_log_full.jsonl", "a", encoding="utf-8")
atexit.register(_OP_LOG_SLIM.close)
atexit.register(_OP_LOG_FULL.close)

def _write_op_log(event: Dict[str, Any]) -> None:
    _OP_LOG_FULL.write(json.dumps(event, ensure_ascii=False) + "\n")
    _OP_LOG_FULL.flush()
    slim = {k: v for k, v in event.items() if k != "candidates"}
    if "candidates" in event:
        top3 = sorted(event["candidates"], key=lambda c: c.get("rank", 99))[:3]
        slim["top_candidates"] = [
            f"{c['model_id']}  dim={c['dimension_score']:.2f}  cost={c['cost_score']:.4f}"
            for c in top3
        ]
    _OP_LOG_SLIM.write(json.dumps(slim, ensure_ascii=False) + "\n")
    _OP_LOG_SLIM.flush()

def _route_event_logger(event: Dict[str, Any]) -> None:
    etype = event.get("type", "")
    if etype == "route_decision":
        workflow_log.info(
            "→ ROUTE  %-20s  %-35s  [%s]  %.0fms",
            event.get("agent_role", ""),
            event.get("selected_model_id", ""),
            event.get("routing_reason", ""),
            event.get("route_latency_ms") or 0,
        )
        route_log.info(
            "ROUTE_DECISION | agent_role=%-20s stage=%-25s "
            "requested=%-30s selected=%-30s reason=%s | "
            "total_ms=%.1f  analysis_ms=%.1f  registry_ms=%.1f  selection_ms=%.1f",
            event.get("agent_role", ""),
            event.get("stage_name", ""),
            event.get("requested_model", ""),
            event.get("selected_model_id", ""),
            event.get("routing_reason", ""),
            event.get("route_latency_ms") or 0,
            event.get("analysis_latency_ms") or 0,
            event.get("registry_latency_ms") or 0,
            event.get("selection_latency_ms") or 0,
        )
    elif etype == "execution_end":
        status = event.get("status", "")
        if status == "failed":
            workflow_log.warning(
                "✗ EXEC_FAIL %-20s  %-35s  %.0fms  %s",
                event.get("agent_role", ""),
                event.get("selected_model_id", ""),
                event.get("execution_latency_ms") or 0,
                str(event.get("error", ""))[:100],
            )
        route_log.info(
            "EXEC_END       | agent_role=%-20s stage=%-25s "
            "model=%-30s status=%s exec_ms=%.1f%s",
            event.get("agent_role", ""),
            event.get("stage_name", ""),
            event.get("selected_model_id", ""),
            status,
            event.get("execution_latency_ms") or 0,
            f" error={event['error']}" if status == "failed" else "",
        )
    elif etype == "quota_fallback":
        workflow_log.warning(
            "⚠ FALLBACK  %-30s → %-30s  %s",
            event.get("failed_model", ""),
            event.get("next_model", ""),
            str(event.get("error", ""))[:80],
        )
        route_log.warning(
            "QUOTA_FALLBACK | failed=%-30s next=%-30s error=%s",
            event.get("failed_model", ""),
            event.get("next_model", ""),
            str(event.get("error", ""))[:120],
        )
    elif etype == "startup_preflight":
        workflow_log.info(
            "PRECHECK     checked=%-3s reachable=%-3s skipped=%-3s filtered=%-3s",
            event.get("checked_models", 0),
            event.get("ok_count", 0),
            event.get("skipped_count", 0),
            len(event.get("filtered_models") or []),
        )
        route_log.info(
            "STARTUP_PREFLIGHT | backend=%s checked=%s ok=%s skipped=%s filtered=%s",
            event.get("backend", ""),
            event.get("checked_models", 0),
            event.get("ok_count", 0),
            event.get("skipped_count", 0),
            len(event.get("filtered_models") or []),
        )
    elif etype == "execution_escalation":
        workflow_log.warning(
            "ESCALATE     %-30s -> %-30s  [%s]  %s",
            event.get("failed_model", ""),
            event.get("next_model", ""),
            event.get("kind", ""),
            str(event.get("error", ""))[:80],
        )
        route_log.warning(
            "EXEC_ESCALATION | kind=%s failed=%-30s next=%-30s error=%s",
            event.get("kind", ""),
            event.get("failed_model", ""),
            event.get("next_model", ""),
            str(event.get("error", ""))[:120],
        )
    _write_op_log(event)

set_global_invoker(RoutedLLMInvoker(event_logger=_route_event_logger))

# ── Route_Agent 子系统日志转发（WARNING+）→ workflow_log + 终端 ─────────────────
_ra_fh = logging.FileHandler(WORKFLOW_LOG, encoding="utf-8")
_ra_fh.setLevel(logging.WARNING)
_ra_fh.setFormatter(logging.Formatter(
    "%(asctime)s  [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
_ra_con = logging.StreamHandler(sys.stdout)
_ra_con.setLevel(logging.WARNING)
_ra_con.setFormatter(logging.Formatter(
    "%(asctime)s [WARN] %(name)s | %(message)s", datefmt="%H:%M:%S"
))
_ra_root = logging.getLogger("route_agent")
_ra_root.setLevel(logging.WARNING)
_ra_root.addHandler(_ra_fh)
_ra_root.addHandler(_ra_con)

# ── 工作流日志辅助 ────────────────────────────────────────────────────────────
import time as _time
_T0: float = _time.perf_counter()

def wlog(msg: str, **kw: Any) -> None:
    elapsed = round(_time.perf_counter() - _T0, 1)
    kw_str = "  " + "  ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
    workflow_log.info("[+%6.1fs] %s%s", elapsed, msg, kw_str)


def _configure_windows_event_loop_policy() -> None:
    """Use the selector loop on Windows to avoid Proactor ready-queue crashes."""
    if os.name != "nt":
        return
    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy is None:
        return
    current_policy = asyncio.get_event_loop_policy()
    if isinstance(current_policy, selector_policy):
        return
    asyncio.set_event_loop_policy(selector_policy())


async def _run_startup_preflight() -> Dict[str, Any]:
    """Run one explicit Route_Agent live preflight before the workflow starts."""
    raw_limit = str(os.getenv("ROUTE_AGENT_PREFLIGHT_MAX_MODELS") or "").strip()
    live_probe_limit = int(raw_limit) if raw_limit.isdigit() else None
    report = await asyncio.to_thread(
        run_live_preflight,
        output_path=OUT_DIR / "live_preflight.json",
        include_live_model_probe=True,
        live_probe_limit=live_probe_limit,
    )
    route_agent_status = str(report.get("route_agent", {}).get("status") or "")
    live_probe = report.get("route_agent", {}).get("live_probe", {})
    reachable = int(report.get("route_agent", {}).get("reachable_model_count") or 0)
    wlog(
        "PREFLIGHT COMPLETE",
        status=route_agent_status,
        checked=live_probe.get("checked_models", 0),
        reachable=reachable,
        filtered=live_probe.get("filtered_count", 0),
        report=str(OUT_DIR / "live_preflight.json"),
    )
    if route_agent_status == "error":
        raise RuntimeError(
            str(report.get("route_agent", {}).get("message") or "Route_Agent startup preflight failed")
        )
    return report

def _save_report(state: Dict[str, Any], filename: str, label: str) -> None:
    report = state.get("report") or state.get("final_draft") or ""
    if isinstance(report, dict):
        report = json.dumps(report, ensure_ascii=False, indent=2)
    path = OUT_DIR / filename
    path.write_text(str(report), encoding="utf-8")
    wlog(f"SAVED {label}", file=str(path), chars=len(str(report)))

def _make_section_key(index: int, header: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(header or "").strip().lower()).strip("_")
    return f"section_{index}_{cleaned[:48] or 'section'}"

# ── 断点辅助 ──────────────────────────────────────────────────────────────────
_BP_SEP = "═" * 72

async def bp(label: str, content: str, *, prompt_msg: str = "按 Enter 继续，或输入意见：") -> str:
    """异步断点：打印内容，等待用户输入，返回输入文本（空字符串表示直接放行）。"""
    print(f"\n{_BP_SEP}")
    print(f"  [BREAKPOINT] {label}")
    print(_BP_SEP)
    print(content)
    print("─" * 72)
    response = await asyncio.to_thread(input, f"{prompt_msg}\n> ")
    return response.strip()

# ── 断点内容格式化 ────────────────────────────────────────────────────────────

def fmt_outline(state: Dict[str, Any]) -> str:
    """BP1：格式化大纲展示。"""
    details  = state.get("section_details") or []
    sections = state.get("sections") or []
    lines    = [f"共 {len(details) or len(sections)} 个 section：\n"]
    for i, d in enumerate(details):
        header = d.get("header") if isinstance(d, dict) else (sections[i] if i < len(sections) else f"Section {i+1}")
        desc   = str(d.get("description") or "") if isinstance(d, dict) else ""
        queries = (d.get("queries") or d.get("research_queries") or []) if isinstance(d, dict) else []
        lines.append(f"  [{i+1}] {header}")
        if desc.strip():
            lines.append(f"       描述: {textwrap.shorten(desc.strip(), 120)}")
        if queries:
            qs = queries if isinstance(queries, list) else [queries]
            lines.append(f"       查询: {' | '.join(str(q) for q in qs[:3])}")
        lines.append("")
    if not details and sections:
        for i, s in enumerate(sections):
            lines.append(f"  [{i+1}] {s}")
    return "\n".join(lines)


def fmt_research(state: Dict[str, Any]) -> str:
    """BP2：按 section 结构化展示所有检索文本。"""
    research_data   = state.get("research_data")  or []
    scrap_packets   = state.get("scrap_packets")   or []
    section_details = state.get("section_details") or []
    sections        = state.get("sections")        or []

    def _header(i: int) -> str:
        if i < len(section_details) and isinstance(section_details[i], dict):
            return str(section_details[i].get("header") or f"Section {i+1}")
        return sections[i] if i < len(sections) else f"Section {i+1}"

    parts: List[str] = [f"共 {len(research_data)} 个 section 检索结果\n"]
    for i, draft in enumerate(research_data):
        title = _header(i)
        parts.append(f"{'━'*60}")
        parts.append(f"  Section {i+1}: {title}")
        parts.append(f"{'━'*60}")

        if isinstance(draft, dict):
            for k, v in draft.items():
                text = str(v or "").strip()
                parts.append(f"[{k}]\n{textwrap.shorten(text, 1200, placeholder=' …（已截断）')}")
        else:
            text = str(draft or "").strip()
            parts.append(textwrap.shorten(text, 1200, placeholder=" …（已截断）"))

        pkt = scrap_packets[i] if i < len(scrap_packets) else None
        if pkt and isinstance(pkt, dict):
            log     = pkt.get("search_log") or []
            sources = pkt.get("sources") or pkt.get("source_urls") or []
            parts.append(f"\n  来源数: {len(sources)}  |  搜索轮次: {len(log)}")
            for s in list(sources)[:5]:
                url = s.get("url") if isinstance(s, dict) else str(s)
                parts.append(f"    · {url}")
        parts.append("")
    return "\n".join(parts)


def fmt_draft_claim(state: Dict[str, Any]) -> str:
    """BP3：展示 writer 草稿摘要 + claim 核验报告。"""
    draft      = str(state.get("final_draft") or state.get("report") or "").strip()
    claim      = str(state.get("claim_confidence_report") or "").strip()
    suspicious = state.get("suspicious_claims") or []

    lines = []
    lines.append("━━━ 初稿（前 1500 字）━━━")
    lines.append(draft[:1500] + (" …（已截断）" if len(draft) > 1500 else ""))
    lines.append("")
    lines.append("━━━ Claim 核验报告 ━━━")
    if claim:
        lines.append(textwrap.shorten(claim, 800, placeholder=" …（已截断）"))
    else:
        lines.append("（无 claim_confidence_report）")
    if suspicious:
        lines.append(f"\n⚠️  可疑 claim 数：{len(suspicious)}")
        for c in suspicious[:3]:
            lines.append(f"  · {str(c)[:120]}")
    reflexion = state.get("claim_reflexion_iterations", 0)
    if reflexion:
        lines.append(f"\nClaim 反思迭代次数：{reflexion}")
    return "\n".join(lines)


def fmt_reviewer(state: Dict[str, Any], iteration: int) -> str:
    """BP4 reviewer：展示 review 意见。"""
    review = str(state.get("review") or "").strip()
    lines  = [f"轮次 {iteration} — Reviewer 意见：\n"]
    lines.append(review if review else "（无意见，Reviewer 认为无需修改）")
    return "\n".join(lines)


def fmt_reviser(state: Dict[str, Any], iteration: int) -> str:
    """BP4 reviser：展示修订说明 + 草稿前后对比。"""
    notes        = str(state.get("revision_notes") or "").strip()
    draft_after  = str(state.get("final_draft") or "").strip()
    draft_before = str(state.get("_draft_before_revision") or "").strip()

    lines = [f"轮次 {iteration} — Reviser 修订完成：\n"]
    if notes:
        lines.append(f"修订说明：\n{textwrap.shorten(notes, 600, placeholder=' …')}\n")

    lines.append("草稿变化（前 400 字对比）：")
    lines.append(f"  [修订前] {draft_before[:400].replace(chr(10), ' ')!r}")
    lines.append(f"  [修订后] {draft_after[:400].replace(chr(10), ' ')!r}")
    return "\n".join(lines)


# ── 手动 Review/Revise 循环（含 BP4）────────────────────────────────────────

async def manual_review_cycle(
    chief: ChiefEditorAgent,
    state: Dict[str, Any],
    *,
    initial_reviewer_note: str | None = None,
) -> Dict[str, Any]:
    """逐轮运行 reviewer→reviser，每轮结束后触发 BP4 供人工介入。"""
    current_node          = "reviewer"
    iteration             = 0
    pending_reviewer_note = initial_reviewer_note

    while True:
        if current_node == "reviewer":
            iteration += 1
            wlog(f"BP4 | reviewer round {iteration} start")
            state = await chief._run_global_node(
                "reviewer",
                chief._run_final_reviewer,
                state,
                None,
                note=pending_reviewer_note,
            )
            pending_reviewer_note = None

            if state.get("review") is None:
                wlog(f"BP4 | reviewer round {iteration}: 无问题，循环结束")
                await bp(
                    f"BP4 | Review Round {iteration} — Reviewer：无需修改",
                    "Reviewer 评估通过，不需要进一步修改。",
                    prompt_msg="按 Enter 继续...",
                )
                break

            reviewer_text = fmt_reviewer(state, iteration)
            feedback = await bp(
                f"BP4 | Review Round {iteration} — Reviewer 意见",
                reviewer_text,
                prompt_msg="按 Enter 继续 Revise，或输入附加意见给 Reviser：",
            )
            pending_reviewer_note = feedback or None
            current_node = "reviser"
            continue

        # reviser
        wlog(f"BP4 | reviser round {iteration} start")
        state["_draft_before_revision"] = str(state.get("final_draft") or "")
        state = await chief._run_global_node(
            "reviser",
            chief._run_final_reviser,
            state,
            None,
            note=pending_reviewer_note,
        )
        pending_reviewer_note = None

        cap_reached  = chief._is_review_cap_reached(state)
        reviser_text = fmt_reviser(state, iteration)
        if cap_reached:
            reviser_text += "\n\n⚠️  已达最大 Review 轮次上限，强制退出。"
        feedback = await bp(
            f"BP4 | Review Round {iteration} — Reviser 修订完成",
            reviser_text,
            prompt_msg="按 Enter 进入下一轮 Review，或输入附加意见给 Reviewer：",
        )
        if cap_reached:
            break
        pending_reviewer_note = feedback or None
        current_node = "reviewer"

    return state


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def run() -> None:
    query = "AI对就业市场的影响"
    wlog("=" * 70)
    wlog("TEST START", query=query, time=datetime.now().isoformat())
    wlog("=" * 70)
    await _run_startup_preflight()

    task_json = _PROJECT_ROOT / "multi_agents" / "task.json"
    with open(task_json, encoding="utf-8") as f:
        task = json.load(f)
    task["query"]                = query
    task["include_human_feedback"] = False
    task["max_sections"]         = 4
    task["publish_formats"]      = {"markdown": True, "pdf": False, "docx": False}

    chief  = ChiefEditorAgent(task, tone=Tone.Analytical, output_dir=OUT_DIR)
    agents = chief._initialize_agents()
    chief._workflow_agents = agents
    state: Dict[str, Any] = {"task": copy.deepcopy(task)}

    # ── Phase 1a: browser ────────────────────────────────────────────────────
    wlog("PHASE 1a | browser: 初始研究")
    state = await chief._run_browser(state, recorder=None)
    wlog("PHASE 1a | browser 完成", chars=len(state.get("initial_research") or ""))

    # ── Phase 1b: planner 生成初始大纲 ──────────────────────────────────────
    wlog("PHASE 1b | planner: 生成初始大纲")
    state = await chief._run_global_node(
        "planner", agents["editor"].plan_research, state, recorder=None
    )
    wlog("PHASE 1b | planner 完成", sections=state.get("sections") or [])

    # ── BP1: 大纲审查（可多轮修改）──────────────────────────────────────────
    loop_count = 0
    while True:
        outline_text = fmt_outline(state)
        feedback = await bp(
            f"BP1 | 大纲审查（第 {loop_count + 1} 次）",
            outline_text,
            prompt_msg="按 Enter 确认当前大纲；或输入修改意见（Planner 将重新规划）：",
        )
        if not feedback:
            wlog("BP1 | 大纲已确认", sections=state.get("sections") or [])
            break
        loop_count += 1
        wlog(f"BP1 | 注入修改意见，重新规划（第 {loop_count} 次）", feedback=feedback)
        state["human_feedback"] = feedback
        state = await chief._run_global_node(
            "planner", agents["editor"].plan_research, state, recorder=None
        )
        state["human_feedback"] = None

    # ── Phase 1d: researcher 并行研究所有 sections ───────────────────────────
    wlog("PHASE 1d | researcher: 并行研究所有 sections")
    state = await chief._run_researcher(state, recorder=None)
    wlog(
        "PHASE 1d | researcher 完成",
        research_items=len(state.get("research_data") or []),
        scrap_packets=len(state.get("scrap_packets") or []),
    )

    # ── BP2: 检索结果审查 ────────────────────────────────────────────────────
    await bp(
        "BP2 | 检索结果审查",
        fmt_research(state),
        prompt_msg="（仅查看）按 Enter 继续进入 Writer：",
    )

    # ── Phase 1e: writer（拆分 _run_writer_and_verify）──────────────────────
    wlog("PHASE 1e | writer: 生成初稿")
    state = chief._prepare_for_writer_pass(state)
    state = await chief._inject_source_index(state)
    state = await chief._run_global_node("writer", agents["writer"].run, state, recorder=None)

    wlog("PHASE 1e | claim_review: 核验声明")
    state = await chief._run_claim_review(state, recorder=None)

    # ── BP3: 初稿 + Claim 审查 ───────────────────────────────────────────────
    reviewer_note_extra = await bp(
        "BP3 | 初稿 + Claim 核验审查",
        fmt_draft_claim(state),
        prompt_msg="按 Enter 进入 Review 循环；或输入附加意见注入 Reviewer：",
    )

    # ── Review/Revise 循环（含 BP4）──────────────────────────────────────────
    wlog("PHASE 1e | review cycle start")
    state = await manual_review_cycle(
        chief, state, initial_reviewer_note=reviewer_note_extra or None
    )
    state = await chief._annotate_if_needed(state)

    # publisher → v1
    wlog("PHASE 1e | publisher: 输出 v1 报告")
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)
    _save_report(state, "report_v1_before_scrap_rerun.md", "v1 初稿（scrap 回溯前）")

    # ── Phase 2: scrap 回溯 ──────────────────────────────────────────────────
    details  = state.get("section_details") or []
    sections = state.get("sections") or []

    def _section_menu() -> str:
        lines = ["当前 section 列表："]
        for i, d in enumerate(details):
            h = (d.get("header") if isinstance(d, dict) else None) or (sections[i] if i < len(sections) else f"Section {i+1}")
            lines.append(f"  [{i}] {h}")
        if not details:
            for i, s in enumerate(sections):
                lines.append(f"  [{i}] {s}")
        return "\n".join(lines)

    scrap_choice = await bp(
        "BP2.5 | Phase 2 — 选择 Scrap 回溯目标 Section",
        _section_menu() + "\n\n将对选中 section 进行 2 次 scrap 回溯，补充更多来源。",
        prompt_msg="输入 section 编号（0 起）；直接按 Enter 随机选择：",
    )

    n_sections = len(details or sections)
    if scrap_choice.isdigit() and 0 <= int(scrap_choice) < n_sections:
        scrap_idx = int(scrap_choice)
    else:
        scrap_idx = random.randint(0, max(0, n_sections - 1))

    if scrap_idx < len(details) and isinstance(details[scrap_idx], dict):
        scrap_header = str(details[scrap_idx].get("header") or f"Section {scrap_idx+1}")
    elif scrap_idx < len(sections):
        scrap_header = sections[scrap_idx]
    else:
        scrap_header = f"Section {scrap_idx+1}"

    scrap_key  = _make_section_key(scrap_idx, scrap_header)
    scrap_note = (
        f"该 section「{scrap_header}」内容不够全面：缺少具体行业数据和案例支撑，"
        f"请重新抓取更多来源，补充 2024-2025 年的最新统计数据和典型企业案例。"
    )
    wlog("PHASE 2 | scrap 回溯 ×2", target=scrap_header, key=scrap_key)

    for run_i in range(1, 3):
        wlog(f"PHASE 2 | scrap 回溯第 {run_i} 次", section=scrap_header)
        state = await chief._run_researcher(
            state, recorder=None,
            note=scrap_note,
            selected_section_key=scrap_key,
            section_start_node="scrap",
        )
        wlog(f"PHASE 2 | scrap 回溯第 {run_i} 次完成")

    # scrap 回溯后重跑 writer + claim + review/revise
    wlog("PHASE 2 | scrap 回溯后重新生成报告")
    state = chief._prepare_for_writer_pass(state)
    state = await chief._inject_source_index(state)
    state = await chief._run_global_node("writer", agents["writer"].run, state, recorder=None)
    state = await chief._run_claim_review(state, recorder=None)

    reviewer_note_v2 = await bp(
        f"BP3.2 | Scrap 回溯后 — 初稿 + Claim（section: {scrap_header}）",
        fmt_draft_claim(state),
        prompt_msg="按 Enter 进入 Review 循环；或输入意见：",
    )
    state = await manual_review_cycle(chief, state, initial_reviewer_note=reviewer_note_v2 or None)
    state = await chief._annotate_if_needed(state)
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)

    safe_header = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", scrap_header.lower()).strip("_")[:30]
    _save_report(state, f"report_v2_after_scrap_rerun_{safe_header}.md", "v2 scrap 回溯后")

    # ── Phase 3a: 保存质疑前版本 ─────────────────────────────────────────────
    _save_report(state, "report_v3_before_logic_challenge.md", "v3 整体逻辑质疑前")

    # ── Phase 3b: 整体逻辑质疑 ───────────────────────────────────────────────
    title         = state.get("title") or "本报告"
    sections_list = state.get("sections") or []
    section_list_str = "、".join(f"「{s}」" for s in sections_list[:4]) if sections_list else "各章节"

    auto_challenge = textwrap.dedent(f"""
        对《{title}》的整体逻辑提出质疑：

        1. 因果链不清晰：报告在 {section_list_str} 中多次断言 AI 导致失业，
           但未区分「技术性失业」与「结构性转型」，请补充论证逻辑。

        2. 反驳视角缺失：报告缺少对「AI 创造新岗位」这一主流反驳观点的系统性回应，
           请在结论部分增加平衡性分析。

        3. 数据时效性：引用数据集中在 2022-2023 年，请核查并更新至 2024-2025 年的
           最新就业统计数据，尤其是制造业与服务业的分类数据。
    """).strip()

    challenge_input = await bp(
        "BP3.3 | Phase 3 — 整体逻辑质疑内容确认",
        f"自动生成的质疑内容如下：\n\n{auto_challenge}",
        prompt_msg="按 Enter 使用以上质疑；或输入自定义质疑内容（将完整替换）：",
    )
    logic_challenge = challenge_input if challenge_input else auto_challenge
    wlog("PHASE 3 | 注入整体逻辑质疑", length=len(logic_challenge))

    state["review"]            = logic_challenge
    state["review_iterations"] = 0
    state = await manual_review_cycle(chief, state, initial_reviewer_note=None)

    # ── Phase 4: publisher 最终输出 ──────────────────────────────────────────
    wlog("PHASE 4 | publisher: 输出最终报告")
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)
    _save_report(state, "report_v4_after_logic_revision.md", "v4 逻辑修订后（最终版）")

    # ── BP5: 最终文档 ────────────────────────────────────────────────────────
    final_text = str(state.get("report") or state.get("final_draft") or "").strip()
    await bp(
        "BP5 | 最终文档（完整版）",
        final_text,
        prompt_msg="按 Enter 结束测试：",
    )

    wlog("=" * 70)
    wlog("TEST COMPLETE", output_dir=str(OUT_DIR))
    wlog("=" * 70)

    _OP_LOG_SLIM.close()
    _OP_LOG_FULL.close()

    print(f"\n所有文件已保存到: {OUT_DIR}")
    print("  report_v1_before_scrap_rerun.md")
    print(f"  report_v2_after_scrap_rerun_{safe_header}.md")
    print("  report_v3_before_logic_challenge.md")
    print("  report_v4_after_logic_revision.md")
    print("  live_preflight.json")
    print("  route_decisions.log")
    print("  workflow.log")
    print("  operation_log.jsonl        (精简版)")
    print("  operation_log_full.jsonl   (完整版)")
    print("  terminal_output.log        (终端完整输出)")


if __name__ == "__main__":
    _configure_windows_event_loop_policy()
    asyncio.run(run())
