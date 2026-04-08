"""
Route_Agent 实战测试脚本（项目内置断点版）
Query: The impact of AI on the job market

断点位置（均由项目内置机制处理，控制台 input() 提示）：
  BP1  planner 输出大纲后 — HumanAgent.review_plan()，可多轮修改
  BP4  每轮 reviewer 完成后 — HumanAgent.collect_review_feedback()，可追加意见或输入 stop 强制发布

输出文件（均在 outputs/<timestamp>_<query>/）：
  report.md
  route_decisions.log
  workflow.log
  operation_log.jsonl        (精简版)
  terminal_output.log        (终端完整输出)
  run the script : uv run python outputs/route_agent_test.py

"""

from __future__ import annotations

import atexit
import asyncio
import copy
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from backend.utils import create_output_session_dir
from gpt_researcher.utils.enum import Tone
from multi_agents.agents import ChiefEditorAgent
from multi_agents.route_agent import RoutedLLMInvoker, set_global_invoker
# from multi_agents.route_agent import RoutedLLMInvoker, run_live_preflight, set_global_invoker

# ── 输出目录 ─────────────────────────────────────────────────────────────────
QUERY = "The impact of AI on the job market"
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
OUT_DIR  = Path(create_output_session_dir(QUERY, base_dir=_SCRIPT_DIR, timestamp=_RUN_TS))

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

# ── operation_log ────────────────────────────────────────────────────────────
_OP_LOG = open(OUT_DIR / "operation_log.jsonl", "a", encoding="utf-8")
atexit.register(_OP_LOG.close)

def _write_op_log(event: Dict[str, Any]) -> None:
    slim = {k: v for k, v in event.items() if k != "candidates"}
    if "candidates" in event:
        top3 = sorted(event["candidates"], key=lambda c: c.get("rank", 99))[:3]
        slim["top_candidates"] = [
            f"{c['model_id']}  dim={c['dimension_score']:.2f}  cost={c['cost_score']:.4f}"
            for c in top3
        ]
    _OP_LOG.write(json.dumps(slim, ensure_ascii=False) + "\n")
    _OP_LOG.flush()

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
        candidates = event.get("candidates") or []
        if candidates:
            for c in sorted(candidates, key=lambda c: c.get("rank", 99)):
                route_log.info(
                    "  CANDIDATE  rank=%-2s  %-35s  dim=%.2f  cost=%.4f  %s",
                    c.get("rank", "?"),
                    c.get("model_id", ""),
                    c.get("dimension_score") or 0,
                    c.get("cost_score") or 0,
                    "(selected)" if c.get("model_id") == event.get("selected_model_id") else "",
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
    elif etype == "execution_fallback":
        workflow_log.warning(
            "✗ FALLBACK   %-30s → %-30s  %s",
            event.get("failed_model", ""),
            event.get("next_model", ""),
            str(event.get("error", ""))[:80],
        )
        route_log.warning(
            "EXEC_FALLBACK  | failed=%-30s next=%-30s next_provider=%s error=%s",
            event.get("failed_model", ""),
            event.get("next_model", ""),
            event.get("next_provider", ""),
            str(event.get("error", ""))[:120],
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
    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy is None:
        return
    current_policy = asyncio.get_event_loop_policy()
    if isinstance(current_policy, selector_policy):
        return
    asyncio.set_event_loop_policy(selector_policy())


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def run() -> None:
    query = QUERY
    wlog("=" * 70)
    wlog("TEST START", query=query, time=datetime.now().isoformat())
    wlog("=" * 70)

    task_json = _PROJECT_ROOT / "multi_agents" / "task.json"
    with open(task_json, encoding="utf-8") as f:
        task = json.load(f)
    task["query"]                  = query
    task["include_human_feedback"] = True   # 启用项目内置断点（BP1 大纲审查、BP4 review 意见）
    task["max_sections"]           = 4
    task["publish_formats"]        = {"markdown": True, "pdf": False, "docx": False}
    task["guidelines"]             = [
        "The report MUST be written in APA format",
        "Each sub section MUST include supporting sources using hyperlinks. "
        "If none exist, erase the sub section or rewrite it to be a part of the previous section",
        "The report MUST be written in Chinese",
    ]

    chief  = ChiefEditorAgent(task, tone=Tone.Analytical, output_dir=OUT_DIR)
    agents = chief._initialize_agents()
    chief._workflow_agents = agents
    state: Dict[str, Any] = {"task": copy.deepcopy(task)}

    # ── Phase 1a: browser ────────────────────────────────────────────────────
    wlog("PHASE 1a | browser: 初始研究")
    state = await chief._run_browser(state, recorder=None)
    wlog("PHASE 1a | browser 完成", chars=len(state.get("initial_research") or ""))

    # ── Phase 1b: planner + BP1（项目内置：大纲审查 input）────────────────────
    wlog("PHASE 1b | planner + human review (BP1)")
    state = await chief._run_planner_loop(state, None, include_human_feedback=True)
    wlog("PHASE 1b | planner 完成", sections=state.get("sections") or [])

    # ── Phase 1d: researcher ─────────────────────────────────────────────────
    wlog("PHASE 1d | researcher: 并行研究所有 sections")
    state = await chief._run_researcher(state, recorder=None)
    wlog(
        "PHASE 1d | researcher 完成",
        research_items=len(state.get("research_data") or []),
        scraping_packets=len(state.get("scraping_packets") or []),
    )

    # ── Phase 1e: writer + claim + BP4（项目内置：review 意见 input）─────────
    wlog("PHASE 1e | writer + claim + review cycle (BP4)")
    state = await chief._run_writer_and_verify(state, recorder=None)

    # publisher → final report
    wlog("PHASE 1e | publisher: 输出最终报告")
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)
    report_path = OUT_DIR / "report.md"
    wlog("PHASE 1e | 最终报告已输出", file=str(report_path))

    wlog("=" * 70)
    wlog("TEST COMPLETE", output_dir=str(OUT_DIR))
    wlog("=" * 70)

    _OP_LOG.close()

    print(f"\n所有文件已保存到: {OUT_DIR}")
    print("  report.md")
    print("  route_decisions.log")
    print("  workflow.log")
    print("  operation_log.jsonl")
    print("  terminal_output.log")
    return

    # ── Phase 2: scraping 回溯（固定 section 0）──────────────────────────────
    details  = state.get("section_details") or []
    sections = state.get("sections") or []
    scraping_idx = 0
    if scraping_idx < len(details) and isinstance(details[scraping_idx], dict):
        scraping_header = str(details[scraping_idx].get("header") or "Section 1")
    elif scraping_idx < len(sections):
        scraping_header = sections[scraping_idx]
    else:
        scraping_header = "Section 1"

    scraping_key  = _make_section_key(scraping_idx, scraping_header)
    scraping_note = (
        f"该 section「{scraping_header}」内容不够全面：缺少具体行业数据和案例支撑，"
        f"请重新抓取更多来源，补充 2024-2025 年的最新统计数据和典型企业案例。"
    )
    wlog("PHASE 2 | scraping 回溯 ×2", target=scraping_header, key=scraping_key)

    for run_i in range(1, 3):
        wlog(f"PHASE 2 | scraping 回溯第 {run_i} 次", section=scraping_header)
        state = await chief._run_researcher(
            state, recorder=None,
            note=scraping_note,
            selected_section_key=scraping_key,
            section_start_node="scraping",
        )
        wlog(f"PHASE 2 | scraping 回溯第 {run_i} 次完成")

    # scraping 回溯后重跑 writer + claim + review cycle（含 BP4）
    wlog("PHASE 2 | scraping 回溯后重新生成报告")
    state = await chief._run_writer_and_verify(state, recorder=None)
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)

    safe_header = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", scraping_header.lower()).strip("_")[:30]
    _save_report(state, f"report_v2_after_scraping_rerun_{safe_header}.md", "v2 scraping 回溯后")

    # ── Phase 3a: 保存质疑前版本 ─────────────────────────────────────────────
    _save_report(state, "report_v3_before_logic_challenge.md", "v3 整体逻辑质疑前")

    # ── Phase 3b: 整体逻辑质疑（自动注入，含 BP4）────────────────────────────
    title            = state.get("title") or "本报告"
    sections_list    = state.get("sections") or []
    section_list_str = "、".join(f"「{s}」" for s in sections_list[:4]) if sections_list else "各章节"

    logic_challenge = textwrap.dedent(f"""
        对《{title}》的整体逻辑提出质疑：

        1. 因果链不清晰：报告在 {section_list_str} 中多次断言 AI 导致失业，
           但未区分「技术性失业」与「结构性转型」，请补充论证逻辑。

        2. 反驳视角缺失：报告缺少对「AI 创造新岗位」这一主流反驳观点的系统性回应，
           请在结论部分增加平衡性分析。

        3. 数据时效性：引用数据集中在 2022-2023 年，请核查并更新至 2024-2025 年的
           最新就业统计数据，尤其是制造业与服务业的分类数据。
    """).strip()

    wlog("PHASE 3 | 注入整体逻辑质疑", length=len(logic_challenge))
    state["review_iterations"] = 0
    state = await chief._run_review_cycle(state, recorder=None, reviewer_note=logic_challenge)

    # ── Phase 4: publisher 最终输出 ──────────────────────────────────────────
    wlog("PHASE 4 | publisher: 输出最终报告")
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)
    _save_report(state, "report_v4_after_logic_revision.md", "v4 逻辑修订后（最终版）")

    wlog("=" * 70)
    wlog("TEST COMPLETE", output_dir=str(OUT_DIR))
    wlog("=" * 70)

    _OP_LOG.close()

    print(f"\n所有文件已保存到: {OUT_DIR}")
    print("  report_v1_before_scraping_rerun.md")
    print(f"  report_v2_after_scraping_rerun_{safe_header}.md")
    print("  report_v3_before_logic_challenge.md")
    print("  report_v4_after_logic_revision.md")
    print("  route_decisions.log")
    print("  workflow.log")
    print("  operation_log.jsonl")
    print("  terminal_output.log")


if __name__ == "__main__":
    _configure_windows_event_loop_policy()
    asyncio.run(run())
