"""
Route_Agent 实战测试脚本
Query: AI对就业市场的影响

流程：
  Phase 1  browser → planner（随机选一个 section 注入修改意见）→ researcher → writer
  Phase 2  随机选一个 section，触发 2 次 scrap 回溯（内容不全面）
  Phase 3  对最终报告注入整体逻辑质疑，触发 reviewer → reviser 链
  Phase 4  publisher 输出最终报告

输出文件（均在 research/route_agent_test/）：
  report_v1_before_scrap_rerun.md                    初稿（scrap 回溯前）
  report_v2_after_scrap_rerun_{section}.md           scrap 回溯后（含被质疑的 section 标题）
  report_v3_before_logic_challenge.md                整体逻辑质疑前
  report_v4_after_logic_revision.md                  逻辑修订后（最终版）
  route_decisions.log                                每次 LLM 路由决策
  workflow.log                                       工作流节点事件
"""

from __future__ import annotations

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
from typing import Any, Dict

# ---------------------------------------------------------------------------
# 路径设置：从 research/route_agent_test/ 运行时，需要把项目根加入 sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # research/route_agent_test/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # Auto_Research_Engine/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from gpt_researcher.utils.enum import Tone
from multi_agents.agents import ChiefEditorAgent
from multi_agents.route_agent import RoutedLLMInvoker, set_global_invoker

# ---------------------------------------------------------------------------
# 输出目录：每次运行创建带时间戳的子文件夹
# ---------------------------------------------------------------------------
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
OUT_DIR  = _SCRIPT_DIR.parent / _RUN_TS
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTE_LOG    = OUT_DIR / "route_decisions.log"
WORKFLOW_LOG = OUT_DIR / "workflow.log"

# ---------------------------------------------------------------------------
# 日志器
# ---------------------------------------------------------------------------
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

# 同时把 workflow 事件打到控制台
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s [WORKFLOW] %(message)s", datefmt="%H:%M:%S"))
workflow_log.addHandler(_console)

# ---------------------------------------------------------------------------
# operation_log 文件句柄（精简版 + 完整版）
# ---------------------------------------------------------------------------
_OP_LOG_SLIM = open(OUT_DIR / "operation_log.jsonl",      "a", encoding="utf-8")
_OP_LOG_FULL = open(OUT_DIR / "operation_log_full.jsonl", "a", encoding="utf-8")

def _write_op_log(event: Dict[str, Any]) -> None:
    """完整版原样写入；精简版将 candidates 压缩为 top3 摘要。"""
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

# ---------------------------------------------------------------------------
# Route_Agent event_logger：写入 route_decisions.log
# ---------------------------------------------------------------------------
def _route_event_logger(event: Dict[str, Any]) -> None:
    etype = event.get("type", "")
    if etype == "route_decision":
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
        route_log.warning(
            "QUOTA_FALLBACK | failed=%-30s next=%-30s error=%s",
            event.get("failed_model", ""),
            event.get("next_model", ""),
            str(event.get("error", ""))[:120],
        )
    _write_op_log(event)

# 注入全局 invoker
set_global_invoker(RoutedLLMInvoker(event_logger=_route_event_logger))

# ---------------------------------------------------------------------------
# 工作流日志辅助
# ---------------------------------------------------------------------------
import time as _time
_T0: float = _time.perf_counter()

def wlog(msg: str, **kw: Any) -> None:
    elapsed = round(_time.perf_counter() - _T0, 1)
    kw_str = "  " + "  ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
    workflow_log.info("[+%6.1fs] %s%s", elapsed, msg, kw_str)

def _save_report(state: Dict[str, Any], filename: str, label: str) -> None:
    report = state.get("report") or state.get("final_draft") or ""
    if isinstance(report, dict):
        report = json.dumps(report, ensure_ascii=False, indent=2)
    path = OUT_DIR / filename
    path.write_text(str(report), encoding="utf-8")
    wlog(f"SAVED {label}", file=str(path), chars=len(str(report)))

# ---------------------------------------------------------------------------
# section_key 生成（与 EditorAgent._make_section_key 保持一致）
# ---------------------------------------------------------------------------
def _make_section_key(index: int, header: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(header or "").strip().lower()).strip("_")
    return f"section_{index}_{cleaned[:48] or 'section'}"

# ---------------------------------------------------------------------------
# 随机选 section
# ---------------------------------------------------------------------------
def _pick_random_section(state: Dict[str, Any]) -> tuple[int, str, str]:
    """返回 (index, header, section_key)"""
    details = state.get("section_details") or []
    if not details:
        raise RuntimeError("section_details 为空，无法选取 section")
    idx = random.randint(0, len(details) - 1)
    header = str(details[idx].get("header") or f"Section {idx + 1}")
    key = _make_section_key(idx, header)
    return idx, header, key

# ---------------------------------------------------------------------------
# 构造 human_feedback（子标题修改意见）
# ---------------------------------------------------------------------------
def _build_section_feedback(header: str) -> str:
    return (
        f"请修改子标题「{header}」的研究范围：当前标题过于宽泛，"
        f"请将其聚焦在「{header}」对蓝领制造业岗位的具体替代机制与数量影响，"
        f"并补充 2025 年后的最新数据要求。"
    )

# ---------------------------------------------------------------------------
# 构造整体逻辑质疑（reviewer_note）
# ---------------------------------------------------------------------------
def _build_logic_challenge(state: Dict[str, Any]) -> str:
    title = state.get("title") or "本报告"
    sections = state.get("sections") or []
    section_list = "、".join(f"「{s}」" for s in sections[:4]) if sections else "各章节"
    return textwrap.dedent(f"""
        对《{title}》的整体逻辑提出质疑：

        1. 因果链不清晰：报告在 {section_list} 中多次断言 AI 导致失业，
           但未区分「技术性失业」与「结构性转型」，请补充论证逻辑。

        2. 反驳视角缺失：报告缺少对「AI 创造新岗位」这一主流反驳观点的系统性回应，
           请在结论部分增加平衡性分析。

        3. 数据时效性：引用数据集中在 2022-2023 年，请核查并更新至 2024-2025 年的
           最新就业统计数据，尤其是制造业与服务业的分类数据。
    """).strip()

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def run() -> None:
    query = "AI对就业市场的影响"
    wlog("=" * 70)
    wlog("TEST START", query=query, time=datetime.now().isoformat())
    wlog("=" * 70)

    # 加载 task 配置（复用 multi_agents/main.py 的 open_task 逻辑）
    task_json = _PROJECT_ROOT / "multi_agents" / "task.json"
    with open(task_json, encoding="utf-8") as f:
        task = json.load(f)
    task["query"] = query
    task["include_human_feedback"] = False   # 由脚本手动控制 human_feedback
    task["max_sections"] = 4
    task["publish_formats"] = {"markdown": True, "pdf": False, "docx": False}

    chief = ChiefEditorAgent(task, tone=Tone.Analytical)
    agents = chief._initialize_agents()
    chief._workflow_agents = agents

    state: Dict[str, Any] = {"task": copy.deepcopy(task)}

    # -----------------------------------------------------------------------
    # Phase 1a: browser
    # -----------------------------------------------------------------------
    wlog("PHASE 1a | browser: 初始研究")
    state = await chief._run_browser(state, recorder=None)
    wlog("PHASE 1a | browser 完成", initial_research_chars=len(state.get("initial_research") or ""))

    # -----------------------------------------------------------------------
    # Phase 1b: planner（第一次，生成大纲）
    # -----------------------------------------------------------------------
    wlog("PHASE 1b | planner: 生成初始大纲")
    state = await chief._run_global_node(
        "planner", agents["editor"].plan_research, state, recorder=None
    )
    sections = state.get("sections") or []
    wlog("PHASE 1b | planner 完成", sections=sections)

    # -----------------------------------------------------------------------
    # Phase 1c: 随机选一个 section，注入 human_feedback，重跑 planner
    # -----------------------------------------------------------------------
    _, chosen_header, _ = _pick_random_section(state)
    feedback = _build_section_feedback(chosen_header)
    wlog(
        "PHASE 1c | 随机选中 section，注入修改意见",
        chosen_section=chosen_header,
        feedback=feedback,
    )

    # _inject_global_note("planner", ...) 会把 human_feedback 写入 state
    # 然后 planner 读取 human_feedback 重新规划
    state["human_feedback"] = feedback
    state_before_replanner = copy.deepcopy(state)
    state = await chief._run_global_node(
        "planner", agents["editor"].plan_research, state, recorder=None
    )
    state["human_feedback"] = None   # 清除，避免影响后续节点
    wlog(
        "PHASE 1c | planner 重规划完成",
        new_sections=state.get("sections") or [],
    )

    # -----------------------------------------------------------------------
    # Phase 1d: researcher（并行研究所有 sections）
    # -----------------------------------------------------------------------
    wlog("PHASE 1d | researcher: 并行研究所有 sections")
    state = await chief._run_researcher(state, recorder=None)
    wlog(
        "PHASE 1d | researcher 完成",
        research_items=len(state.get("research_data") or []),
        scrap_packets=len(state.get("scrap_packets") or []),
    )

    # -----------------------------------------------------------------------
    # Phase 1e: writer（生成初稿）
    # -----------------------------------------------------------------------
    wlog("PHASE 1e | writer: 生成初稿")
    state = await chief._run_writer_and_verify(state, recorder=None)
    wlog("PHASE 1e | writer 完成，进入 publisher 生成 v1 报告")

    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)
    _save_report(state, "report_v1_before_scrap_rerun.md", "v1 初稿（scrap 回溯前）")

    # -----------------------------------------------------------------------
    # Phase 2: 随机选一个 section，触发 2 次 scrap 回溯
    # -----------------------------------------------------------------------
    scrap_idx, scrap_header, scrap_key = _pick_random_section(state)
    scrap_note = (
        f"该 section「{scrap_header}」内容不够全面：缺少具体行业数据和案例支撑，"
        f"请重新抓取更多来源，补充 2024-2025 年的最新统计数据和典型企业案例。"
    )
    wlog(
        "PHASE 2 | scrap 回溯 ×2",
        target_section=scrap_header,
        section_key=scrap_key,
        note=scrap_note,
    )

    for run_i in range(1, 3):
        wlog(f"PHASE 2 | scrap 回溯第 {run_i} 次", section=scrap_header)
        state = await chief._run_researcher(
            state,
            recorder=None,
            note=scrap_note,
            selected_section_key=scrap_key,
            section_start_node="scrap",
        )
        wlog(f"PHASE 2 | scrap 回溯第 {run_i} 次完成")

    # scrap 回溯后重新跑 writer
    wlog("PHASE 2 | scrap 回溯后重新生成报告")
    state = await chief._run_writer_and_verify(state, recorder=None)
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)

    safe_header = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", scrap_header.lower()).strip("_")[:30]
    _save_report(
        state,
        f"report_v2_after_scrap_rerun_{safe_header}.md",
        f"v2 scrap 回溯后（section: {scrap_header}）",
    )

    # -----------------------------------------------------------------------
    # Phase 3a: 保存质疑前版本
    # -----------------------------------------------------------------------
    _save_report(state, "report_v3_before_logic_challenge.md", "v3 整体逻辑质疑前")

    # -----------------------------------------------------------------------
    # Phase 3b: 整体逻辑质疑，触发 reviewer → reviser 链
    # -----------------------------------------------------------------------
    logic_challenge = _build_logic_challenge(state)
    wlog("PHASE 3 | 整体逻辑质疑，注入 reviewer_note")
    wlog("PHASE 3 | 质疑内容", challenge=logic_challenge.replace("\n", " | "))

    # 注入质疑作为 reviewer_note，强制触发 reviser
    state["review"] = logic_challenge
    state["review_iterations"] = 0
    state = await chief._run_review_cycle(
        state,
        recorder=None,
        start_node="reviser",   # 直接从 reviser 开始，因为 review 已注入
    )
    wlog(
        "PHASE 3 | reviewer→reviser 链完成",
        review_iterations=state.get("review_iterations", 0),
        has_revision_notes=bool(state.get("revision_notes")),
    )

    # -----------------------------------------------------------------------
    # Phase 4: publisher 输出最终版
    # -----------------------------------------------------------------------
    wlog("PHASE 4 | publisher: 输出最终报告")
    state = await chief._run_global_node("publisher", agents["publisher"].run, state, recorder=None)
    _save_report(state, "report_v4_after_logic_revision.md", "v4 逻辑修订后（最终版）")

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
    print("  route_decisions.log")
    print("  workflow.log")
    print("  operation_log.jsonl        (精简版)")
    print("  operation_log_full.jsonl   (完整版)")


if __name__ == "__main__":
    asyncio.run(run())
