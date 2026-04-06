from typing import Optional, TypedDict


class DraftState(TypedDict):
    task: dict
    topic: str
    iteration_index: int
    research_context: dict
    extra_hints: Optional[str]
    audit_feedback: Optional[dict]
    check_data_action: str
    check_data_verdict: Optional[dict]
    scraping_packet: Optional[dict]
    coverage_metrics: Optional[dict]
    query_target_map: Optional[list]
    draft: dict
    review: str
    revision_notes: str
