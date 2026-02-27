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
    scrap_packet: Optional[dict]
    draft: dict
    review: str
    revision_notes: str
