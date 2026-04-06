from typing import List, Optional, TypedDict


class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: List[str]
    section_details: List[dict]
    research_data: List[dict]
    scraping_packets: List[dict]
    check_data_reports: List[dict]
    audit_feedback_queue: List[dict]
    extra_hints: Optional[str]
    human_feedback: str
    review: str
    revision_notes: str
    review_iterations: int
    final_draft: str
    # Report layout
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: List[str]
    claim_annotations: Optional[list]
    report: str
    # Claim confidence
    claim_confidence_report: Optional[list]
    claim_reflexion_iterations: int
    source_index: Optional[dict]
    indexed_research_data: Optional[str]
    model_decisions: Optional[list]
    # Review diff tracking
    _draft_before_revision: Optional[str]
