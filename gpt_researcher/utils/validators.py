"""Pydantic validation models for GPT Researcher."""

from typing import List

from pydantic import BaseModel, Field


class Subtopic(BaseModel):
    """Model representing a single research subtopic.

    Attributes:
        task: The name or description of the subtopic task.
    """
    task: str = Field(description="Task name", min_length=1)


class Subtopics(BaseModel):
    """Model representing a collection of research subtopics.

    Used for parsing and validating subtopic lists generated
    by the LLM during research planning.

    Attributes:
        subtopics: List of Subtopic objects.
    """
    subtopics: List[Subtopic] = []


class SectionOutline(BaseModel):
    """A single section in the research outline with enriched context."""
    header: str = Field(description="Section title", min_length=1)
    description: str = Field(description="One-sentence scope description")
    key_points: List[str] = Field(default_factory=list, description="2-3 points to investigate")
    research_queries: List[str] = Field(default_factory=list, description="2-3 search queries")


class ResearchOutline(BaseModel):
    """Complete structured research outline from the planner."""
    title: str = Field(description="Report title", min_length=1)
    date: str = Field(description="Today's date")
    query_intent: str = Field(
        description="Classified intent: analytical, descriptive, comparative, or exploratory"
    )
    sections: List[SectionOutline] = Field(
        description="Ordered list of research sections",
        min_length=1,
    )
