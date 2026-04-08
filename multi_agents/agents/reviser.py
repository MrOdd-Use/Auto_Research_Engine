from .utils.views import print_agent_output
from .utils.llms import call_model
from multi_agents.route_agent import build_route_context

sample_revision_notes = """
{
  "draft": "The complete revised draft in full markdown format",
  "revision_notes": "Your message to the reviewer about the changes you made to the draft based on their feedback"
}
"""


class ReviserAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def revise_draft(self, draft_state: dict):
        """
        Review a draft article
        :param draft_state:
        :return:
        """
        review = draft_state.get("review")
        task = draft_state.get("task") or {}
        draft_report = draft_state.get("draft")
        checkpoint_note = (
            str(task.get("checkpoint_note") or "").strip()
            if task.get("checkpoint_target") == "reviser"
            else ""
        )
        note_block = (
            f"Additional rerun instruction for this revision pass: {checkpoint_note}\n"
            if checkpoint_note
            else ""
        )
        pending_opinions = draft_state.get("pending_opinions") or ""
        opinions_block = (
            f"\n## Opinion Items (please address each one, do not skip)\n\n{pending_opinions}\n\n"
            "After addressing each item, briefly note the location and method of revision in revision_notes.\n"
            if pending_opinions
            else ""
        )
        resolved_opinions = draft_state.get("resolved_opinions") or ""
        resolved_block = (
            f"\n## Previously Resolved Opinion Items\n\n{resolved_opinions}\n\n"
            "Do not regress these already-satisfied requirements while revising the draft.\n"
            if resolved_opinions
            else ""
        )
        prompt = [
            {
                "role": "system",
                "content": "You are an expert writer. Your goal is to revise drafts based on reviewer notes.",
            },
            {
                "role": "user",
                "content": f"""Draft:
{draft_report}

Reviewer's notes:
{review}
{opinions_block}
{resolved_block}
You have been tasked by your reviewer with revising the following draft, which was written by a non-expert.
If you decide to follow the reviewer's notes, please write a new draft and make sure to address all of the points they raised.
Please preserve the existing report structure and section headings unless the reviewer explicitly asks you to change them.
Please keep all other aspects of the draft the same.
{note_block}
You MUST return nothing but a JSON in the following format:
{sample_revision_notes}
""",
            },
        ]

        route_context = build_route_context(
            application_name=str(task.get("application_name") or "auto_research_engine"),
            shared_agent_class="reviser_agent",
            agent_role="reviser",
            stage_name="draft_revision",
            system_prompt="You are an expert writer. Your goal is to revise drafts based on reviewer notes.",
            task=str(task.get("query") or ""),
            state=draft_state,
            task_payload=task,
        )
        response = await call_model(
            prompt,
            model=task.get("model"),
            response_format="json",
            route_context=route_context,
        )
        return response

    async def run(self, draft_state: dict):
        print_agent_output(f"Rewriting draft based on feedback...", agent="REVISOR")
        revision = await self.revise_draft(draft_state)

        if draft_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "revision_notes",
                    f"Revision notes: {revision.get('revision_notes')}",
                    self.websocket,
                )
            else:
                print_agent_output(
                    f"Revision notes: {revision.get('revision_notes')}", agent="REVISOR"
                )

        return {
            "draft": revision.get("draft"),
            "revision_notes": revision.get("revision_notes"),
        }
