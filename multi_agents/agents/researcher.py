from gpt_researcher import GPTResearcher
from colorama import Fore, Style
from multi_agents.route_agent import build_route_scope, route_scope
from .utils.views import print_agent_output
from .scrap import ScrapAgent


class ResearchAgent:
    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone

    async def research(self, query: str, research_report: str = "research_report",
                       parent_query: str = "", verbose=True, source="web", tone=None, headers=None):
        # Initialize the researcher
        researcher = GPTResearcher(query=query, report_type=research_report, parent_query=parent_query,
                                   verbose=verbose, report_source=source, tone=tone, websocket=self.websocket, headers=self.headers)
        scope = build_route_scope(
            application_name="auto_research_engine",
            shared_agent_class="research_agent",
            agent_role="researcher" if research_report == "subtopic_report" else "browser",
            stage_name="section_research" if research_report == "subtopic_report" else "initial_research",
            task=query,
        )
        with route_scope(scope):
            # Conduct research on the given query
            await researcher.conduct_research()
            # Write the report
            report = await researcher.write_report()

        return report

    async def run_subtopic_research(self, parent_query: str, subtopic: str, verbose: bool = True, source="web", headers=None):
        try:
            report = await self.research(parent_query=parent_query, query=subtopic,
                                         research_report="subtopic_report", verbose=verbose, source=source, tone=self.tone, headers=headers)
        except Exception as e:
            print(f"{Fore.RED}Error in researching topic {subtopic}: {e}{Style.RESET_ALL}")
            report = None
        return {subtopic: report}

    async def run_initial_research(self, research_state: dict):
        task = research_state.get("task")
        query = task.get("query")
        source = task.get("source", "web")

        if self.websocket and self.stream_output:
            await self.stream_output("logs", "initial_research", f"Running initial research on the following query: {query}", self.websocket)
        else:
            print_agent_output(f"Running initial research on the following query: {query}", agent="RESEARCHER")
        return {"task": task, "initial_research": await self.research(query=query, verbose=task.get("verbose"),
                                                                      source=source, tone=self.tone, headers=self.headers)}

    async def run_depth_research(self, draft_state: dict):
        task = draft_state.get("task")
        topic = draft_state.get("topic")
        research_context = draft_state.get("research_context") or {}
        parent_query = task.get("query")
        source = task.get("source", "web")
        verbose = task.get("verbose")

        enriched_topic = self._build_enriched_topic(topic, research_context)

        if self.websocket and self.stream_output:
            await self.stream_output("logs", "depth_research", f"Running in depth research on the following report topic: {topic}", self.websocket)
        else:
            print_agent_output(f"Running in depth research on the following report topic: {topic}", agent="RESEARCHER")
        research_draft = await self.run_subtopic_research(parent_query=parent_query, subtopic=enriched_topic,
                                                          verbose=verbose, source=source, headers=self.headers)
        return {"draft": research_draft}

    async def run_depth_scrap(self, draft_state: dict):
        """
        Compatibility bridge for legacy callsites that still use ResearchAgent.
        Delegates to ScrapAgent when ASA flow is desired.
        """
        scrap_agent = ScrapAgent(
            websocket=self.websocket,
            stream_output=self.stream_output,
            tone=self.tone,
            headers=self.headers,
        )
        return await scrap_agent.run_depth_scrap(draft_state)

    @staticmethod
    def _build_enriched_topic(topic: str, context: dict) -> str:
        """Combine section header with planner context for a richer research query."""
        if not context or not context.get("description"):
            return topic
        parts = [topic, f"Focus: {context['description']}"]
        if context.get("key_points"):
            points = "; ".join(context["key_points"])
            parts.append(f"Key points to cover: {points}")
        return " | ".join(parts)
