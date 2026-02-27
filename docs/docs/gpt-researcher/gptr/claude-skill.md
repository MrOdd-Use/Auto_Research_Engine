# Claude Skill

Auto_Research_Engine is available as a [Claude Skill](https://skills.sh/MrOdd-Use/Auto_Research_Engine/Auto_Research_Engine), allowing you to extend Claude's research capabilities directly within Claude Code and other Claude-powered applications.

## What are Claude Skills?

Skills are modular packages that extend Claude's capabilities by providing specialized knowledge, workflows, and tools. When you install Auto_Research_Engine as a skill, Claude gains access to deep research procedures, helping it conduct comprehensive research with citations.

## Installation

Install Auto_Research_Engine as a Claude Skill using the skills CLI:

```bash
npx skills add MrOdd-Use/Auto_Research_Engine
```

This installs the skill from the [Auto_Research_Engine GitHub repository](https://github.com/MrOdd-Use/Auto_Research_Engine).

## What's Included

The Auto_Research_Engine skill provides Claude with:

- **Architecture Knowledge** - Understanding of the planner-executor-publisher pattern
- **Component Signatures** - Method signatures for `GPTResearcher`, `ResearchConductor`, `ReportGenerator`
- **Integration Patterns** - How to add features, retrievers, and customize workflows
- **Configuration Reference** - All environment variables and config options
- **API Reference** - REST and WebSocket API documentation

## Usage

Once installed, Claude can help you with:

- Understanding Auto_Research_Engine's architecture
- Adding new features following the 8-step pattern
- Debugging research pipelines
- Integrating MCP data sources
- Customizing report generation
- Adding new retrievers

## Skill Structure

The skill is located in the `.claude/` directory of the repository:

```
.claude/
├── SKILL.md              # Main skill file (lean, <500 lines)
└── references/           # Detailed documentation
    ├── architecture.md
    ├── components.md
    ├── flows.md
    ├── prompts.md
    ├── retrievers.md
    ├── mcp.md
    ├── deep-research.md
    ├── multi-agents.md
    ├── adding-features.md
    ├── advanced-patterns.md
    ├── api-reference.md
    └── config-reference.md
```

## Learn More

- [Skills.sh - Auto_Research_Engine](https://skills.sh/MrOdd-Use/Auto_Research_Engine/Auto_Research_Engine) - View on skills.sh registry
- [Claude Code Documentation](https://docs.claude.com/en/docs/claude-code/skills) - Official skills documentation
- [Auto_Research_Engine Documentation](https://docs.gptr.dev) - Full project documentation
