# 编排与工作流

> 覆盖题号：Q01, Q02, Q03, Q04, Q05, Q06, Q14, Q16, Q22, Q23, Q24, Q29, Q35

### Q01（基础）：什么是 LLM Agent？它和普通“单次调用 LLM”的区别是什么？

**标准答案（可直接说）：**
一句话：LLM Agent（智能体）是把 LLM 当作“决策与生成内核”，并通过**状态（state）+ 工具（tools）+ 工作流（workflow）**持续迭代完成任务的系统，而不是一次性问答。

展开：单次调用 LLM 通常是“输入提示词 → 输出文本”，缺少持续状态与反馈回路。Agent 会维护任务状态（例如研究大纲、证据包、审阅意见），按工作流推进，并在必要时调用外部能力（检索、抓取、格式导出等）。在本项目里，多智能体把角色拆分为浏览/规划/研究/写作/审阅/修订/发布，并由 LangGraph 状态机串起来，形成可回流、可控的端到端链路（workflow-driven agent）。

常见坑/反杀点：
- 把 “Agent = 多次调用 LLM” 说得太虚；要强调 state、路由、工具与闭环。
- 把“多智能体”当成必须；实际上单智能体也能跑，但可控性与鲁棒性不同。

**相关模块（对应实现）：**
- gpt_researcher/agent.py
- multi_agents/agents/orchestrator.py
- multi_agents/memory/research.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：用 LangGraph 的全局状态图（`ResearchState`）把 browser→planner→human→researcher→writer→reviewer/reviser→publisher 串成可回流链路（见 `multi_agents/agents/orchestrator.py`），产物持续写入 `ResearchState`（见 `multi_agents/memory/research.py`）。
- 为什么：把“决策/生成”拆成可观测节点，允许基于 state 字段做条件路由（例如 `human_feedback`、`review`）。
- 利弊：利是可控/可插 gate/易扩展；弊是编排与状态管理开销更高，且 state 变大后调试与隐私风险上升。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：在示例里，“追问记录/大纲/证据包/审阅意见”等就是 state；“检索/抓取/导出”等外部能力是 tools；从追问→大纲→证据门禁→写作→审阅→发布就是 workflow。

**定义：** Agent 里“状态”和“工具”分别指什么？  
答题要点：状态是可序列化的任务上下文（大纲、证据、审阅意见）；工具是外部能力（检索、抓取、导出）与路由策略。

**实现：** 你们的 state 里有什么关键字段？  
答题要点：`ResearchState` 包含初始研究、章节信息、并行研究结果、审阅/修订、最终稿与发布内容等（见 `multi_agents/memory/research.py`）。

**边界：** 什么场景不适合做成 Agent？  
答题要点：需求非常确定、输出短且不需要外部证据/多轮迭代的任务；Agent 引入的编排与评测成本可能得不偿失。

**取舍：** 为什么用“状态机编排”，而不是写一堆 if/else？  
答题要点：状态机可视化、可回流、可插拔节点；更容易加 HITL、加 gate、加评测/观测。

**优化：** 如果要更“像产品级 Agent”，你会加什么？  
答题要点：更强的上下文工程（证据片段结构化、引用对齐）、抗注入、可观测性指标面板、离线回归门禁。

---

### Q02（基础）：多智能体 vs 单智能体：什么时候值得上多智能体？

**标准答案（可直接说）：**
一句话：当任务是“多阶段、多目标、需要外部证据、需要质量门禁”的长链流程时，多智能体更值；当任务短、结构简单时，单智能体更划算。

展开：多智能体的价值在于**角色分工与质量控制**。例如规划阶段关注 MECE 章节边界，研究阶段关注证据采集与覆盖度，写作阶段关注结构与引用格式，审阅阶段关注一致性与可发布性。在本项目里，链路包含 HITL 回流、ASA 多轮证据采集、Check Data gate、Reviewer–Reviser 迭代、Publisher 多格式发布，这些更适合用“分工 + 状态机路由”来保证可控性。

常见坑/反杀点：
- 只说“并行更快”；更重要的是质量门禁、可控性与可解释性。
- 忽略协调成本：多智能体要解决信息共享、重复、冲突与预算控制。

**相关模块（对应实现）：**
- multi_agents/README.md
- multi_agents/agents/orchestrator.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：角色被拆成独立节点与提示（Writer/Editor/Research/Human/Reviewer/Reviser/Publisher），并行研究在编辑/规划模块里按“章节”并发执行并聚合结果（见 `multi_agents/agents/editor.py`）。
- 为什么：把规划、证据采集、写作、审阅的目标函数分离，避免一个超长 prompt 同时优化多目标导致不稳定。
- 利弊：利是质量门禁更清晰、并行加速；弊是协调成本（重复/冲突/预算）上升，需要更强的去重、共享缓存与一致性校验。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 你怎么理解“角色分工”？  
答题要点：每个 agent 关注不同目标（覆盖/事实性/结构/可发布性），并通过 state 传递产物。

**实现：** 你们有哪些角色？各自负责什么？  
答题要点：Human/Chief Editor/Researcher/Editor/Writer/Reviewer/Reviser/Publisher（见 `multi_agents/README.md`）。

**边界：** 多智能体最容易失败在哪里？  
答题要点：重复劳动、互相矛盾、上下文爆炸、路由不稳定、评测不闭环。

**取舍：** 为什么不做成“一个超强单体 agent + 工具”？  
答题要点：单体难在同一提示中兼顾规划、检索、写作与审阅；拆分更易控与更易迭代。

**优化：** 如何降低多智能体的协调成本？  
答题要点：结构化 state、证据片段规范、共享检索缓存、统一评测与回归门禁。

---

### Q03（基础）：LangGraph 是什么？为什么用“状态机/有向图”来编排 agent？

**标准答案（可直接说）：**
一句话：LangGraph 是构建“有状态、多角色、可路由”的 LLM 应用编排库，用图结构表达节点与条件分支，适合多智能体工作流。

展开：本项目把研究流程建模为 state machine：节点对应阶段能力（初始研究、规划、人工审阅、并行深研、写作、审阅、修订、发布），条件分支决定流程走向（例如有反馈则回到规划；审阅通过则发布，否则修订再审阅）。相比手写流程控制，图更容易扩展（插入 gate、插入评测、插入新数据源）且更利于调试与追踪。

常见坑/反杀点：
- 把 LangGraph 误解为“并发框架”；它更像可组合的状态机/工作流模型。
- 忽略“回流”能力是关键价值之一（HITL/修订循环）。

**相关模块（对应实现）：**
- multi_agents/agent.py
- multi_agents/agents/orchestrator.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：全局用 LangGraph 的状态图（`ResearchState`），章节级用状态图（`DraftState`）；条件分支通过读取 state 字段决定走向（见 `multi_agents/agents/orchestrator.py`、`multi_agents/agents/editor.py`）。
- 为什么：图结构天然支持插拔节点（加 gate/评测/HITL）与回流（revise→review），比手写 if/else 更易维护与可视化。
- 利弊：利是扩展与回放更容易；弊是需要理解图模型与状态边界，节点间契约（state shape）不稳会放大故障面。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 状态机编排相对线性 pipeline 的差异？  
答题要点：强调显式 state、显式路由与可回流；pipeline 多为线性步骤，回流与分支更难维护。

**实现：** 你们图里有哪些关键分支？  
答题要点：HITL 节点“有反馈回规划、无反馈继续”；审阅节点“通过发布、否则修订再审阅”（见 orchestrator）。

**边界：** 状态机会在什么情况下变得难维护？  
答题要点：分支爆炸、缺少统一 state schema、节点输出不规范；需要 schema 约束与容错。

**取舍：** 为什么不用通用工作流引擎？  
答题要点：LangGraph 更贴近 LLM 应用的路由与结构化输出；生产级可靠性可再叠加工程化工作流（可选）。

**优化：** 生产化你会加哪些增强？  
答题要点：幂等与重试、任务持久化、限流、指标与追踪、失败回放。

---

### Q04（基础）：你项目的端到端链路是什么？每段输入/输出是什么？

**标准答案（可直接说）：**
一句话：链路是“初始资料收集 → 结构化大纲 → HITL 审阅回流 → 分章节并行深研 → 汇编写作 → 审阅/修订循环 → 多格式发布”，通过 `ResearchState` 传递与聚合结果。

展开：初始资料收集产生初步研究文本；规划阶段把初始研究转换为结构化大纲（章节、描述、关键点、章节研究查询）；HITL 节点可修改大纲并触发回流；随后按章节并行做深研，产出每章草稿与证据包；写作阶段生成引言/目录/结论/引用与统一 headers；终稿经过审阅/修订循环后进入发布，导出 markdown/pdf/docx 等。

常见坑/反杀点：
- 不要把“写作”说成只拼接：Writer 会生成引言/结论/目录与来源列表。
- 面试官会追问“你怎么传数据、怎么回流”，要能落到 state 字段。

**相关模块（对应实现）：**
- multi_agents/agents/orchestrator.py
- multi_agents/memory/research.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：链路节点在 `multi_agents/agents/orchestrator.py` 明确：初始研究写入 `initial_research`；规划写入 `section_details`；并行深研聚合 `research_data/scrap_packets/check_data_reports`；写作生成 `introduction/conclusion/table_of_contents/sources`；审阅/修订产出 `final_draft`；发布落到 `report` 与多格式文件（见 `multi_agents/agents/publisher.py`）。
- 为什么：把“方向对齐（大纲）”放在前面，把“内容打磨（Reviewer–Reviser）”放在后面，避免在错误方向上堆成本。
- 利弊：利是每段输入输出可解释、可复用；弊是步骤变多，端到端时延更长，且每段的失败需要降级/兜底策略。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：用示例把端到端产物串起来：追问收敛→结构化大纲（`section_details`）→按章证据包→终稿内容→发布（markdown/pdf/docx 等）。

**定义：** 最关键的质量门禁点是什么？  
答题要点：HITL 防方向偏离；Check Data 防证据不足；Reviewer–Reviser 防终稿不可发布。

**实现：** state 在链路里如何演进？  
答题要点：LangGraph 节点通常只返回“增量 dict”，运行时按 key 合并进 state；因此 state 是“逐步累积 + 覆盖更新”的。
- 全局用 `ResearchState`（见 `multi_agents/memory/research.py`）：入口只有 `task`，随后依次写入 `initial_research` → `sections/section_details/title/date` → `human_feedback`（有反馈回流） → `research_data/scrap_packets/check_data_reports`（并行聚合） → `introduction/conclusion/table_of_contents/sources/headers` → `review/final_draft/review_iterations` → `report`（见 `multi_agents/agents/orchestrator.py`、`multi_agents/agents/publisher.py`）。
- 章节级用 `DraftState`（见 `multi_agents/memory/draft.py`）：每章携带 `research_context/extra_hints/audit_feedback/iteration_index`；研究节点产出 `draft/scrap_packet`，Check Data 写入 `check_data_action/check_data_verdict` 并在 `retry` 时更新 `iteration_index/extra_hints`，驱动下一轮补证据；`accept` 时不覆盖 `draft`，保留上一轮产物（见 `multi_agents/agents/editor.py`、`multi_agents/agents/check_data.py`）。
- 条件路由只看少数“信号字段”：全局主要是 `human_feedback`、`review`、`review_iterations`；章节主要是 `check_data_action`（见 `multi_agents/agents/orchestrator.py`、`multi_agents/agents/editor.py`）。

**边界：** 哪些输入会导致链路天然不稳定？  
答题要点：过宽泛或缺少公开证据的题目；强时效信息；被 SEO spam 污染的主题。

**取舍：** 为什么把并行放在章节粒度？  
答题要点：章节天然可并行、边界清晰；便于控制覆盖面与避免重复。

**优化：** 如何减少章节间重复与冲突？  
答题要点：跨章节去重缓存、冲突检测、最终写作阶段统一一致性校验（可选升级）。

---

### Q05（基础）：什么是 Human-in-the-loop（HITL）？你们把它放在哪个节点，为什么？

**标准答案（可直接说）：**
一句话：HITL 是在关键决策点引入人工审阅与纠偏；我们放在**大纲规划之后、深研之前**，先对齐研究路径再投入检索与写作成本。

展开：规划阶段输出结构化大纲（章节边界/关键点/研究查询），随后进入 HITL 节点展示大纲并等待用户反馈。若用户反馈存在，工作流回流到规划阶段并融合反馈；若无反馈则继续深研。把人工介入放在这里的原因是：方向一旦错，后续检索/写作成本会指数放大；先对齐能显著减少“写得很长但不对题”的失败。

常见坑/反杀点：
- 不要把 HITL 说成“随时人工插手”；要强调在最有价值的关口插入。
- 面试官会问“如何避免无限回流”：当前实现偏用户自控，工程化上限属于可选优化。

**相关模块（对应实现）：**
- multi_agents/agents/human.py
- multi_agents/agents/orchestrator.py
- multi_agents/task.json

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：HITL 节点读取/展示规划结果并产出 `human_feedback`；工作流根据 `human_feedback` 是否为空决定回到 planner 还是进入 researcher（见 `multi_agents/agents/orchestrator.py`、`multi_agents/agents/human.py`）。
- 为什么：规划阶段修改的“杠杆”最大——改对方向能减少后续检索与写作返工。
- 利弊：利是显著降低长链漂移；弊是依赖用户响应、可能阻塞流水线；生产级通常要加回流上限与结构化反馈表单。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** HITL 的价值是什么？  
答题要点：把意图对齐/方向选择交给人，降低长链漂移风险。

**实现：** 回流触发条件是什么？  
答题要点：以 `human_feedback` 是否为空作为分支信号；有反馈回规划，无反馈进入深研。

**边界：** 用户反馈很含糊怎么办？  
答题要点：当前实现将反馈原样带入规划；生产级可用结构化反馈表单与校验（可选升级）。

**取舍：** 为什么不把 HITL 放在终稿阶段？  
答题要点：终稿阶段改动成本更高；规划阶段改动收益更大。

**优化：** 如何把 HITL 做得更产品化？  
答题要点：大纲差异对比、章节锁定、章节级批注、回流次数上限（可选升级）。

---

### Q06（基础）：什么是 MECE？在章节拆解/大纲规划里怎么落地？

**标准答案（可直接说）：**
一句话：MECE（Mutually Exclusive, Collectively Exhaustive）是“互斥且穷尽”的结构化拆解原则；用于把主题拆成不重叠、覆盖全面的章节边界。

展开：规划阶段会先识别查询意图（analytical/descriptive/comparative/exploratory），再把主题拆成维度，并要求章节之间不重叠、不包含“Introduction/Conclusion/References”等泛化标题。最终输出结构化章节信息，作为后续并行深研的单位，从源头控制重复与遗漏（MECE outline decomposition）。

常见坑/反杀点：
- MECE 不是“多列几个点”；要强调边界与覆盖。
- 章节拆得太细会导致上下文碎片化；太粗会导致深研不聚焦。

**相关模块（对应实现）：**
- multi_agents/agents/editor.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：规划 system prompt 明确 MECE 约束与禁用泛化章节（Introduction/Conclusion/References 等），并限制章节数（见 `multi_agents/agents/editor.py` 的规划提示与标题过滤）。
- 为什么：章节边界清晰才能把“章节”作为并行深研单位，并减少重复证据与互相打架的结论。
- 利弊：利是并行可控、覆盖更稳定；弊是对含糊/争议主题可能“硬拆”，需要在大纲里显式加入口径/争议维度。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例卡片里的 6 个章节就是 MECE 的直观形态：边界清晰、尽量不重叠，并覆盖关键维度；每章都配套 `research_queries` 便于并行深研。

**定义：** 互斥与穷尽分别怎么判断？  
答题要点：互斥看章节边界是否重叠；穷尽看是否覆盖关键维度（不要求覆盖所有细节）。

**实现：** 如何防止泛化章节混入？  
答题要点：对章节标题做过滤与去重，禁止高频泛化标题并限制最大章节数（见 editor 约束）。

**边界：** 哪些主题很难做到 MECE？  
答题要点：跨学科/定义不统一/争议大主题；需要显式加入“口径说明/争议维度”。

**取舍：** MECE 和“按检索结果堆砌”相比？  
答题要点：MECE 更可控、更易并行；但对规划质量要求更高。

**优化：** 如何评估与迭代 MECE 质量？  
答题要点：统计章节重复率、覆盖缺失项、Reviewer 反馈缺口；将信号回流到规划约束（可选升级）。

---

### Q14（基础）：Reviewer–Reviser 在修什么？它与 Check Data 的分工边界是什么？

**标准答案（可直接说）：**
一句话：Reviewer–Reviser 是终稿的质量控制循环，面向结构、逻辑、一致性与 guidelines；Check Data 是研究阶段证据门禁，面向证据足够性与口径正确性。

展开：Writer 生成引言/结论/目录/来源并形成布局后，Reviewer 根据 guidelines 质检，若满足则输出空反馈表示接受，否则给出修订意见；Reviser 按意见修订并回到 Reviewer 再审。该循环可设置最大轮次上限以平衡质量与时延（上限策略可写进面试口径）。

常见坑/反杀点：
- 不要把 Reviewer–Reviser 说成事实校验器；它更偏“可发布性”。
- 要能解释为什么不对每章都做 Reviewer（成本与价值）。

**相关模块（对应实现）：**
- multi_agents/agents/reviewer.py
- multi_agents/agents/reviser.py
- multi_agents/agents/orchestrator.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：终稿阶段由 Reviewer 输出 `review`（为空表示接受），Orchestrator 依据 `review` 路由到发布或修订；修订次数写入 `review_iterations`，达到 `max_review_rounds`（默认 3，可配置）后直接发布当前最优稿（见 `multi_agents/agents/orchestrator.py`、`multi_agents/memory/research.py`、`multi_agents/agents/reviewer.py`、`multi_agents/agents/reviser.py`）。
- 为什么：Reviewer–Reviser 面向“可发布性”（结构/一致性/指南），与研究阶段的证据门禁分工，避免把所有校验都堆在一个节点。
- 利弊：利是产物质量更稳定；弊是如果前置证据不足，后置润色可能“更像真的”，因此必须强调先 gate 后写作/审阅的顺序。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例里 Reviewer 主要看结构、逻辑、一致性与指南满足；不把它说成事实校验器，事实风险更多靠“证据门禁 + 引用约束”。

**定义：** 可发布性包含哪些维度？  
答题要点：结构完整、逻辑连贯、规范一致、引用格式合理、满足 guidelines。

**实现：** “接受/修订”判定信号是什么？  
答题要点：Reviewer 是否返回空反馈（或等价信号）；空则进入发布，否则进入修订再回审。

**边界：** 评审循环常见风险？  
答题要点：过严导致反复；过松导致质量不过关；需要上限与提示约束。

**取舍：** 为什么把校验 gate 放研究阶段？  
答题要点：证据不足时先补证据；先写再评会浪费成本且可能掩盖根因。

**优化：** 如何让 Reviewer 意见更可执行？  
答题要点：结构化 checklist、按段落定位、差异修订、与评测信号联动（可选升级）。

---

### Q16（基础）：Prompt Engineering 在你项目里主要约束什么？为什么强调 JSON 输出？

**标准答案（可直接说）：**
一句话：Prompt Engineering 的核心是把不稳定的自然语言输出约束成可消费的结构（JSON），从而支撑路由、并行与回流；本项目在规划/写作/修订等环节都强调结构化输出。

展开：多智能体系统里，上游输出是下游输入。如果输出是自由文本，很难稳定解析与校验。本项目要求规划阶段输出包含章节详情与章节级 research queries 的 JSON；Writer/Reviser 也以 JSON 约束返回引言/结论/目录/来源与修订说明，保证 Publisher 可以稳定生成最终布局并导出多格式。这种做法把“语言不确定性”限制在可控边界内（structured output）。

常见坑/反杀点：
- JSON 约束不等于 100% 稳；需要解析容错与降级策略。
- 面试官会追问“如何防止输出多余文本”；要回答严格格式约束与回退策略。

**相关模块（对应实现）：**
- multi_agents/agents/editor.py
- multi_agents/agents/writer.py
- multi_agents/agents/reviser.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：规划/写作等关键节点使用 `response_format="json"` 强制结构化输出，并用 `json_repair`/JSON 解析器做容错（见 `multi_agents/agents/utils/llms.py`、`multi_agents/agents/editor.py`、`multi_agents/agents/writer.py`、`multi_agents/agents/scrap.py`）。
- 为什么：链路是程序编排，不稳定的自然语言输出会导致解析失败、字段缺失与路由不可控。
- 利弊：利是可消费、可校验、可回放；弊是过强格式约束会降低表达自由度，且仍需要降级策略处理“半合规 JSON”。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 为什么结构化输出对 agent 编排关键？  
答题要点：可校验、可路由、可回流、可并行；减少“格式崩”导致的链路中断。

**实现：** 哪些阶段强制结构化？  
答题要点：规划（章节结构）、写作（引言/结论/目录/来源与 headers）、修订（新稿与修订说明）。

**边界：** 模型不按 JSON 输出怎么办？  
答题要点：现有实现包含解析与降级路径；生产级可加更严格 schema 校验与重试（可选升级）。

**取舍：** 为什么不用 YAML/XML？  
答题要点：JSON 通用、解析器成熟、易与 schema 验证结合；但对长文本可读性一般。

**优化：** 如何提升结构化稳定性？  
答题要点：字段级约束、分步生成、低温度、失败样本回归集与自动化检测（可选升级）。

---

### Q22（复杂）：Planner 的“意图分类 + 章节边界 + research_queries”如何设计提示与校验？模型不合规怎么处理？

**标准答案（可直接说）：**
一句话：Planner 用 system prompt 明确“意图识别→维度拆解→MECE 章节→每章研究查询”的步骤，并要求输出结构化 JSON；随后用 schema 校验与降级解析把输出规范化，保证链路不断。

展开：规划阶段的关键是把“初始资料收集结果”转成“可执行研究计划”。提示强调：章节互斥、覆盖全面、逻辑顺序、禁止引言/结论/参考等泛化章节，并要求为每章生成边界描述与研究查询列表。输出后先走结构校验（字段存在、章节数量上限、去重/过滤），若不合规则走降级路径（例如将章节解析成扁平列表并补齐默认字段），确保后续并行深研仍可执行。

常见坑/反杀点：
- 面试官会追问“为什么不是研究 agent 自己决定搜什么”；要强调全局结构与并行可控。
- 规划输出不稳定是常态；必须回答容错与降级策略。

**相关模块（对应实现）：**
- multi_agents/agents/editor.py
- gpt_researcher/utils/validators.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：Planner 通过 system prompt 强制“意图分类→维度拆解→MECE 章节→每章 research_queries”，并用 `ResearchOutline` 做字段校验；不合规时走归一化/兜底章节保证链路继续（见 `multi_agents/agents/editor.py`、`gpt_researcher/utils/validators.py`）。
- 为什么：规划输出是后续并行深研的“控制面”，必须结构稳定；失败时宁可降级推进，也不要无限重跑导致更不稳定。
- 利弊：利是可执行与可回归；弊是降级会牺牲章节边界质量，需要 HITL/Reviewer 在后面兜底。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例里 Planner 的核心产物就是可执行的 `section_details` 与每章 `research_queries`；追问收敛后的口径会直接改变章节边界与查询方向。

**定义：** 意图分类对后续有什么用？  
答题要点：决定章节组织方式（对比/因果/探索），影响 research queries 的方向与覆盖点。

**实现：** 如何限制章节数量与重复？  
答题要点：`max_sections` 约束，标题去重与过滤，禁用泛化标题。

**边界：** 降级解析的风险？  
答题要点：章节边界与研究查询变弱；需要 HITL 与后续 Reviewer 兜底。

**取舍：** 为什么不失败就强制重跑规划？  
答题要点：重跑可能更不稳定；降级能保证链路推进，失败信号留给后续门禁。

**优化：** 怎么提升规划质量？  
答题要点：更严格 schema、分步生成、少样本示例、回归评测驱动 prompt 迭代（可选升级）。

---

### Q23（复杂）：HITL 反馈如何并入下一轮规划？如何避免反馈导致结构崩坏/无限回流？

**标准答案（可直接说）：**
一句话：当前实现把用户反馈作为规划输入的一部分，触发回流重规划；无限回流的风险需要用“回流次数上限/结构化反馈/差异对比”工程化约束（可选升级）。

展开：HITL 节点展示结构化大纲并收集反馈；若反馈为空则继续深研，否则回到规划阶段重新生成并融合反馈。这样让“方向对齐”发生在成本最低的时点。边界上，用户可能连续提出反馈导致多次回流；当前实现偏用户自控。生产级可把回流上限写入 state，并把反馈结构化到章节级（增删改/锁定/优先级），避免自然语言反馈造成大纲漂移。

常见坑/反杀点：
- 别声称已实现“上限/表单化”如果仓库没有；要放在优化里讲。
- 要能说明反馈会影响每章 research queries，而不仅是章节标题。

**相关模块（对应实现）：**
- multi_agents/agents/human.py
- multi_agents/agents/orchestrator.py
- multi_agents/agents/editor.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：HITL 的自然语言反馈通过 `human_feedback` 回到 planner，规划提示里可选择是否融合反馈（见 `multi_agents/agents/orchestrator.py` 的条件边、`multi_agents/agents/editor.py` 的 feedback block）。
- 为什么：让用户在大纲层面“锁方向/改边界”，比深研后再改更省成本。
- 利弊：利是纠偏早；弊是自然语言反馈可能导致结构崩坏，生产级应把反馈结构化成章节级操作（增删改/锁定/优先级）并加入回流次数上限（可选升级）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例里用户反馈最好落到“章节级增删改/优先级/锁定”，再回流更新 `section_details` 与 research_queries；工程上需要回流上限避免无限循环。

**定义：** HITL 的输入输出是什么？  
答题要点：输入是大纲展示；输出是 `human_feedback`，供规划阶段融合。

**实现：** 回流触发条件？  
答题要点：反馈是否为空；有反馈回规划，无反馈继续。

**边界：** 多次回流的代价？  
答题要点：时延/成本上升；需要上限与更强结构化交互（可选升级）。

**取舍：** 为什么不在深研后再改大纲？  
答题要点：那时改动会浪费已采集证据；规划阶段纠偏收益最大。

**优化：** 如何将反馈变成稳定指令？  
答题要点：结构化反馈 schema、章节级 diff、锁定章节、回流上限与自动总结（可选升级）。

---

### Q24（复杂）：分章节并行深研如何组织结果聚合？失败如何降级？

**标准答案（可直接说）：**
一句话：以“章节”为并行单位，每章携带规划上下文独立深研，产出章节草稿与证据/校验信息；最终聚合到 `ResearchState` 的 `research_data/scrap_packets/check_data_reports` 等字段。

展开：规划阶段产出 `section_details`（描述/关键点/研究查询），深研阶段为每章构造独立输入并执行采集与写作草稿，形成章节级结果。聚合时把章节草稿列表写入 `research_data`，证据包写入 `scrap_packets`，校验报告写入 `check_data_reports`，便于 Writer 汇编与后续审阅。降级方面，当缺少结构化章节详情时可退化为扁平章节列表并补齐默认字段；当单章失败可返回缺省或占位，避免整体链路中断（具体边界以实现为准）。

常见坑/反杀点：
- 并行可能导致重复与冲突；要讲规划边界、去重与终稿一致性兜底。
- 不要把“失败降级”说得绝对；要说明是工程折中与可扩展点。

**相关模块（对应实现）：**
- multi_agents/agents/editor.py
- multi_agents/memory/research.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：以章节为并行单位，把 `section_details` 注入每章 `DraftState.research_context`，并行跑研究/校验闭环后，将 `draft/scrap_packet/check_data_verdict` 聚合回 `ResearchState` 的 `research_data/scrap_packets/check_data_reports`（见 `multi_agents/agents/editor.py`、`multi_agents/memory/*`）。
- 为什么：章节边界清晰，天然适合并行；聚合字段也方便 Writer/Reviewer 消费。
- 利弊：利是吞吐高；弊是会引入跨章节重复与结论冲突，通常需要跨章去重与一致性检查（可选升级）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例里以“章节”为并行单位：每章独立产出草稿+证据包+门禁结论，再聚合成全局可写作与可审阅的输入。

**定义：** 为什么章节粒度适合并行？  
答题要点：结构天然可拆、边界明确、便于汇编与审阅。

**实现：** 规划信息如何进入章节深研？  
答题要点：每章携带 description/key_points/research_queries，作为研究上下文与约束提示输入。

**边界：** 并行的新问题？  
答题要点：重复证据、冲突结论、共享资源竞争；需要缓存与一致性校验（可选升级）。

**取舍：** 为什么不按“搜索目标”并行？  
答题要点：目标太细会导致写作碎片化；章节更贴近报告交付结构。

**优化：** 如何做跨章节一致性？  
答题要点：全局 claims 表、跨章节去重/冲突检测、终稿统一校验与改写（可选升级）。

---

### Q29（复杂）：为什么把“校验 gate”放在研究阶段、把 Reviewer–Reviser 放在终稿阶段？如果每章都 Reviewer 会怎样？

**标准答案（可直接说）：**
一句话：研究阶段先保证“证据口径与足够性”，终稿阶段再保证“可发布性与整体一致性”；每章都 Reviewer 会显著放大成本与时延，还可能在证据不足时做无效打磨。

展开：章节研究首先要解决“有没有足够且匹配的证据”，否则写得再漂亮也不可靠，因此先用 Check Data gate 驱动补证据；终稿阶段才适合做结构、引言/结论、引用格式与整体一致性控制。若每章都 Reviewer–Reviser，会造成大量重复审阅与修订，并可能把根因（证据不足）掩盖在表面润色里。

常见坑/反杀点：
- 面试官会问“章节草稿不需要质量吗”；要强调 gate + 规划上下文已在控制，终稿再提升整体质量。

**相关模块（对应实现）：**
- multi_agents/agents/editor.py
- multi_agents/agents/orchestrator.py
- multi_agents/agents/reviewer.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：研究阶段先过 Check Data gate 决定是否继续补证据；终稿阶段才进入 Reviewer–Reviser 打磨，并用 `review_iterations/max_review_rounds` 对终稿修订循环做硬上限控制（见 `multi_agents/agents/editor.py`、`multi_agents/agents/orchestrator.py`、`multi_agents/memory/research.py`）。
- 为什么：证据不足时继续写作/审阅是浪费；把 gate 前置能尽早暴露风险并控制回流。
- 利弊：利是节省大头成本；弊是如果每章都做 Reviewer，会把审阅成本线性放大且在证据不足时“越修越像真”，因此更适合只在终稿做一致性与规范检查。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 证据口径正确 vs 可发布性的差别？  
答题要点：前者是事实/证据对齐；后者是结构/表达/规范与整体一致性。

**实现：** 终稿审阅循环如何进入发布？  
答题要点：审阅反馈为空则发布，否则进入修订再回审，直到接受或达到工程上限（上限策略可产品化）。

**边界：** 这种分层的风险？  
答题要点：章节草稿可能仍有逻辑问题；需要 Writer 汇编与终稿审阅兜底。

**取舍：** 为什么不在章节阶段就严格写作规范？  
答题要点：章节阶段重点是证据与覆盖；过早规范化会增加成本且不提升事实性。

**优化：** 如何低成本提升章节质量？  
答题要点：轻量 checklist、结构化草稿模板、跨章节重复检测，只在失败时触发更强审阅（可选升级）。

---

### Q35（复杂）：为什么不用别的设计？（Tree-of-Thought 搜索、全局向量库、每步强制引用渲染、每章都 Reviewer 等）

**标准答案（可直接说）：**
一句话：我们选择“状态机编排 + 结构化规划 + 多轮证据采集 + 校验 gate + 终稿审阅 + benchmark 回归”是为了在工程可控性、成本与质量之间平衡；替代设计并非不可用，但会在复杂度、成本或可维护性上带来更高代价。

展开：更复杂的搜索/推理能提升深度，但在线成本与调参复杂度更高；全局向量库适合内网知识与长期记忆，但开放 web 仍需抓取与清洗且建设成本高；每步强制引用渲染能更可验证，但显著增加标注与写作复杂度；每章都 Reviewer 会把成本放大并可能在证据不足时无效打磨。当前架构的取舍原则是：用规划与证据选择控制覆盖，用 gate 控制事实风险，用终稿审阅控制可发布性，用评测驱动迭代。

常见坑/反杀点：
- 不要否定替代方案；要体现你理解适用场景并能解释为什么当前不选。

**相关模块（对应实现）：**
- multi_agents/agents/orchestrator.py
- multi_agents/agents/editor.py
- multi_agents/agents/scrap.py
- evals/README.md

**技术细节（实现 / 为什么 / 利弊）：**
- 实现（现状）：当前架构以“规划（MECE）+ ASA 证据采集 + Check Data gate + 终稿 Reviewer–Reviser + Publisher 多格式”作为主干（见 `multi_agents/agents/*`）。
- 为什么：相比 Tree-of-Thought/全局向量库/每步强制引用渲染等更复杂方案，这条链路更容易在线落地：控制面清晰、成本可控、失败可回流。
- 利弊：利是工程复杂度较低且可迭代；弊是深推理与断言级可验证性仍受限，需要在关键断言上逐步引入更强对齐/更强检索策略（可选升级）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 架构取舍的评价维度有哪些？  
答题要点：质量（事实/覆盖/一致性）、成本/时延、可维护性、可观测性、可扩展性。

**实现：** 当前架构如何体现这些维度？  
答题要点：结构化大纲与 HITL 控制方向；ASA+MMR 控制证据；Check Data 门禁；终稿审阅；evals 回归。

**边界：** 当前架构短板？  
答题要点：开放 web 安全防护仍需加强；断言级对齐与跨章节一致性仍可提升；部分评测受 source 质量限制。

**取舍：** 为什么不先做全局向量库？  
答题要点：当前瓶颈主要在抓取/清洗/证据选择；向量库更适合内网/长期记忆，作为后续演进。

**优化：** 3 个最有性价比的升级点？  
答题要点：结构化 RETRY 指令；关键断言优先的对齐校验；来源可信度与反证检索，并纳入回归门禁。

## 附：Top12 对应条目（从原附录迁移）

### 1) 为什么用 LangGraph 状态机编排，而不是 while + if/else？
**一句话结论：** 把长链路拆成可路由节点 + 显式状态，能稳定支持回流、插 gate 与可观测。

**实现落点：**
- `multi_agents/agents/orchestrator.py`：全局用 LangGraph 状态图编排 browser→planner→human→researcher→writer→reviewer/reviser→publisher；关键路由读 `human_feedback`、`review`、`review_iterations`。
- `multi_agents/memory/research.py`：端到端状态承载 `initial_research/section_details/research_data/scrap_packets/check_data_reports/final_draft/report` 等产物。

**为什么这样设计：**
- while/if 的问题是“回流”与“插拔”会把控制流写成面条；状态图能把每个决策点（HITL、审阅接受/修订、发布）工程化。

**利弊与边界：**
- 利：可观测（每节点可打点/日志）、可扩展（加 gate/加评测节点）、可回放（按 state 看每步产物）。
- 弊：状态字段一多就会膨胀，必须做摘要/裁剪与字段契约（schema）管理（可选升级）。

**对照题库：** Q03（LangGraph/状态机编排）

**追问补充：为什么不是 Airflow/Celery/自研调度**
- Airflow 更偏离线 DAG/批处理；要做 HITL、条件路由与回流（loop）通常得叠加额外的状态存储与交互层。本仓库的关键价值点是把 `human_feedback`（HITL）与 `review/review_iterations`（review-revise loop）变成显式路由。
- Celery 能做任务分发/并发，但不提供“状态机语义”；你仍需要自建状态存储、条件路由、回流上限、幂等与可观测，最终会重造一套“调度+状态机”。
- 自研调度器同样会重造“状态、路由、回流、可观测”的轮子，维护成本高且难做成可回归的契约。
- 状态持久化（现状澄清）：`multi_agents/agents/orchestrator.py` 未启用 LangGraph checkpointer（`MemorySaver` 被注释）；当前可落地的持久化主要是 `outputs/` 产物与 `multi_agents/agent_state.json`（tiers），不应宣称“可断点续跑”。可选升级：引入 `langgraph-checkpoint` 持久化 checkpoint 以支持断点恢复/回放。

**如何验证：**
- `tests/test_orchestrator_flow.py` 覆盖“审阅直接接受”与“修订后接受”的主流程分支。

### 3) HITL 回流怎么触发？为什么放在规划后、深研前？
**一句话结论：** 用 `human_feedback` 做条件路由：有反馈回 planner，无反馈进并行深研，把纠偏放在成本最低的点。

**实现落点：**
- `multi_agents/agents/orchestrator.py`：human 节点条件路由读取 `human_feedback` 是否为空，决定回 planner 还是进 researcher。
- `multi_agents/agents/human.py`：负责输出 `human_feedback`（实现里有 normalize）。

**为什么这样设计：**
- 规划阶段改章节边界会影响后续所有检索与写作；放到终稿才改会浪费整条链路的证据成本。

**利弊与边界：**
- 利：减少“写很长但不对题”的失败；用户能直接锁定方向与章节边界。
- 弊：依赖用户交互；反馈若是自然语言且含糊，会导致规划漂移（可选升级：结构化反馈 schema + 回流次数上限）。

**对照题库：** Q05（HITL）

**追问补充：用户改完后如何对齐旧计划（diff 还是全量）？**
- 当前实现是把自然语言 `human_feedback` 注入 planner prompt 后**全量重生成 outline**，不是 diff-based patch/局部修补。
- 可选升级：把反馈收敛为结构化 schema（比如“删/改/增某一章、约束每章边界、固定维度标签”），再做 outline diff/patch，从而降低漂移并提升可回归性（未实现）。

**如何验证：**
- 可用 `tests/test_orchestrator_flow.py` 的主链路分支思路做类似 human 分支测试（当前测试主要覆盖 reviewer 分支）。

### 4) Planner 如何产出 `section_details`（MECE+JSON）？如何校验/兜底？
**一句话结论：** 用强约束 prompt + JSON 输出 + schema 校验，把“可读大纲”变成“可执行计划”。

**实现落点：**
- `multi_agents/agents/editor.py`：规划 system prompt 强制“意图分类→维度拆解→MECE 章节→每章 key_points + research_queries”，并禁用 Introduction/Conclusion/References 等泛化章节。
- `gpt_researcher/utils/validators.py`：`ResearchOutline/SectionOutline` 定义了 `header/description/key_points/research_queries` 的结构约束。
- `tests/test_editor_planner.py`：覆盖章节去重、禁用泛化标题、`max_sections` 限制与空输出兜底。

**对照题库：** Q06（MECE）

**追问补充：如何约束 MECE、防重复/漏项？是否做自动 MECE 评估？**
- 约束手段：planner system prompt 明确 MECE（互斥/穷尽）并禁用泛化标题；代码侧做章节标题禁用与去重；输出走 `ResearchOutline` schema 校验，失败时走兜底解析以保持链路不断。
- 自动评估（现状）：当前没有“MECE 得分/coverage 指标”的自动评估或自动打分。
- 可选升级（未实现）：做章节间语义 overlap 检测（重复度阈值触发重写）；做 coverage checklist（关键维度/关键问题覆盖率）来发现遗漏并触发 planner 重试。

**为什么这样设计：**
- 不结构化就无法稳定并行、无法把章节级上下文注入后续检索、也无法做“缺口驱动”的回流。

**利弊与边界：**
- 利：可校验、可回归；后续节点只消费固定字段，降低提示工程复杂度。
- 弊：模型仍会格式漂移；当前靠解析/归一化兜底，极端情况下章节边界会变弱。

**如何验证：**
- 直接跑 `tests/test_editor_planner.py` 看“禁用标题 + 去重 + 限制章节数”的行为是否稳定。

### 5) 为什么并行单位选“章节”？并行结果如何聚合到 `ResearchState`？
**一句话结论：** 章节天然是交付结构；并行后把“草稿+证据+门禁结论”聚合回全局，供写作与终稿审阅消费。

**实现落点：**
- `multi_agents/agents/editor.py`：按 `section_details` 构造每章输入（topic + research_context），并行执行后聚合 `research_data/scrap_packets/check_data_reports`。
- `multi_agents/memory/research.py`：聚合字段在全局 state 中有固定位置，Writer/Publisher 直接消费。
- `tests/test_editor_planner.py`：有用例覆盖“空章节列表时退化为 1 章”与聚合 scrap/check_data 报告。

**为什么这样设计：**
- 如果按“搜索目标”并行，会把写作切得过碎，最终还要做大规模重组；章节并行更贴近最终报告结构。

**利弊与边界：**
- 利：吞吐高、边界清晰；章节可以天然并行且便于审阅。
- 弊：跨章重复与冲突更常见，需要终稿阶段做一致性兜底（可选升级：跨章去重/冲突检测）。

**如何验证：**
- `tests/test_editor_planner.py` 的并行 research 聚合用例能验证 `scrap_packets/check_data_reports` 的聚合契约。

### 11) Reviewer–Reviser 为什么不做事实校验？“上限 3 轮”如何实现与计数？
**一句话结论：** Reviewer 只做“可发布性”质检；事实风险用研究阶段 gate 控制；终稿修订用 `review_iterations/max_review_rounds` 做硬上限。

**实现落点：**
- `multi_agents/agents/reviewer.py`：在 `follow_guidelines=true` 时输出 review 反馈；返回 `None` 表示接受（实现里是检测 response 是否包含 “None”）。
- `multi_agents/agents/orchestrator.py`：`review` 为空则发布，否则进入修订；每次修订 `review_iterations += 1`，达到 `max_review_rounds`（默认 3）后直接发布当前最优稿。
- `tests/test_orchestrator_flow.py`：覆盖“修订后接受”；并有“达到 max_review_rounds 后强制发布”的回归用例。

**为什么这样设计：**
- 事实校验如果放在 Reviewer，会把“证据不足”掩盖成“文风优化问题”；先 gate 再审阅能把成本花在刀刃上。

**追问补充：怎么避免自嗨式循环修改？为什么最多 3 轮？**
- 现状机制是“软约束 + 硬上限”：
  - Reviewer prompt 倾向“only if critical… aim return None”，减少无意义的细碎迭代；接受条件是 `review is None`。
  - 修订循环硬上限由 `review_iterations/max_review_rounds` 控制（默认 3），达到上限后强制发布当前最优稿，避免无限循环。
- 边界（实现细节）：当前 Reviewer 的“接受”是通过检测返回文本是否包含 `None` 来判定，存在误判空间（例如模型输出里出现 “None” 但仍带其他意见）。如需更严格，应改为结构化输出（可选升级，未实现）。

**追问补充：质量 vs 成本曲线怎么量化？**
- 现状：仓库未内建在线 A/B，也没有把“每轮改进幅度”量化进路由。
- 建议（可做但不宣称已做）：用 `evals/` 做离线对照（固定样本，只改 `max_review_rounds` 或 reviewer 策略），观察质量指标（可读性/一致性/引用对齐）与成本指标（token/时延）的拐点，决定默认上限为何是 3。

**利弊与边界：**
- 利：质量/时延可控；避免无限修订循环。
- 弊：达到上限后仍可能存在未满足的指南项（但这是明确的工程折中）；如需更强保障应把部分指南前移到写作模板或加更细粒度 gate（可选升级）。

**如何验证：**
- 跑 `pytest -q tests/test_orchestrator_flow.py`，确认 reviewer/reviser 调用次数与最终发布稿符合上限策略。
