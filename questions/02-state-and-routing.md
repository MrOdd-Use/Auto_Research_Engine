# 状态建模与路由

> 覆盖题号：Q07, Q21, Q21a

### Q07（基础）：“结构化大纲”包含哪些字段？为什么要结构化而不是纯文本？

**标准答案（可直接说）：**
一句话：结构化大纲把“章节标题 + 边界描述 + 关键点 + 章节研究查询”固定为 JSON，便于路由、并行执行与可校验。

展开：纯文本大纲很难稳定被程序消费，容易出现字段缺失、章节重复、不可并行等问题。本项目在规划阶段输出 `section_details`，典型字段包括：`header`、`description`、`key_points`、`research_queries`。这些字段直接进入章节深研阶段，指导证据采集与写作聚焦，同时也便于 HITL 展示与用户修改。

常见坑/反杀点：
- 只说“方便解析”；更要强调“能驱动后续动作与评测”。
- 结构化不是万能：需要 schema 校验与降级策略。

**相关模块（对应实现）：**
- multi_agents/agents/editor.py
- multi_agents/memory/research.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：Planner 以 JSON 形式返回 `section_details`（header/description/key_points/research_queries），并用 `ResearchOutline` 做结构校验与归一化（见 `gpt_researcher/utils/validators.py`、`multi_agents/agents/editor.py`）。
- 补充：当前实现对 `research_queries` 的要求是“字段名固定 + 值为字符串列表”；运行期会做列表去重与清洗，但不会把 `queries/search_queries` 这类别名自动映射为 `research_queries`。若规划输出不符合 schema，会走兜底路径并可能得到空 `research_queries`。
- 为什么：结构化结果才能被后续节点稳定消费（并行、路由、校验、HITL 修改），避免纯文本难解析/易漂移。
- 利弊：利是可校验、可回流；弊是模型可能格式漂移，需要容错解析/降级（当前实现包含归一化与兜底章节）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例卡片里的 `section_details` 展示了结构化大纲长什么样：每章有 header/description/key_points/research_queries，便于路由、并行与可校验。

**定义：** 结构化输出解决了什么核心问题？  
答题要点：稳定消费、可并行执行、可校验、可回流修改。

**实现：** 如何保证输出字段可用？  
答题要点：要求 JSON 输出并做解析/校验/去重；不合规时走降级路径补齐默认字段。

**边界：** 结构化输出仍可能失败在哪？  
答题要点：模型格式漂移、字段语义漂移；需要容错与回退。

**取舍：** 为什么不让研究阶段自己决定要搜什么？  
答题要点：覆盖不可控且易重复；规划集中决策更利于全局结构与并行。

**优化：** 如何提升结构化输出稳定性？  
答题要点：更严格 schema、分步生成、低温度、失败样本回归集（可选升级）。

---

### Q21（复杂）：你们如何建模状态（`ResearchState` / `DraftState`）？哪些字段用于路由与回流？

**标准答案（可直接说）：**
一句话：全局链路用 `ResearchState` 表达，章节级并行深研用 `DraftState` 表达；路由与回流主要依赖 `human_feedback`、ClaimVerifier/Reviewer 的反馈信号、以及 Check Data 输出的 `accept/retry/blocked` 等信号。

展开：`ResearchState` 承载端到端流程所需的关键信息：初始资料收集结果、章节列表与结构化详情、并行深研结果、写作生成的布局要素、`source_index` / `claim_annotations` / `claim_confidence_report` 等事实性状态、审阅/修订信息、最终稿与发布内容等。`DraftState` 承载单章节研究输入（topic、研究上下文、额外约束提示）与输出（草稿、证据包、校验结论、审阅意见、修订说明）。路由上：HITL 依据 `human_feedback`；claim reflexion 依据可疑断言映射到 `section_key`；终稿审阅依据 `review` 是否为空；章节研究闭环依据 `check_data_action`。

常见坑/反杀点：
- 只讲概念不讲字段；面试官会追问 state 里到底有什么。
- state 不结构化会导致节点耦合与分支难维护；要强调 schema 的价值。

**相关模块（对应实现）：**
- multi_agents/memory/research.py
- multi_agents/memory/draft.py
- multi_agents/agents/orchestrator.py
- multi_agents/agents/editor.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：全局状态用 `ResearchState` 保存端到端产物（章节详情、并行研究结果、审阅/修订、发布稿等），章节状态用 `DraftState` 保存单章闭环信息（topic/context/iteration/check_data_action/scrap_packet/draft 等）（见 `multi_agents/memory/research.py`、`multi_agents/memory/draft.py`）。
- 为什么：全局 state 负责路由与聚合，章节 state 负责并行与局部回流，降低节点耦合。
- 利弊：利是并行与回放更清晰；弊是 state 膨胀与字段语义漂移风险，需要摘要化与 schema 版本化（可选升级）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例里会进入 state 的典型信息包括：追问收敛后的约束、`section_details`、各章证据包与门禁结果、审阅反馈与终稿/发布内容。

**定义：** 为什么要拆成全局 state 与章节 state？  
答题要点：全局管路由与汇总；章节管并行与局部闭环；降低耦合、提高可扩展。

**实现：** 哪些字段最关键？  
答题要点：全局有 `section_details/research_data/source_index/claim_confidence_report/final_draft/report` 等；章节有 `topic/research_context/extra_hints/check_data_action/draft/review` 等。

**边界：** state 过大有什么问题？  
答题要点：序列化成本高、调试难、隐私风险；需要分层与摘要化（可选升级）。

**取舍：** 为什么不让每个 agent 私有存储？  
答题要点：私有存储会导致不可观测与不可路由；共享 state 更利于协调与回归评测。

**优化：** 如何让 state 更适合回放与审计？  
答题要点：结构化事件日志、版本化 schema、持久化到数据库、支持回放与差异比对（可选升级）。

---

### Q21a（复杂）：`DraftState` / `ResearchState` 的字段分别是什么意思？（逐字段速查）

**标准答案（可直接说）：**
一句话：`ResearchState` 是端到端全局状态（规划→并行深研聚合→写作布局→审阅修订→发布），`DraftState` 是单章节闭环状态（输入约束→证据包→门禁→草稿→审阅修订）。字段不是一次性填满，而是沿工作流逐步写入。

**字段解释：`DraftState`（单章）**（见 `multi_agents/memory/draft.py`）
- `task`：全局任务配置（query、模型、是否 verbose、参数上限、发布格式等）。
- `topic`：当前章节标题/研究主题。
- `iteration_index`：当前重试轮次（用于“补检索/补证据”的迭代计数）。
- `research_context`：规划阶段注入的章节上下文（常见含 `description/key_points/research_queries`）。
- `extra_hints`：额外约束提示（常由门禁失败生成，用于下一轮更定向检索）。
- `audit_feedback`：门禁/满意度信号包（如 `is_satisfied/confidence_score/instruction/new_query_suggestion`），用于驱动是否进入下一轮。
- `scrap_packet`：证据包与过程记录（例如每个 search target 的 top passages、使用的引擎、轮次信息等）。
- `check_data_action`：章节门禁的路由动作（典型：`accept` / `retry` / `blocked`）。
- `check_data_verdict`：门禁判定报告（包含约束抽取、命中/缺失原因、分数、反馈包等）。
- 补充：`check_data_verdict.atomic_claims` 是“原子化约束”（常见字段名为 `subject` / `time_constraint` / `metric` / `negative_constraints`），用于把“本章必须对齐的口径”变成可校验的最小集合：
  - `subject`：本章核心主体（公司/机构/地区/产品等），优先从 `topic` 与 `research_context.key_points` 的实体词抽取。
  - `time_constraint`：目标年份/季度（如 `2023`、`2023Q2`），通常从 `topic` 里抽取。
  - `metric`：关注指标（如 `revenue`/`market share`/`营收`），通常从 `topic/description/key_points` 的指标词表命中。
  - `negative_constraints`：排除项（如排除非目标年份、排除 `projection/forecast` 等高风险措辞）。
  - 范例（示意）：`topic="Nvidia revenue 2023"` 且 key points 提到 “revenue” 时，可能抽到：`atomic_claims={subject:Nvidia, time_constraint:2023, metric:revenue, negative_constraints:[\"2024\",\"2025\",\"projection\",\"forecast\"]}`（具体以实现与输入为准）。
- 产生方式说明：该抽取是轻量的规则/启发式（正则 + 关键词词表 + 兜底策略），用于“门禁与回流”的硬约束信号；抽取不到时字段可能为空，需要通过后续 RETRY 的更强提示/补检索来弥补（见 `multi_agents/agents/check_data.py`）。
- `draft`：该章节的草稿文本（通常是 dict，把 `topic` 映射到章节内容；blocked 时可能是“证据不足占位稿”）。
- `review`：章节级审阅意见（为 `None` 通常表示通过；非空表示需要修订）。
- `revision_notes`：修订说明（告诉审阅者“我按哪些意见改了什么”）。

**这些字段是如何产生的？（单章闭环，口述版）**
一般是先由章节级调度把输入拼成一份 `DraftState`：带上全局 `task`、本章 `topic`、初始 `iteration_index=1`，以及 Planner 给该章的 `research_context`（description/key_points/research_queries）（见 `multi_agents/agents/editor.py`、`multi_agents/memory/draft.py`）。随后进入证据采集阶段，产出 `scrap_packet`（证据片段/来源/引擎与轮次信息），并在此基础上形成该章 `draft`（见 `multi_agents/agents/scrap.py`）。接着校验阶段会读取 `scrap_packet` 与 `research_context`，生成 `check_data_verdict`（含 `atomic_claims`、缺口与评分等）并写入 `check_data_action`：如果需要补证据则把缺口与建议查询写回到 `audit_feedback` 与 `extra_hints`、同时递增 `iteration_index` 触发下一轮；如果通过则保留当前草稿与证据包并结束；如果证据明显不足则输出占位稿并结束（见 `multi_agents/agents/check_data.py`）。若启用审阅/修订链路，Reviewer 会写入 `review`，Reviser 会基于 `review` 产出修订版 `draft` 并写入 `revision_notes`（见 `multi_agents/agents/reviewer.py`、`multi_agents/agents/reviser.py`）。

**字段解释：`ResearchState`（全局）**（见 `multi_agents/memory/research.py`）
- `task`：全局任务配置（同上）。
- `initial_research`：初始资料收集产物（用于规划阶段生成大纲）。
- `sections`：扁平章节列表（兼容字段；只含标题字符串）。
- `section_details`：结构化章节列表（每章包含 header/description/key_points/research_queries）。
- `human_feedback`：人类反馈（HITL）：为空/None 表示接受大纲；非空表示回流重规划。
- `audit_feedback_queue`：可选的门禁反馈队列（用于给并行章节提供不同的起始约束/提示）。
- `extra_hints`：全局额外提示（会被注入到各章节的检索/写作约束里）。
- `research_data`：并行章节产物的聚合列表（每项通常是“章节标题→章节草稿”的 dict）。
- `scrap_packets`：并行章节证据包的聚合列表（用于审计/复盘/引用）。
- `check_data_reports`：并行章节门禁报告的聚合列表（用于质量追踪与调参）。
- 补充：原子化约束抽取（`subject/time_constraint/metric/negative_constraints`）是否在 `ResearchState`？不作为顶层字段单独存，但会包含在 `check_data_reports` 的每条章节门禁报告里（每章 `DraftState.check_data_verdict.atomic_claims` 聚合到全局）。
- `title`：报告标题。
- `date`：报告日期字符串。
- `headers`：报告的版式标题文案（如 title/date/introduction/table_of_contents/conclusion/references 的展示标题）。
- `table_of_contents`：目录（markdown 文本）。
- `introduction`：引言（markdown 文本）。
- `conclusion`：结论（markdown 文本）。
- `sources`：参考来源列表（通常是 URL 或引用条目字符串）。
- `claim_annotations`：Writer 输出的断言-引用映射（每条 factual claim 对应哪些 `S*` source IDs）。
- `source_index`：append-only 证据索引，形如 `S1 -> {content, source_url, domain, section_key}`，供 Writer、ClaimVerifier、Reviewer 共用。
- `indexed_research_data`：把 `source_index` 渲染成 Writer 可直接消费的带 `[S*]` 证据文本。
- `claim_confidence_report`：ClaimVerifier 产出的断言级校验结果（包含 `confidence/note/source_ids/source_urls` 等）。
- `claim_reflexion_iterations`：ClaimVerifier/Reflexion 触发的补检索轮次计数。
- `final_draft`：终稿正文（审阅/修订循环中的当前版本）。
- `review`：终稿审阅意见（为 `None` 表示通过；非空表示需要修订）。
- `revision_notes`：终稿修订说明（用于下一轮审阅的上下文）。
- `review_iterations`：终稿审阅-修订已循环的次数（用于上限保护）。
- `_draft_before_revision`：Reviewer 做 diff-based source audit 时，用来对比 Reviser 修改前后的终稿版本。
- `report`：发布阶段输出的最终报告文本（通常为完整 markdown）。

**这些字段是如何产生的？（全局链路，口述版）**
全局流程从 `task` 启动，先在初始研究阶段写入 `initial_research`，供后续规划使用（见 `multi_agents/agents/orchestrator.py`）。规划阶段会基于初始研究生成章节结构与可执行细节，写入 `section_details`（并保留兼容字段 `sections`），同时补齐 `title/date` 等全局信息（见 `multi_agents/agents/editor.py`、`multi_agents/memory/research.py`）。如果开启 HITL，`human_feedback` 用来决定是否接受大纲或回流重规划（见 `multi_agents/agents/orchestrator.py`）。进入并行深研后，每章会跑各自的章节闭环，再把章节草稿聚合到 `research_data`，证据包聚合到 `scrap_packets`，门禁报告聚合到 `check_data_reports`（见 `multi_agents/agents/editor.py`）。Writer 会在聚合结果之上生成引言/结论/目录/来源、`claim_annotations` 与版式标题等内容，写入 `introduction/conclusion/table_of_contents/sources/claim_annotations/headers`（见 `multi_agents/agents/writer.py`、`multi_agents/memory/research.py`）。随后 ClaimVerifier 会构建 `source_index`、刷新 `indexed_research_data`、输出 `claim_confidence_report`，并在必要时触发按章节的 reflexion rerun（见 `multi_agents/agents/claim_verifier.py`、`multi_agents/agents/orchestrator.py`）。终稿层面再进入审阅/修订循环：`review/revision_notes/review_iterations/_draft_before_revision` 记录审阅意见、修订说明、回合数与 diff 对照基线，`final_draft` 保存当前可发布版本（见 `multi_agents/agents/orchestrator.py`、`multi_agents/agents/reviewer.py`、`multi_agents/agents/reviser.py`）。最后 Publisher 生成完整布局写入 `report` 并导出多格式文件（见 `multi_agents/agents/publisher.py`）。

常见坑/反杀点：
- `sections` vs `section_details`：前者是“仅标题”的旧结构，后者是可执行的结构化大纲；面试要说清楚两者关系。
- `headers`：这里指“报告版式标题文案”，不是 HTTP 请求头。

---

## 附：Top12 对应条目（从原附录迁移）

### 2) `ResearchState` / `DraftState` 分别存什么？哪些字段参与路由？
**一句话结论：** 全局 state 管聚合与终稿，章节 state 管单章证据闭环；路由只看少量“信号字段”。

**实现落点：**
- `multi_agents/memory/research.py`：`ResearchState`（全局）含 `human_feedback`（HITL 路由）、`claim_confidence_report/source_index`（Writer 后事实性状态）、`review`（终稿接受/修订）、`review_iterations`（终稿上限）、以及聚合字段 `scrap_packets/check_data_reports`。
- `multi_agents/memory/draft.py`：`DraftState`（章节）含 `check_data_action`（accept/retry/blocked）、`iteration_index`、`extra_hints`、`scrap_packet`、`check_data_verdict`。
- `multi_agents/agents/editor.py`：章节级 workflow 读取 `check_data_action` 决定重试还是结束该章。

**为什么这样设计：**
- 把“并行单位”限制在章节，避免全局 state 被每个搜索细节污染；路由字段尽量少，降低分支爆炸。

**追问补充：回流重跑是全量还是增量？状态如何复用/回滚？**
- HITL 回流是“规划节点重跑”：有 `human_feedback` 时回到 planner，但不会回到 `browser`（初始研究复用），因此不是全链路全量重跑。
- 章节证据回流发生在 `DraftState` 内：`check_data_action=retry` 会路由回章节 researcher/scrap，`iteration_index` 递增；ClaimVerifier 的 `SUSPICIOUS` 断言也会按真实 `section_key` 触发章节级 rerun，并在重跑后只追加新的 evidence 到 `source_index`。
- `Rerun from Checkpoint` 是另一条更广义的回放链路：`multi_agents/workflow_session.py` 会持久化 global/section checkpoints，手动 rerun 会从选中的 checkpoint 恢复 `state_before` 并重算下游路径，同时保留 parent/child session 历史与 `last_successful_session_id`。
- 缓存现状：ASA `ScrapAgent` 没有跨轮 evidence cache；每轮仅做 URL 归一化去重与抓取物化，再进入分段与 MMR 选片。可选升级：跨轮缓存“已抓取正文/已选片段”并做内容级 canonical，以减少重复抓取与同质证据。

**利弊与边界：**
- 利：并行/回流粒度清晰；调试时能定位“是某一章证据不足”还是“终稿审阅不通过”。
- 弊：跨章节一致性需要额外处理（冲突结论、重复证据）（可选升级）。

**如何验证：**
- `tests/test_editor_check_data_workflow.py` 覆盖章节级 RETRY/ACCEPT/BLOCKED 分支与 `check_data_action` 路由契约。
