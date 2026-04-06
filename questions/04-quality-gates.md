# 质量门禁与校验

> 覆盖题号：Q13, Q27, Q28, Q30, Q33

### Q13（基础）：Check Data Agent 在校验什么？为什么需要 ACCEPT/RETRY/BLOCKED？

**标准答案（可直接说）：**
一句话：Check Data 是研究阶段的证据门禁，校验证据包是否满足主题关键约束；三态决策用于可控回流：通过 ACCEPT，不足 RETRY，超上限 BLOCKED。

展开：Check Data 从主题与研究上下文抽取原子化约束（主体/时间/指标等），对证据片段做硬守卫检查，并对“预测/估算”等高风险措辞做告警降分。综合得到分数后与阈值比较输出状态。RETRY 会生成更强约束提示用于下一轮证据采集；BLOCKED 用于在证据明显不足时保护系统不输出高风险结论。

常见坑/反杀点：
- 这是“校验驱动的检索回流”，不是“模型自反思”。
- 不要承诺零幻觉；正确口径是“降低风险并提供门禁”。

**相关模块（对应实现）：**
- multi_agents/agents/check_data.py
- multi_agents/agents/editor.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：Check Data 从 `scraping_packet` 提取证据片段，抽取原子化约束（subject/time/metric/negative constraints），先做硬守卫（缺主体/缺时间直接 hard fail），再算分并输出 `ACCEPT/RETRY/BLOCKED`，同时生成 `instruction/new_query_suggestion` 写入 `audit_feedback/extra_hints` 驱动下一轮（见 `multi_agents/agents/check_data.py`）。
- 为什么：把“证据不足”显式化并可控回流，避免在不可靠证据上继续写作与润色。
- 利弊：利是降低高风险输出；弊是启发式抽取与字符串匹配会误判（误放行/误阻断），复杂推理或数字计算错误仍可能漏掉。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。
> 本题映射：示例里如果证据与口径不匹配（时间窗/地域/指标缺失），就应触发 RETRY 回到补检索；证据明显不足则 BLOCKED，避免输出高风险结论。

**定义：** Check Data、ClaimVerifier、Reviewer 的分工？
答题要点：Check Data 面向章节证据与口径；ClaimVerifier 面向 Writer 断言级引用与冲突；Reviewer 面向终稿可发布性、指南满足与 source-aware 修订审查。

**实现：** 回流怎么触发？  
答题要点：章节级 workflow 读取 `check_data_action`，`retry` 回到研究节点，`accept/blocked` 结束该章。

**边界：** 它捕获不到什么？  
答题要点：细粒度事实错误、复杂推理错误、数字计算错误；需更强校验（可选升级）。

**取舍：** 为什么不只靠“谨慎写作提示”？  
答题要点：谨慎不等于有证据；gate 将证据不足显式化并驱动补检索。

**优化：** 如何升级为断言级对齐？  
答题要点：抽取 claims、做支持/矛盾/未知判定、绑定引用跨度并以缺口驱动 RETRY（可选升级）。

---

### Q27（复杂）：Check Data 的“原子化约束抽取 + 硬守卫 + 可疑措辞告警 + 分数阈值”怎么设计？误判怎么处理？

**标准答案（可直接说）：**
一句话：先抽取主题关键约束（主体/时间/指标），用硬守卫保证关键约束出现，再对“预测/估算”等高风险措辞做告警降分，最后用阈值与重试上限输出 ACCEPT/RETRY/BLOCKED；误判通过更具体的重试提示与上限保护缓解，并可升级为断言级对齐（可选）。

展开：硬守卫用于尽早拦截明显跑题/时间错配；可疑措辞告警用于避免把预测当事实；阈值把多维信号压缩成稳定路由决策。误判不可避免：过严会造成不必要重试，过松会放过问题证据。工程上通过最大重试次数限制与反馈包（缺失项/建议查询）控制风险；进一步可升级为语义级校验与来源权重。

常见坑/反杀点：
- 不要宣称“保证零幻觉”；正确说法是“降低风险并提供门禁”。
- 面试官会挑战“字符串命中太弱”；要强调这是轻量 gate 并给出升级路线。

**相关模块（对应实现）：**
- multi_agents/agents/check_data.py
- multi_agents/memory/draft.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：Check Data 分三层信号：原子化约束抽取（subject/time/metric/negative）、硬守卫（缺主体/缺时间 hard fail）、可疑措辞告警（projection/forecast 且无 actual/audited），最后按 `PASS_THRESHOLD` 给出 ACCEPT/RETRY/BLOCKED（见 `multi_agents/agents/check_data.py` 常量与 guard/eval 流程）。
- 为什么：先用硬规则挡掉明显不匹配，再用分数做“可控回流”，让系统能在证据不足时自动补检索。
- 利弊：利是实现简单、可解释；弊是规则/词表会导致误判，且对“复杂推理正确但未显式出现关键词”的证据不友好，需要更强对齐（可选升级）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 什么叫硬守卫？  
答题要点：对关键约束做必须满足的检查（主体/时间缺失直接判失败倾向）。

**实现：** 为什么要对“预测/估算”做特殊处理？  
答题要点：这类表述易误导；缺少 audited/official/actuals 线索时应谨慎。

**边界：** 最容易误判的场景？  
答题要点：同义表达、简称、跨语言、时间描述多样、正文隐含约束不显式。

**取舍：** 为什么先做轻量 gate？  
答题要点：在线成本可控；先过滤明显问题，深校验只在必要时触发（可选升级）。

**优化：** 升级到更强校验怎么做？  
答题要点：claims 抽取 + NLI/LLM-judge 对齐 + 引用跨度绑定 + 缺口驱动 RETRY（可选升级）。

---

### Q28（复杂）：RETRY 的“更强约束提示”如何生成、如何驱动下一轮检索？为什么能减少漂移？

**标准答案（可直接说）：**
一句话：RETRY 时把失败原因转成更具体的检索约束与建议查询，写入章节 state 的提示字段，驱动下一轮 ASA 的目标分解与检索，使下一轮定向补缺口而不是重复抓取。

展开：校验阶段输出反馈包（缺失约束、建议查询、避免高风险措辞等），与历史提示合并形成下一轮输入的一部分。ASA 在生成搜索目标时会把提示转成明确约束，从而改变下一轮检索目标与引擎组合，提升信息增益并减少长链漂移。相比“无差别扩大 top-k”，缺口驱动能更快收敛到可验证证据。

常见坑/反杀点：
- 不要说成“自动修正事实”；它是“定向补证据”。
- 当前提示更偏文本合并；结构化指令属于可选升级。

**相关模块（对应实现）：**
- multi_agents/agents/check_data.py
- multi_agents/agents/scraping.py
- multi_agents/memory/draft.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现（链路怎么“回写→合并→再消费”）：
  - Check Data 将本轮失败点结构化为 `check_data_verdict.feedback_packet`（缺失约束/纠偏指令/建议检索式等），并把可执行部分写回章节 state：`check_data_action="retry"`、`iteration_index` 递增、`audit_feedback` 写入 `instruction/new_query_suggestion/confidence_score`、`extra_hints` 写入“指令 + 建议检索式”的合并文本（见 `multi_agents/agents/check_data.py`、`multi_agents/memory/draft.py`）。
  - 章节级 workflow 读取 `check_data_action` 做条件分支：`retry` 回到证据采集节点，`accept/blocked` 结束该章（见 `multi_agents/agents/editor.py`）。
  - ASA 在下一轮生成 2-4 个 search targets 时，会把 `audit_feedback` 与 `extra_hints` 一起注入提示，要求“把提示转成显式检索约束”，从而让 targets 更定向地补缺口（例如强制包含目标年份/主体/指标、加入 `actual/audited`，并排除 `projection/forecast` 等高风险措辞）（见 `multi_agents/agents/scraping.py`、`multi_agents/agents/check_data.py`）。
  - 引擎组合也会随轮次变化：首轮偏通用覆盖，进入重试轮次后按领域路由到更专业的引擎集合，并对垂直引擎失败做通用引擎回退，配合更强约束提升信息增益（见 `multi_agents/agents/scraping.py`）。
- 为什么：用“缺口驱动”的定向补检索替代盲目扩大 top-k，能更快收敛并减少漂移。
- 利弊：利是收敛更快、成本更可控；弊是缺口抽取若不准会把系统带偏，需要把失败样本进入回归集迭代词表/提示。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 为什么缺口驱动比扩大抓取量更好？  
答题要点：更高信息增益、更少噪声、更快收敛。

**实现：** 反馈提示通常包含哪些类别？  
答题要点：缺失约束、建议查询、纠偏指令，并与上一轮提示合并进入下一轮输入。

**边界：** 提示可能带来哪些负面效果？  
答题要点：过度约束导致漏检；错误提示导致偏离；需要上限与 HITL 兜底。

**取舍：** 为什么不直接让 Planner 重新规划？  
答题要点：这是章节内证据不足，优先补证据；章节结构问题才回规划阶段。

**优化：** 如何把提示升级为结构化指令？  
答题要点：输出 `{must_include, must_exclude, time_window, preferred_sources, query_templates}` 让检索模块直接消费（可选升级）。

---

### Q30（复杂）：你们如何处理“时效性/时间约束”（年份、季度、latest）与错误对齐？当前实现边界在哪？

**标准答案（可直接说）：**
一句话：当前实现主要通过 Check Data 抽取并守卫时间约束（如年份/季度），要求证据语料出现对应时间线索；对更复杂的“最新/滚动更新”仍有边界，需要更强的发布时间元数据与时间窗过滤（可选升级）。

展开：时效性错配常见于“最新进展、最近一季”等查询。如果证据语料不包含正确时间线索，就容易把旧信息当新信息。当前仓库在校验阶段会从主题中抽取年份/季度线索并检查语料是否出现，缺失则触发重试或降分/阻断。边界在于时间表达多样、抓取元数据缺失、以及检索结果偏旧。生产级可引入发布时间元数据、时间窗过滤、按时间重排与章节级“背景/最新”分层策略。

常见坑/反杀点：
- 不要宣称“保证最新”；正确口径是“有时间守卫与缺口回流”。

**相关模块（对应实现）：**
- multi_agents/agents/check_data.py
- multi_agents/agents/scraping.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：规划与写作提示中显式注入“今天日期”（见 `multi_agents/agents/editor.py`、`multi_agents/agents/writer.py`），Check Data 会从 topic 中用正则抽取年份/季度并作为硬守卫信号之一（见 `multi_agents/agents/check_data.py` 的 time_constraint 抽取）。
- 为什么：Web 证据时效性强，必须把时间窗当成约束，否则容易“老数据套新问题”。
- 利弊：利是能挡掉明显时间错配；弊是当前时间约束抽取较粗（主要看年/季度字符串），遇到“latest/近三月/滚动窗口”等口径仍需要更精细的元数据与重排（可选升级）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 时效性错配的典型症状？  
答题要点：引用旧年份/旧季度或混用多个时期数据导致矛盾。

**实现：** 现在如何做时间约束守卫？  
答题要点：抽取年份/季度线索并要求在语料中出现，缺失触发重试或降分。

**边界：** 为什么仍可能失败？  
答题要点：时间表达不统一、正文不显式日期、元数据缺失、检索偏旧。

**取舍：** 为什么不强制只用最近 X 天来源？  
答题要点：会误伤背景/定义内容；更合理是章节分层：背景允许旧、最新进展强约束（可选升级）。

**优化：** 更强的时间一致性方案？  
答题要点：发布时间元数据、时间窗过滤、时间重排、写作阶段显式标注时间范围与口径（可选升级）。

---

### Q33（复杂）：当前仓库已经有 ClaimVerifier；如果把这种“断言级证据对齐（claim-evidence alignment）”进一步前移到 Check Data，你会怎么改数据结构与路由？

**标准答案（可直接说）：**
一句话：当前仓库已在 Writer 之后通过 `ClaimVerifier` 做断言级引用校验；如果继续把这套能力前移到 Check Data，就需要把章节证据也拆成可验证 claims，并基于“未支撑/矛盾 claim 比例”更早驱动补检索。

展开：当前实现里，Writer 会输出 `claim_annotations`，ClaimVerifier 会基于 append-only `source_index` 对引言/结论做 `HIGH/MEDIUM/SUSPICIOUS/HALLUCINATION` 分级，并在 `SUSPICIOUS` 时触发按章节的 Reflexion 补检索。若继续把这套能力前移到 Check Data，可分三步：先在章节证据包里保留更结构化片段（句子/段落级 + 来源 URL + 元数据），再把章节草稿拆成 claims 并做支持/矛盾/未知判定，最后把缺口清单与建议查询模板直接写回 `DraftState`，让 `check_data_action` 更早基于 claim 级结果触发 RETRY/BLOCKED/ACCEPT。

常见坑/反杀点：
- 要明确“Writer 后 ClaimVerifier 已是仓库现状”，可选升级的是“把同类能力前移到 Check Data 阶段”。
- 面试官会追问成本：要讲分层校验与只对关键断言深校验。

**相关模块（对应实现）：**
- multi_agents/agents/check_data.py
- multi_agents/agents/claim_verifier.py
- multi_agents/agents/scraping.py
- evals/hallucination_eval/

**技术细节（实现 / 为什么 / 利弊）：**
- 现状：Check Data 仍主要是“原子约束 + 字符串匹配 + 分数阈值”的章节级 gate；但 Writer 后已经有 `ClaimVerifier` 做断言级引用解析、跨域佐证、冲突检测与 `section_key` 级 Reflexion 路由（见 `multi_agents/agents/check_data.py`、`multi_agents/agents/claim_verifier.py`）。
- 升级实现（可选升级）：在 `DraftState`/`ResearchState` 增加 `claims`（列表）与 `claim_evidence`（支持/矛盾/未知 + 引用片段/URL），把“未知/矛盾 claim 触发 RETRY 并生成定向查询”的规则直接前移到章节门禁，让 Writer 进入时拿到更干净的证据集合。
- 利弊：利是可验证性显著提升；弊是抽取/对齐本身成本高且易错，需要更强的评测与人工抽检闭环。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 相比关键词守卫的提升点？  
答题要点：覆盖细粒度事实、可解释缺口、适合做质量门禁。

**实现：** 数据结构要新增什么？  
答题要点：claims 列表、每条 claim 的证据引用（source_url + 片段/跨度）、判定结果与置信度。

**边界：** 哪类 claim 难自动对齐？  
答题要点：跨文档综合推理、统计结论、隐含前提；需要证据图或人评。

**取舍：** 为什么不对所有句子都做对齐？  
答题要点：成本高；只对数字/结论句等关键断言深校验更划算。

**优化：** 如何既强又便宜？  
答题要点：分层筛选（规则→NLI→judge）、缓存向量、批量判定、只在失败样本启用深校验。

---

## 附：Top12 对应条目（从原附录迁移）

### 9) Check Data gate 具体怎么判 ACCEPT/RETRY/BLOCKED？阈值/硬守卫/告警是什么？
**一句话结论：** 先硬守卫“主体/时间”再打分，`PASS_THRESHOLD=0.7`；不足就 RETRY，超过重试上限就 BLOCKED。

**实现落点：**
- `multi_agents/agents/check_data.py`：`PASS_THRESHOLD=0.7`、`MAX_RETRIES=3`；抽取原子约束（subject/time/metric/negative），缺主体/缺时间属于 hard fail，会把分数 cap 到 `<0.7`（实现里是 `0.69`）。
- 高风险措辞告警：命中 projection/forecast/预计/预测/预估 且未命中 actual/audited/official filing 等“实绩”词时降分并进入缺口提示。
- 输出契约：写入 `DraftState.check_data_action`（accept/retry/blocked）与 `check_data_verdict`，并在 RETRY 时把 `extra_hints` 带回下一轮。
- `tests/test_check_data_agent.py`：覆盖缺约束 RETRY、满足约束 ACCEPT、第三次失败 BLOCKED、以及 `extra_hints` 传播。

**对照题库：** Q13（Check Data 三态门禁）

**追问补充：原子化 constraint 结构是什么？如何抓“预测/估算”风险？**
- constraint 结构（来自实现）：`{subject, time_constraint, metric, negative_constraints}`。hard guard 先要求 `subject/time_constraint` 在证据片段语料中可命中，否则直接 hard fail。
- “预测/估算”风险识别（现状）：token-set 告警。`SUSPICIOUS_TOKENS`（projection/forecast/预计/预测/预估等）命中且 `ACTUAL_TOKENS`（actual/audited/reported/official filing 等）不命中则标记 suspicious，并把分数 cap 到 `<0.7`。

**追问补充：ACCEPT/RETRY/BLOCKED 决策逻辑与误判**
- 决策逻辑（现状）：按 `PASS_THRESHOLD=0.7` 判 ACCEPT；未达标则在 `MAX_RETRIES` 内走 RETRY；超过上限走 BLOCKED。
- 常见误判来源：
  - false negative：同义改写/指代替换导致“词面不命中但事实存在”。
  - false positive：词面命中但上下文是否定/反例，或命中不等于“支持该断言”。
  - 复杂推理/数值计算错误：当前字符串匹配无法覆盖。
- 现有缓解：RETRY 生成更强约束提示（`instruction/new_query_suggestion`）并注入 `extra_hints`，让下一轮定向补缺口；同时有重试上限防止无限回流。
- 可选升级（未实现）：rule-based guard + LLM judge/NLI 对齐（支持/矛盾/未知）做 hybrid 决策，并把“未知/矛盾的 claim”转成下一轮定向检索约束。

**为什么这样设计：**
- 研究链路里最贵的是“写作与审阅”；先 gate 能让证据不足尽早暴露，避免在不可靠证据上打磨。

**利弊与边界：**
- 利：把证据不足变成可控回流，降低高风险结论输出概率。
- 弊：启发式抽取与字符串匹配会误判；复杂推理/数值计算错误不一定能抓到（可选升级：断言级对齐）。

**如何验证：**
- 直接跑 `tests/test_check_data_agent.py`，看三态决策与阈值行为是否一致。

### 10) RETRY 的“更强约束提示”怎么生成、怎么注入下一轮（`extra_hints`）？
**一句话结论：** 把缺口写成 `feedback_packet`（instruction + new_query_suggestion），合并成 `extra_hints` 驱动下一轮 targets 与查询约束。

**实现落点：**
- `multi_agents/agents/check_data.py`：RETRY 时生成 `feedback_packet`，并把 `instruction/new_query_suggestion` 写入 `audit_feedback` 与 `extra_hints`。
- `multi_agents/agents/scraping.py`：每轮会合并 `extra_hints` 与 `audit_feedback`，并在“目标分解（3 targets）”时把 hints 转成明确检索约束。
- `tests/test_editor_check_data_workflow.py`：覆盖“第一次 RETRY 生成更强约束→第二次 ACCEPT”的闭环行为。

**为什么这样设计：**
- 相比无差别扩大 top-k，缺口驱动能把下一轮检索变成“定向补证据”，更快收敛。

**利弊与边界：**
- 利：收敛更快、预算更可控；回流理由可解释（哪些约束缺失）。
- 弊：如果缺口抽取不准，会把系统带偏；需要把失败样本进入回归集迭代缺口模板（可选升级）。

**如何验证：**
- 跑 `tests/test_editor_check_data_workflow.py` 看 RETRY→ACCEPT 是否按预期传播 hints。
