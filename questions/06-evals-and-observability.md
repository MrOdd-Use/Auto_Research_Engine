# 评测与可观测性

> 覆盖题号：Q17, Q18, Q20, Q32, Q34

### Q17（基础）：你们怎么做 benchmark？SimpleQA 和 hallucination eval 各测什么？

**标准答案（可直接说）：**
一句话：SimpleQA 评估短事实问答正确性（Accuracy/F1/Answer rate 等），hallucination eval 评估输出与来源语料的对齐性（是否出现非事实内容），两者形成回归门禁。

展开：SimpleQA 使用固定问题集与标准答案，通过独立评分模型给输出分级（CORRECT/INCORRECT/NOT_ATTEMPTED）并统计 Accuracy、F1、成本、来源数等。Hallucination eval 将研究输出与系统收集到的 source text 对照，由 judge 模型判定是否出现不对齐内容并给理由。它们分别覆盖“任务正确性”和“证据对齐性”，适合在策略迭代时做对照与回归。

常见坑/反杀点：
- “自己评自己”偏差：要强调评测模型可独立配置，评测与研究解耦。
- hallucination eval 依赖 source text 质量；source 缺失会导致跳过（要说明边界）。

**相关模块（对应实现）：**
- evals/simple_evals/
- evals/hallucination_eval/
- evals/README.md

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：短事实性用 `evals/simple_evals`（SimpleQA 题集 + LLM grader）做回归；长文幻觉用 `evals/hallucination_eval` 把输出与 source materials 对照判定（见 `evals/README.md`、`evals/simple_evals/*`、`evals/hallucination_eval/*`）。
- 为什么：把“研究系统的质量”拆成可度量维度：短答事实性 vs 长文对齐与编造率。
- 利弊：利是能做回归与对照组；弊是 grader 有偏差、题集覆盖有限，且不同检索/抓取条件会影响可重复性。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** benchmark 的价值是什么？  
答题要点：量化质量、对比策略、做回归门禁，防止“改了策略但悄悄变差”。

**实现：** SimpleQA 输出哪些指标？  
答题要点：分级统计与 Accuracy/F1，以及成本、来源数、成功率等（见 `evals/simple_evals/`）。

**边界：** SimpleQA 的局限？  
答题要点：偏短事实；对长文结构与一致性覆盖有限，需要补充长文评测/人评（可选）。

**取舍：** 为什么 hallucination eval 用 judge？  
答题要点：可规模化自动化；但存在 judge 偏差，需要交叉验证（优化）。

**优化：** 更完整的评测矩阵怎么做？  
答题要点：加入引用覆盖率、来源多样性、时间一致性、冲突率；建立失败样本库与回归门禁（可选升级）。

**补充：如何评估系统生成的长文档？**
答题要点：不要把整篇报告当成一个“答案”来打分，而要拆成 `文档 -> 章节 -> claim -> 引用/来源` 四层。

- `Accuracy`：长文建议改成 claim-level accuracy，而不是整篇文档 accuracy。
  ```text
  claim_accuracy = supported_claims / judged_claims
  ```
- `F1`：只有在有 gold claims 或 gold outline 时才有意义。
  ```text
  precision = matched_supported_claims / predicted_claims
  recall = matched_gold_claims / gold_claims
  f1 = 2PR / (P + R)
  ```
- `hallucination_rate`：建议用 claim 级未支撑率。
  ```text
  hallucination_rate = unsupported_claims / total_claims
  ```
- `引用覆盖率`：至少分成“有引用”和“引用确实支撑内容”两层。
  ```text
  citation_coverage = claims_with_citation / claims_requiring_citation
  supported_citation_coverage = claims_with_valid_supporting_citation / claims_requiring_citation
  ```
- `来源多样性`：不要只看 URL 数量，至少看 `unique_domains`、`top1_domain_share`、`source_type_count`、`domain_entropy`。
- `报告完整性`：用 planner 产出的 `section_details`、`research_queries`、`key_points` 做对照，统计 `section_completion`、`query_coverage`、`keypoint_coverage`。

**在本项目里的落点：**
- `evals/hallucination_eval/`：适合做 report-vs-source groundedness。
- `multi_agents/agents/check_data.py`：已有 `final_score`、`section_coverage`、`ACCEPT/RETRY/BLOCKED`，适合做章节级质量信号。
- `multi_agents/agents/claim_verifier.py`：在线产出 `source_index`、`claim_annotations`、`claim_confidence_report`，适合做断言级引用覆盖和冲突分布统计。
- `backend/server/server_utils.py`：日志里已有 `query/sources/context/report/costs`，适合抽取文档级统计。
- `multi_agents/workflow_session.py`：有 checkpoint，可做章节级回归与 rerun 分析。

**推荐的文档评分卡（可选升级）：**
```text
groundedness = 1 - hallucination_rate
citation_quality = supported_citation_coverage
completeness = 0.5 * section_completion
             + 0.3 * keypoint_coverage
             + 0.2 * query_coverage
```

---

### Q18（基础）：你如何做可观测性/Tracing？为什么要分“研究模型”和“评测模型”？

**标准答案（可直接说）：**
一句话：可观测性用于定位链路失败与质量退化；评测模型与研究模型分离能降低偏差并支持不同成本/稳定性配置。

展开：多智能体链路支持 LangSmith tracing（环境变量开启），并输出阶段性日志（规划、并行研究、写作、审阅等）。评测侧使用独立的评分/评审模型对输出打分或做幻觉判定，避免“同一个模型既生成又裁判”的偏差，并允许评测阶段使用更稳定/更强的模型以提高一致性。

常见坑/反杀点：
- 不要只说“打日志”；要讲链路级追踪、阶段级指标与回归对照。
- 面试官会追问关键指标：时延、失败率、来源数、重试次数、通过率、成本等。

**相关模块（对应实现）：**
- multi_agents/main.py
- evals/simple_evals/
- evals/hallucination_eval/

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：运行期通过 websocket/日志把每个阶段的事件与产物输出；ClaimVerifier 还会记录 citation coverage fallback、claim counts 与 suspicious/hallucination 数量；同时支持 LangSmith tracing（`LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY`）（见 `backend/server/server_utils.py`、`docs/docs/gpt-researcher/handling-logs/langsmith-logs.md`）。配置层面区分 fast/smart/strategic LLM（见 `gpt_researcher/config/config.py`），评测脚本也可用独立 grader 模型（见 `evals/README.md`）。
- 为什么：研究链路长、节点多；必须可观测才能定位慢点/贵点/错点，同时把“产出模型”和“评测模型”分离以减少自评偏差。
- 利弊：利是更易调参与成本分析；弊是 tracing 带来额外开销与数据合规/隐私治理需求。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 可观测性在 agent 系统里解决什么？  
答题要点：定位哪个节点导致质量/时延异常；复盘回流原因；支撑 A/B 对比。

**实现：** 当前有哪些观测手段？  
答题要点：阶段日志输出；可选 LangSmith tracing；评测日志留存用于历史对比。

**边界：** 现有观测不足在哪里？  
答题要点：缺少统一指标面板与结构化事件；缺少 per-node 成本/通过率统计（可选升级）。

**取舍：** 为什么评测与研究解耦？  
答题要点：降低偏差；让研究模型可为成本/速度优化，评测模型可为稳定性优化。

**优化：** 生产级你会加哪些指标？  
答题要点：每节点成功率/时延/成本、引用覆盖率、重试次数分布、来源多样性、审阅回合数（可选升级）。

---

### Q20（基础）：什么是模型分层/路由（tiering）？你们怎么按 agent 选择不同模型档位？

**标准答案（可直接说）：**
一句话：tiering 是把模型按能力/成本分层，并按 agent 角色与轮次策略选择档位；本项目用持久化状态记录每个 agent 的当前档位，实现按轮次/阶段调整模型。

展开：仓库提供持久化的“档位状态管理”，为不同 agent（planner/scrap/check_data/writer/review/revise）配置 tier 列表与当前索引，写入 `agent_state.json`。在证据采集与校验阶段，会根据轮次/重试次数设置档位，从而在关键轮次使用更强模型，在常规轮次使用更轻量模型，达到质量与成本的平衡（tiered routing）。

常见坑/反杀点：
- 不要把 tiering 说成“随机换模型”；要强调有状态、可控、按阶段策略。
- 面试官会追问“换模型会不会风格漂移”；要讲结构化输出与门禁如何兜底。

**相关模块（对应实现）：**
- multi_agents/agents/state_controller.py
- multi_agents/agent_state.json
- multi_agents/agents/scrap.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：多智能体侧用 `StateController` + `agent_state.json` 维护每个 agent 的模型 tiers 与当前档位，并在 Scrap/CheckData 等阶段按轮次切换 tier（见 `multi_agents/agents/state_controller.py`、`multi_agents/agent_state.json`、`multi_agents/agents/scrap.py`、`multi_agents/agents/check_data.py`）。单体研究侧在配置里区分 fast/smart/strategic LLM（见 `gpt_researcher/config/config.py`）。
- 为什么：把“便宜快”和“贵强”用在不同节点与不同失败模式上，控制成本同时保住关键质量。
- 利弊：利是预算可控、失败可升级；弊是路由策略需要回归评测支撑，否则容易出现风格漂移或关键节点用错模型。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 为什么不同 agent 需要不同模型档位？  
答题要点：不同角色对能力要求不同（规划/校验/写作/审阅），成本敏感度也不同。

**实现：** 档位状态如何持久化与更新？  
答题要点：JSON 记录 tiers 与当前索引；写入采用原子替换，避免文件损坏；可按轮次设置档位。

**边界：** tiering 的风险是什么？  
答题要点：输出不一致、策略漂移；需要结构化 schema、校验与回退策略。

**取舍：** 为什么不只按 token 预算即时选择？  
答题要点：有状态能表达“轮次升级/降级”；即时选择可作为补充策略（可选升级）。

**优化：** 更复杂路由如何做？  
答题要点：按任务复杂度分类、按失败模式升级、引入健康度与速率限制、A/B 评测驱动策略更新（可选升级）。

## 二、复杂题（15）

### Q32（复杂）：评测体系如何避免“自己评自己”的偏差？如何做基线/对照组/可重复性？

**标准答案（可直接说）：**
一句话：通过评测与研究解耦、固定数据集与日志、明确对照组配置，尽量降低偏差并保证可重复性。

展开：SimpleQA 使用固定问题集与标准答案，并用可配置评分模型给出分级与指标；评测日志用于历史对比。Hallucination eval 使用 judge 对输出与 source text 做对齐判断。为降低偏差，应在评测中固定样本、固定温度、固定评测配置，并确保评分模型与研究模型可分离。对照组做法是：固定其它变量，只改变一个策略（如是否启用 ASA 或 Check Data），对比 Accuracy/F1 与 hallucination 指标变化。

常见坑/反杀点：
- judge 也会错：要讲交叉评测与人评小样本校准（可选升级）。

**相关模块（对应实现）：**
- evals/simple_evals/
- evals/hallucination_eval/
- evals/README.md

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：评测代码与研究链路解耦：SimpleQA 用独立 grader 模型对短答打分；hallucination_eval 将报告与 sources 对照判定（见 `evals/README.md`、`evals/simple_evals/*`、`evals/hallucination_eval/*`）。
- 为什么：避免“同一模型既产出又裁判”的系统性偏差；同时保留可重复的题集与历史日志做对照。
- 利弊：利是更接近客观回归；弊是 grader 仍可能偏好某种表述，且检索/抓取的随机性会影响稳定性，需要固定配置与种子/缓存（可选升级）。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 评测偏差有哪些来源？  
答题要点：评分模型偏好、数据集分布偏、提示词泄漏、同模型自洽偏差。

**实现：** 如何保证可重复？  
答题要点：固定数据集文件、固定评测配置、保存日志用于回归对比。

**边界：** 哪些指标仍难自动评测？  
答题要点：长文逻辑一致性、论证严谨性、写作风格；需要人评或更复杂指标（可选）。

**取舍：** 为什么不只用一个指标？  
答题要点：准确性与证据对齐性是不同维度；单指标容易被投机优化。

**优化：** 更完整的评测矩阵？  
答题要点：加入引用覆盖率、来源多样性、时间一致性、冲突率；构建失败样本库与回归门禁（可选升级）。

---

### Q34（复杂）：系统如何控制成本/时延？模型分层、迭代上限、证据压缩、并发粒度分别怎么做？

**标准答案（可直接说）：**
一句话：通过 tiering（按角色/轮次选模型）、迭代与重试上限（防止无止境回流）、证据压缩（去重+MMR）、以及章节级并行（提升吞吐）来平衡质量与成本/时延。

展开：tiering 把强模型用在关键轮次与关键节点；多轮采集与校验有上限，避免无限重试；证据侧通过去重与 MMR 将上下文压缩到高信息密度片段，减少 token；并行以章节为单位，降低互相干扰并缩短 wall time。整体策略是“尽早暴露失败并回流补证据”，避免在错误路径上投入写作与审阅成本。

常见坑/反杀点：
- 并行不等于更省钱：它省 wall time，但需要配合限流与预算管理（可选升级）。

**相关模块（对应实现）：**
- multi_agents/agents/state_controller.py
- multi_agents/agents/scrap.py
- multi_agents/agents/check_data.py

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：成本/时延控制主要靠：迭代上限（`scrap_max_iterations`、`check_data_max_retries`）、检索条数上限、MMR 时间预算上限、按轮次切换更强/更弱模型 tiers（见 `multi_agents/agents/scrap.py`、`multi_agents/agents/check_data.py`、`multi_agents/agents/state_controller.py`）。并行粒度以“章节”为主（见 `multi_agents/agents/editor.py`）。
- 为什么：研究链路的主要开销来自检索/抓取与高阶模型；需要把“升级”变成可控的条件分支而不是默认路径。
- 利弊：利是预算可预测；弊是上限过紧会导致覆盖不足，上限过松会导致成本爆炸，需要用评测与在线指标（失败率/覆盖度/引用数）驱动调参。

**追问链（定义 → 实现 → 边界 → 取舍 → 优化）：**
> **举例提示（面试官听不懂就用）：** 把本题映射到“AI 对就业市场的影响”示例卡片来讲——先追问口径 → 产出结构化大纲（`section_details`）→ 按章生成 `research_queries` → 证据采集与门禁 → 写作/审阅/发布。这样抽象概念会变成“我在这个例子里具体做了什么”。

**定义：** 成本/时延控制的核心原则？  
答题要点：早失败、少噪声、结构化输出减少重跑、把重成本操作放到必要时。

**实现：** 上限在哪里体现？  
答题要点：多轮采集有最大轮次；校验重试有上限；超过上限会阻断或降级。

**边界：** 哪些主题仍会很贵？  
答题要点：超宽泛主题、强反证要求、抓取失败率高；需要缩小范围或 HITL。

**取舍：** 为什么不一直用最强模型？  
答题要点：成本不可控；分层能把强模型用在最需要的节点/轮次。

**优化：** 如何做硬预算？  
答题要点：预算写入 state，按节点分配配额；超配额触发降级（减少轮次/减少 sources/输出不确定性声明）。

---

## 附：Top12 对应条目（从原附录迁移）

### 12) 多模型分层/tiers 与可观测性怎么落地？如何控成本与回归验证？
**一句话结论：** 用 tiers 把“贵模型”留给关键轮次，用 tracing+evals 做回归门禁，避免越改越飘、越跑越贵。

**实现落点：**
- `multi_agents/agents/state_controller.py` + `multi_agents/agent_state.json`：保存每个 agent 的 tiers 与当前档位；scrap/check_data 会按轮次调用 tier 切换（见 `multi_agents/agents/scrap.py`、`multi_agents/agents/check_data.py`）。
- `gpt_researcher/config/config.py`：单体研究侧区分 fast/smart/strategic LLM，并支持 `REASONING_EFFORT` 等配置。
- tracing：LangSmith 通过 `LANGCHAIN_TRACING_V2` 与 `LANGCHAIN_API_KEY`（见 `docs/docs/gpt-researcher/handling-logs/langsmith-logs.md`、`backend/server/server_utils.py`）。
- evals：`evals/simple_evals`（短事实性回归）与 `evals/hallucination_eval`（长文对 sources 对照）（见 `evals/README.md`）。

**为什么这样设计：**
- 研究链路的最大成本来自高阶模型与多轮检索；分层能把“升级”变成条件分支；观测与评测让策略可回归。

**利弊与边界：**
- 利：成本可预测、失败可升级；能用 evals 做回归对照，减少“改 prompt 变更风格”带来的隐性退化。
- 弊：tiers/路由策略需要持续用回归集校准；tracing 带来额外开销与隐私治理要求。

**如何验证：**
- `tests/test_state_controller.py` 可回归 tiers 状态读写；评测可跑 `evals/simple_evals/run_eval.py` 做短答回归（需要配置 API key）。

<!-- QUESTIONS_END -->
