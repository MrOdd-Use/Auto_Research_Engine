# MCP 集成与部署（Model Context Protocol）

> 本文目标：把 MCP 讲清楚到“能落地部署、能解释取舍、能抗追问”。所有“已实现”口径以仓库代码为准。

---

## Q01（基础）：MCP 是什么？在这个项目里解决什么问题？

**一句话结论：** MCP 是把“外部工具/数据源”以统一协议接入 LLM 的方式；在本项目里，它把检索从“只能搜网页”扩展为“可以用任意 MCP server 的工具”，例如本地数据库、内部知识系统、文件系统、业务 API 等。

**相关模块（对应实现）：**
- `gpt_researcher/retrievers/mcp/retriever.py`
- `gpt_researcher/mcp/`（client/tool_selector/research/streaming）
- `gpt_researcher/actions/retriever.py`

**技术细节（实现 / 为什么 / 利弊）：**
- 实现：MCP 在系统里表现为一种 retriever（`retriever=mcp` 或把 `mcp` 加入 retrievers 列表）。检索阶段会把 query 交给 MCP retriever，并返回与其他搜索引擎一致的标准结果结构（title/href/body），以便后续统一处理。
- 为什么：相比把所有外部系统“硬编码成专用 retriever”，MCP 能用统一协议接入更多工具，降低集成成本并提升可扩展性。
- 利弊：利是扩展面更广；弊是部署/权限/稳定性需要工程化治理（见安全与运维）。

---

## Q02（基础）：本仓库 MCP 检索的核心流程是什么？（两阶段）

**一句话结论：** “工具发现 → 工具选择 → 绑定工具做研究 → 标准化结果”四步走，避免“把所有工具全跑一遍”的浪费。

**相关模块（对应实现）：**
- `gpt_researcher/retrievers/mcp/retriever.py`
- `gpt_researcher/mcp/client.py`
- `gpt_researcher/mcp/tool_selector.py`
- `gpt_researcher/mcp/research.py`
- `gpt_researcher/mcp/streaming.py`

**技术细节（实现 / 为什么 / 利弊）：**
- 工具发现：从配置的 MCP server 拉取可用 tools（支持多 server），并做缓存以减少重复枚举。
- 工具选择：用 LLM 从所有 tools 中挑 2–3 个最相关的（默认上限 3），并带 fallback（当 LLM 选择失败时用规则匹配选工具）。
- 工具执行：把选中的 tools 绑定到 LLM，让 LLM 在一个研究 prompt 内决定调用哪些工具、传什么参数；对 tool 返回做容错解析。
- 结果标准化：将各种工具返回整理成统一的 `[{title, href, body}, ...]`，与 web 搜索 retriever 对齐，便于后续证据处理链路复用。

---

## Q03（基础）：MCP server 怎么“部署/接入”？支持哪几种连接方式？

**一句话结论：** 支持三类接入：本地 stdio 启动、WebSocket 连接、HTTP（streamable HTTP）连接；通过 `mcp_configs` 列表描述 server。

**相关模块（对应实现）：**
- `gpt_researcher/mcp/client.py`
- `gpt_researcher/mcp/README.md`

**mcp_configs（本仓库支持字段）**：
- `name`：server 名称（可选；不填会自动生成）
- `command` / `args`：stdio 模式启动命令与参数（本地启动）
- `env`：stdio server 需要的环境变量（可选）
- `connection_url`：WebSocket/HTTP server 地址（可选）
- `connection_type`：不填时默认 stdio；有 URL 时会自动推断（可显式覆盖）
- `connection_token`：鉴权 token（可选）
- `tool_name`：指定使用某个 tool（可选；具体行为取决于 server 暴露的工具）

**部署模式要点：**
- stdio：适合本机工具（访问本地文件/数据库），优点是网络面小；缺点是部署形态更像“随进程启动”。
- WebSocket：适合长连接与实时交互；需要服务端稳定运行与鉴权。
- HTTP：适合对接已有 API 网关；注意本仓库在适配器侧使用 streamable HTTP transport。

---

## Q04（实操）：在本项目里怎么启用 MCP？（Python 端与后端服务端）

**一句话结论：** 两件事：1）把 `mcp_configs` 传进 `GPTResearcher`；2）把检索器列表里包含 `mcp`（单独用或与 tavily/google/bing 混用）。

**相关模块（对应实现）：**
- `gpt_researcher/agent.py`（接收 `mcp_configs`）
- `gpt_researcher/actions/retriever.py`（`mcp` retriever 工厂）
- `gpt_researcher/skills/researcher.py`（识别 MCP retriever 并注入 researcher/websocket）
- `backend/server/websocket_manager.py`（服务端启用 MCP 并透传配置）

**方式 A：Python 直接调用（离线/脚本场景）**
- 在创建 `GPTResearcher(...)` 时传入 `mcp_configs=[{...}, ...]`
- 同时通过 headers/config 指定 retrievers，把 `mcp` 加进去（例如 `retrievers=["tavily","mcp"]`）

**方式 B：后端 WebSocket（前端/服务化场景）**
- `backend/server/websocket_manager.py` 支持 `mcp_enabled`、`mcp_strategy`、`mcp_configs` 参数，并在启用时把 `mcp` 追加进 retriever 配置，同时设置策略参数。

**常见坑：**
- 未安装依赖：需要 `langchain-mcp-adapters`，否则 MCP client 无法创建（会返回空结果而不是抛错）。
- 只配 retriever 不传 `mcp_configs`：MCP retriever 会检测到缺配置并跳过，避免阻断主链路，但你会“看起来启用了其实没生效”。

---

## Q05（基础）：工具是怎么选的？如何避免“全工具乱跑”？

**一句话结论：** 先让 LLM 从全量 tools 中选少量最相关工具，再在绑定工具的上下文里执行；失败则用规则匹配的 fallback 选工具。

**相关模块（对应实现）：**
- `gpt_researcher/mcp/tool_selector.py`

**技术细节（实现 / 为什么 / 利弊）：**
- 选择上限：默认最多选 3 个工具，控制工具调用面与 latency。
- 解析容错：LLM 输出不一定是严格 JSON，会尝试从文本里提取 JSON；完全失败时走 fallback。
- fallback：按工具名/描述的关键词打分，选最像“研究/检索/读取”的工具，保证最差情况下也能跑通。

---

## Q06（基础）：工具返回结果怎么变成“统一的搜索结果”？（标准化与容错）

**一句话结论：** MCP 工具返回的结构多样（list/dict/含 structured_content/content），会被归一化为统一的 `title/href/body` 结构，供后续证据链路复用。

**相关模块（对应实现）：**
- `gpt_researcher/mcp/research.py`

**为什么重要：**
- 后续模块（抓取/去重/证据选择/写作）不需要关心“这是来自 tavily 还是 MCP tool”，只处理标准结构，降低系统耦合。

---

## Q07（运维）：日志与排障怎么做？怎么确认 MCP 真在工作？

**一句话结论：** 看 WebSocket logs 的 MCP stage 输出 + 看返回结果的数量/内容长度；必要时打开 debug 日志查看工具选择与 tool args。

**相关模块（对应实现）：**
- `gpt_researcher/mcp/streaming.py`
- `gpt_researcher/retrievers/mcp/retriever.py`
- `backend/server/websocket_manager.py`

**排障 checklist：**
- 依赖是否安装：`langchain-mcp-adapters`
- `mcp_configs` 是否非空、transport 是否正确推断（stdio/ws/http）
- server 是否能暴露 tools（工具列表是否为空）
- 工具选择是否选到了工具（LLM 选择失败会走 fallback）
- tool 执行是否报错（错误会被吞掉并返回空结果，避免阻断主链路）

---

## Q08（安全）：MCP 的安全边界怎么讲？（可辩护口径）

**一句话结论：** MCP 扩大了系统的“能力面”，必须用最小权限、凭据治理、来源隔离与审计来控风险；把工具返回当“数据”而不是“指令”。

**建议口径（可选升级，不冒充已做）：**
- server 级 allowlist + tool allowlist（只连可信 server，只开放必要工具）
- token/密钥只走环境变量与 secret manager，不写入日志/产物
- 工具结果注入模型前做内容清洗与截断；对高风险来源降权
- 审计：记录“选了哪些工具/传了什么参数/返回了什么摘要”以便追责与回放

