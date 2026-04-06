<div align="center">
  <h1>Auto_Research_Engine</h1>

  <p>
    面向 Web 与本地文档场景的开源深度研究系统。
    将规划、检索、验证、写作、审校与发布整合进一套多智能体流程。
  </p>

</div>

Auto_Research_Engine 是一个面向研究类任务的多智能体系统，支持从 Web、本地文档或混合数据源中生成长篇、有来源支撑的研究报告。它不是让单个模型一次性生成长答案，而是把任务拆成明确阶段：规划、证据收集、数据验证、写作、审校、修订与发布。

它适合市场分析、技术调研、公司研究、内部知识整理、研究型 Copilot、报告生成产品等场景。

## 为什么选择 Auto_Research_Engine

- **默认多智能体。** 将规划、检索、证据把关、写作、审校、发布拆分为不同职责的 agent。
- **同时支持 Web 与文档研究。** 搜索、抓取、本地文件、混合研究都走同一套流程。
- **章节级并行研究。** 报告会先拆成多个 section，再分别研究并聚合。
- **支持节点回溯。** `节点回溯`（`Rerun from Checkpoint`）允许只重跑某个节点或某个章节，不必重新跑完整流程。
- **面向产品集成。** 仓库内包含 FastAPI、WebSocket 流式输出、Next.js 前端、CLI、导出能力、测试与文档。
- **可插拔架构。** 检索器、抓取器、模型供应商、向量存储与 MCP 集成都可替换扩展。

## 工作方式

1. 用户通过 CLI、API、WebSocket 或前端提交研究任务。
2. Planner 基于任务和初始研究结果生成结构化大纲。
3. 每个 section 独立进入研究子流程，并行收集证据。
4. `check_data` 判断当前 section 的证据是否足够、是否需要重试，或是否应停止。
5. Writer 基于已验证的 section 结果拼装完整报告。
6. Reviewer 与 Reviser 在最终草稿上循环工作，Publisher 负责导出 Markdown、PDF、DOCX。

## 节点回溯

`multi_agents` 工作流支持 `节点回溯`（`Rerun from Checkpoint`）。

- 每次初次运行或回溯都会生成一个新的 workflow session，并绑定到同一个 `report_id`
- 全局节点如 `browser`、`planner`、`researcher`、`writer`、`reviewer`、`reviser`、`publisher` 都会保存为可回溯节点
- 章节级节点也会被保存；开启 ASA 时包含 `scraping` 与 `check_data`
- 回溯只会重算被选中节点及其下游依赖，不会无条件从头开始
- 多轮修改会以父子 session 的形式保留历史；最新失败回溯不会覆盖上一版成功结果

工作流 session 数据保存在：

```text
data/workflows/{report_id}/
```

## 使用入口

| 入口 | 适用场景 | 位置 |
| --- | --- | --- |
| CLI | 在终端直接生成研究报告 | `python cli.py "<query>" --tone objective` |
| FastAPI | 报告 CRUD、聊天、上传、workflow 接口 | `backend/server/app.py` |
| WebSocket | 流式研究输出与节点回溯 | `/ws` |
| Next.js 前端 | 研究界面、历史记录、节点回溯 | `frontend/nextjs/` |
| Python 包 | 在代码中集成研究能力 | `gpt_researcher/` |

## 快速开始

### 前置要求

- Python 3.11+
- 如果要运行 Next.js 前端，需要 Node.js 18+
- 至少一个可用的大模型提供方与一个检索提供方

推荐最小环境变量：

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

### 1. 克隆并安装

```bash
git clone https://github.com/MrOdd-Use/Auto_Research_Engine.git
cd Auto_Research_Engine

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

最小示例：

```env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
DOC_PATH=./my-docs
```

### 3. 启动后端

```bash
python main.py
```

默认后端地址：

- `http://localhost:8000`

### 4. 启动 Next.js 前端

```bash
cd frontend/nextjs
npm install --legacy-peer-deps
npm run dev
```

默认前端地址：

- `http://localhost:3000`

也可以使用 Docker 启动完整栈：

```bash
docker compose up --build
```

## 使用示例

### CLI

```bash
python cli.py "电池技术最近有哪些关键突破？" --tone objective
```

### Workflow 接口

前端使用的 workflow 层暴露了以下能力：

- `GET /api/reports/{id}/workflow`：获取某个报告的 session 列表与回溯树
- WebSocket `start`：支持带 `report_id` 启动一个稳定绑定的研究任务
- WebSocket `rerun`：支持传入 `report_id`、`checkpoint_id`、可选 `note`，触发 `节点回溯`

### 本地文档研究

1. 把文件放到本地目录，例如 `./my-docs`
2. 在 `.env` 中设置 `DOC_PATH`
3. 在工作流配置中选择 local 或 hybrid 模式

支持 PDF、TXT、CSV、Excel、Markdown、PowerPoint、Word 等格式。

## 仓库结构

```text
backend/          FastAPI、WebSocket、报告与 workflow 持久化
docs/             文档站点
evals/            评测脚本与数据
frontend/         静态前端与 Next.js 前端
gpt_researcher/   核心研究引擎
multi_agents/     多智能体编排、writer/reviewer/reviser 流程
tests/            自动化测试
```

## 项目亮点

- `gpt_researcher/` 提供 retrieve-scrape-compress-write 的核心研究引擎
- `multi_agents/` 在其上叠加规划、章节研究、证据门禁、最终审校
- `backend/server/workflow_store.py` 为 `节点回溯` 提供 workflow session 持久化
- `frontend/nextjs/` 提供报告历史、session 切换、`Rerun from Checkpoint` 控件
- `tests/test_workflow_sessions.py` 覆盖多轮回溯与 session 行为

## 文档

- 总文档：https://docs.gptr.dev
- 前端文档：`docs/docs/gpt-researcher/frontend/nextjs-frontend.md`
- 多智能体说明：`multi_agents/README.md`

## 参与贡献

欢迎贡献。如果你想参与：

1. 提交 issue 描述 bug、回归或功能建议
2. Fork 仓库并创建聚焦的开发分支
3. 对行为变更补充测试
4. 发起 PR，并清楚说明改动内容与影响范围

## 社区

- Discord: https://discord.gg/QgZXvJAccX
- Issues: https://github.com/MrOdd-Use/Auto_Research_Engine/issues

## 许可证

MIT
