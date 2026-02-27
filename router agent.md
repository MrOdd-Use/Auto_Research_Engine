好的，这份 **v3.3 终版开发规范**是为 Coding Agent 量身定制的。它包含了**动态模型发现**、**主备池扩容**的所有技术细节，并提供了精确的伪代码和开发指令。

请将以下内容直接提供给你的 Coding Agent。

---

# Route Agent (RA) 系统架构与开发规范 v3.3

## 1. 项目元数据 (Project Metadata)
*   **项目名称**: Route Agent (RA)
*   **版本**: 3.3 (集成动态发现与弹性扩容)
*   **核心职责**: AutoResearch 生态系统的基础设施中间件。负责模型资源的**动态发现**、**静态分级**、**运行时路由**、**健康监控**与**弹性扩容**。
*   **技术栈**:
    *   **Language**: Python 3.10+
    *   **Persistence**: SQLite
    *   **Hot Storage**: Redis
    *   **Config**: JSON (GitOps), `.env`

## 2. 环境变量配置 (`.env`)
```bash
# 授权的供应商列表 (JSON 格式字符串)
AUTHORIZED_VENDORS='["openai", "anthropic", "google", "deepseek"]'

# 可用模型池的最小数量阈值
MIN_POOL_SIZE=9
```

## 3. 核心数据结构 (Data Schema)

### 3.1 配置文件 (`model_benchmark.json`)
*   **用途**: 定义所有可用模型的物理属性与能力基准。**该文件应包含尽可能多的模型版本，由 Discovery 模块进行筛选**。
```json
{
  "models": {
    "gpt-4-o-2024-05-13": { "provider": "openai", ... },
    "gpt-4-turbo-2024-04-09": { "provider": "openai", ... },
    "claude-3-5-sonnet-20240620": { "provider": "anthropic", ... },
    "claude-3-opus-20240229": { "provider": "anthropic", ... }
  }
}
```

### 3.2 SQLite 数据库 (`ra_metadata.db`)

#### 表 1: `agent_configs` (静态菜单)
存储每个 Agent 专属的、**按能力降序排列**的**主备**模型列表。
```sql
CREATE TABLE agent_configs (
    agent_id TEXT PRIMARY KEY,       
    primary_tier_list_json TEXT,  -- 主池 (JSON List)
    reserve_tier_list_json TEXT,  -- 备用池 (JSON List)
    updated_at DATETIME
);
```

#### 表 2: `agent_states` (运行时状态)
```sql
CREATE TABLE agent_states (
    agent_id TEXT PRIMARY KEY,
    current_mode INTEGER DEFAULT 0  -- 0=Default(省钱), 1=Performance(强力)
);
```

### 3.3 Redis 键值设计
*   **模型健康度 (Hash)**
    *   Key: `ra:health:{model_id}`
    *   Fields: `fail_count` (int), `status` (string: "HEALTHY" | "DOWN")
    *   TTL: 600秒
*   **复杂度缓存 (String)**
    *   Key: `ra:cache:complexity:{md5(prompt)}`
    *   Value: "SIMPLE" | "COMPLEX"
    *   TTL: 24小时

---

## 4. 模块详细逻辑 (Module Specifications)

### 4.1 DataHub (初始化与模型发现)

#### 功能 A: 动态模型发现 (`ModelDiscovery`)
*   **输入**: `raw_data` (来自 `model_benchmark.json`), `authorized_vendors` (来自 `.env`)
*   **输出**: `primary_pool` (字典列表), `reserve_pool` (字典列表)
*   **逻辑**:
    1.  **分组**: 将 `raw_data` 中所有模型按 `provider` 字段分组。
    2.  **供应商过滤**: 只保留在 `authorized_vendors` 列表中的分组。
    3.  **版本提取与排序**:
        *   对每个分组内的模型列表，使用正则表达式 `(\d{4}-?\d{2}-?\d{2}|v?\d+\.\d+)` 从 `model_id` 中提取版本信息。
        *   按版本信息**降序**排列。
    4.  **切片**:
        *   将每个分组排序后的前 4 个模型添加到 `primary_pool`。
        *   将每个分组排序后的第 5 个及之后的模型添加到 `reserve_pool`。

#### 功能 B: 静态 Rerank (Agent 注册时)
*   **输入**: `agent_description`, `primary_pool`, `reserve_pool`
*   **逻辑**:
    1.  为 `primary_pool` 和 `reserve_pool` 分别打分排序，生成 `primary_tier_list` 和 `reserve_tier_list`。
    2.  将这两个列表存入 SQLite `agent_configs` 表。

### 4.2 Reranker (运行时路由引擎) - **核心实现**

#### 步骤 1: 可用性过滤辅助函数 (`_filter_available_models`)
*   **输入**: `tier_list` (模型ID列表), `tokens` (int)
*   **输出**: `available_list` (通过过滤的模型ID列表)
*   **逻辑**:
    1.  遍历 `tier_list`。
    2.  对每个 `model_id`，执行**上下文硬约束过滤**和**Redis 健康度检测**。
    3.  将通过的模型加入 `available_list` 并返回。

#### 步骤 2: 核心路由逻辑 (`get_execution_config`)
*   **输入**: `agent_id`, `input_text`
*   **输出**: 模型配置字典
*   **流程**:
    1.  从 `.env` 读取 `MIN_POOL_SIZE`。
    2.  计算 `input_text` 的 Token 数。
    3.  从 SQLite 读取该 `agent_id` 的 `primary_tier_list` 和 `reserve_tier_list`。
    4.  **过滤主池**: 调用 `_filter_available_models` 处理 `primary_tier_list`，得到 `available_models`。
    5.  **弹性扩容**:
        *   判断 `if len(available_models) < MIN_POOL_SIZE:`
        *   如果为真，调用 `_filter_available_models` 处理 `reserve_tier_list`，得到 `available_reserves`。
        *   将 `available_reserves` 中的模型**不重复地**追加到 `available_models` 列表尾部。
    6.  **熔断**: `if not available_models:` 抛出 `NoAvailableModelError`。
    7.  **复杂度判定**: 调用 `_classify_complexity`。
    8.  **状态获取**: 从 SQLite 读取 `agent_states`。
    9.  **最终选择**:
        *   若需强力模型，取 `available_models[0]`。
        *   否则，取 `available_models[-1]`。
    10. 返回该模型的完整配置。

---

## 5. 伪代码实现 (For Coding Agent)

### 5.1 `ModelDiscovery` Class

```python
import re
import os
import json

class ModelDiscovery:
    def __init__(self, vendors: list):
        self.vendors = vendors
        self.version_pattern = re.compile(r'(\d{4}-?\d{2}-?\d{2}|v?\d+\.\d+)')

    def _extract_version(self, model_id: str) -> str:
        match = self.version_pattern.search(model_id)
        return match.group(1).replace('-', '') if match else "0"

    def discover_pools(self, raw_data: dict) -> (list, list):
        primary_pool, reserve_pool = [], []
        grouped_models = {}

        for m_id, m_info in raw_data['models'].items():
            vendor = m_info.get('provider')
            if vendor in self.vendors:
                grouped_models.setdefault(vendor, []).append({"id": m_id, **m_info})

        for vendor, models in grouped_models.items():
            sorted_models = sorted(models, key=lambda x: self._extract_version(x['id']), reverse=True)
            primary_pool.extend(sorted_models[:4])
            reserve_pool.extend(sorted_models[4:])
            
        return primary_pool, reserve_pool
```

### 5.2 `RouteAgent` Class Core Logic

```python
class RouteAgent:
    def __init__(...):
        self.MIN_POOL_SIZE = int(os.getenv("MIN_POOL_SIZE", 9))
        # ... other initializations

    def _filter_available_models(self, tier_list: list, tokens: int) -> list:
        available = []
        for model_id in tier_list:
            meta = self.benchmark['models'][model_id]
            if meta.get('context_window', 0) < (tokens + 1000):
                continue
            health_status = self.redis.hget(f"ra:health:{model_id}", "status") or b"HEALTHY"
            if health_status.decode() == "DOWN":
                continue
            available.append(model_id)
        return available

    def get_execution_config(self, agent_id: str, input_text: str) -> dict:
        tokens = self._estimate_tokens(input_text)
        primary_list, reserve_list = self.db.get_tier_lists(agent_id)
        
        available_models = self._filter_available_models(primary_list, tokens)
        
        if len(available_models) < self.MIN_POOL_SIZE:
            available_reserves = self._filter_available_models(reserve_list, tokens)
            
            # Use a set for efficient deduplication
            existing_ids = set(available_models)
            for model_id in available_reserves:
                if model_id not in existing_ids:
                    available_models.append(model_id)
        
        if not available_models:
            raise NoAvailableModelError("No models available after filtering.")
            
        complexity = self._classify_complexity(input_text)
        agent_mode = self.db.get_agent_mode(agent_id)
        
        if complexity == "COMPLEX" or agent_mode == 1:
            selected_model_id = available_models[0]
        else:
            selected_model_id = available_models[-1]
            
        return self.benchmark['models'][selected_model_id]
```

---

## 6. 给 Coding Agent 的特别指令 (Final Prompt v3.3)

1.  **Environment Variable Parsing**: `os.getenv("AUTHORIZED_VENDORS")` will return a string. It must be parsed using `json.loads()` to become a Python list.
2.  **Robust Dictionary Access**: When accessing `model_benchmark.json`, use `.get('field', default_value)` to prevent `KeyError` if a model's metadata is incomplete.
3.  **Redis Connection Handling**: All Redis calls must be wrapped in `try-except redis.exceptions.ConnectionError`. In case of failure, log the error and proceed as if all models are healthy (fail-open strategy).
4.  **Logging**: Add detailed logging at each critical step:
    *   Log the size of the primary and reserve pools after discovery.
    *   Log when the reserve pool is activated.
    *   Log the final `available_models` list before selection.
    *   Log the selected model and the reason (e.g., "COMPLEX task, selected Top Tier").