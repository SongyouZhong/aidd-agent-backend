# Agent-as-a-Tool 架构重构实施说明

> 实施日期：2026-05-11  
> 重构目标：消除"两套大脑"架构偏差，将 `target_discovery_graph` 封装为 CORE 工具，通过 LLM 原生 Function Calling 触发，恢复单一执行路径。

---

## 1. 背景与问题

原架构中存在两条完全独立的执行路径：

| 路径 | 入口 | 执行逻辑 |
|---|---|---|
| **前端聊天** | `POST /api/v1/chat` → `chat_service.py` | 手动 ReAct 循环，`provider.stream()` |
| **LangGraph dev** | `langgraph dev` → `graph.py` | LangGraph 图，`intent_router_node` → `target_discovery_node` |

前端聊天路径完全绕过了 LangGraph 图，导致靶点发现功能无法被触发，意图路由、子图、Checkpoint 全部失效。

本次重构采用**全新架构方案**中提出的 **Agent-as-a-Tool** 模式：

> 不让 Graph 控制对话，让对话控制 Graph。

---

## 2. 架构变更总览

### 2.1 Before（双路径）

```
POST /api/v1/chat
  └─ chat_service.stream_chat()
       └─ provider.stream()   ← 直接调 LLM，无意图路由
            └─ LLM 自行决定是否调工具（靶点发现工具不存在）

langgraph dev（调试专用，不是前端路径）
  └─ graph.py:graph
       └─ intent_router_node
            └─ target_discovery_node（5节点子图）
```

### 2.2 After（单路径）

```
POST /api/v1/chat
  └─ chat_service.stream_chat()
       └─ provider.stream()       ← 保留原生 token 流 / thinking
            └─ LLM Function Calling
                 └─ run_target_discovery (CORE tool)  ← 新增
                      └─ target_discovery_graph.astream()  ← 5节点并行子图
                           ├─ 每个节点切换 → research_progress SSE
                           ├─ final_report → SessionFile (S3 + DB)
                           └─ 返回紧凑摘要 → ToolMessage → 对话历史
```

---

## 3. 新增文件

### 3.1 `app/services/chat_context.py`（新建）

**职责**：定义跨模块的 per-request ContextVar，让工具函数在不改变 `@tool` 签名的前提下访问 session/user 上下文和进度回调。

```python
# 核心 ContextVar
current_chat_context: ContextVar[Optional[ChatRequestContext]]
progress_callback: ContextVar[Optional[ProgressCallback]]
deep_research_running: ContextVar[bool]  # 防止同 session 重入
```

**关键设计**：
- `current_chat_context` 在 `stream_chat()` 入口 `.set()`，无需在工具签名中传参
- `progress_callback` 是 `async def (event_type: str, payload: dict) -> None` 类型的回调，由 chat_service 注入，工具调用以发送进度事件
- `deep_research_running` 防止同一对话轮次内 LLM 重复调用深度调研工具

---

### 3.2 `app/services/target_report_service.py`（新建）

**职责**：将 `TargetReport` JSON 持久化为 `SessionFile`（S3 + PostgreSQL），供前端下载并供后续对话通过 `_load_file_context` 按需加载。

```python
async def save_report_as_session_file(
    *,
    session_id: str,
    user_id: str,
    project_id: str | None,
    target_query: str,
    report: dict[str, Any],
) -> SessionFile
```

**关键设计**：
- 自动从 `sessions` 表解析 `project_id`（若调用方未提供）
- 复用 `file_service.upload_file()`，mime_type 为 `application/json`
- 生成文件名如 `TARDBP_target_report.json`
- 失败时 raise `RuntimeError`，工具层捕获并降级（返回 `status:ok_no_file`）

---

### 3.3 `app/tools/deep_research.py`（新建）

**职责**：将 `target_discovery_graph` 封装为标准 LangChain `@tool`，这是本次重构的核心。

```python
@tool
async def run_target_discovery(target_query: str) -> str:
    """Run a multi-source deep research pipeline for a protein/gene target.
    ...
    """
```

**执行流程**：

```
run_target_discovery("TARDBP")
  ├─ 1. 重入检查：deep_research_running.get()
  ├─ 2. 上下文检查：get_chat_context() → session_id, user_id
  ├─ 3. 启动子图：td_graph.astream(initial)
  │     └─ 每个 node 切换 → progress_callback("research_progress", {"phase": node_name})
  ├─ 4. 捕获 final_report（synthesize 节点输出）
  ├─ 5. 持久化：target_report_service.save_report_as_session_file(...)
  └─ 6. 返回紧凑摘要 JSON（含 report_file_id, counts, function_brief, notes）
```

**返回值示例**：
```json
{
  "status": "ok",
  "target": "TARDBP",
  "target_meta": {"name": "TARDBP", "gene_symbol": "TARDBP", "uniprot_ids": ["Q13148"]},
  "counts": {
    "proteins": 1,
    "papers": 12,
    "disease_associations": 8,
    "pathways": 14,
    "small_molecule_drugs": 5,
    "peptide_drugs": 2,
    "antibody_drugs": 0
  },
  "function_brief": "TDP-43 (TAR DNA-binding protein 43) is a nuclear protein...",
  "notes": [],
  "report_file_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "report_filename": "TARDBP_target_report.json"
}
```

**错误降级**：所有异常内部捕获并以 `{"status":"error","message":"..."}` 返回，不破坏 ReAct 消息序列。

**循环导入解决方案**：`build_target_discovery_graph` 以懒加载方式在 `_run_pipeline()` 函数体内导入，而非模块顶层，避免 `app.tools` ↔ `app.tools.deep_research` ↔ `target_discovery_graph` ↔ `app.tools` 的循环。

---

## 4. 修改文件

### 4.1 `app/tools/registry.py`

1. **新增导入**：`from app.tools.deep_research import run_target_discovery`
2. **`CORE_TOOL_NAMES` 集合新增** `"run_target_discovery"`
3. **默认注册块新增**：

```python
reg.register(
    run_target_discovery,
    category="core",
    keywords=[
        "target", "discovery", "deep research", "target discovery",
        "靶点", "靶点发现", "靶点分析", "深度调研", "调研",
        "protein analysis", "gene analysis",
    ],
)
```

选为 **CORE** 而非 DEFERRED 的原因：自然语言输入（"帮我做 TDP-43 靶点分析"）无法被 `tool_search` 的关键词评分可靠命中；CORE 工具始终出现在 System Prompt 中，LLM 可通过 Function Calling 直接选择。

---

### 4.2 `app/tools/__init__.py`

新增导入和 `__all__` 导出：
```python
from app.tools.deep_research import run_target_discovery
# ...
__all__ = [..., "run_target_discovery", ...]
```

---

### 4.3 `app/services/chat_service.py`

#### 新增 imports
```python
import asyncio
from app.agent.context_manager import CompactTrackingState, apply_compaction, maybe_compact
from app.services.chat_context import ChatRequestContext, current_chat_context, progress_callback
```

#### `stream_chat()` 签名新增 `project_id` 参数
```python
async def stream_chat(
    session_id: str,
    user_content: str,
    user_id: str,
    plan_mode: bool = False,
    file_ids: list[str] | None = None,
    project_id: str | None = None,          # ← 新增
) -> AsyncGenerator[str, None]:
```

#### 入口：设置 per-request ContextVar + 进度队列
```python
# 设置上下文供工具使用
current_chat_context.set(ChatRequestContext(session_id, user_id, project_id))

# asyncio.Queue 作为工具 → SSE 的事件桥
progress_queue: asyncio.Queue[dict] = asyncio.Queue()
async def _push_progress(event_type, payload):
    await progress_queue.put({"event": event_type, "data": payload})
progress_callback.set(_push_progress)

produced_file_ids: list[str] = []
compact_tracking = CompactTrackingState()
```

#### ReAct 循环每轮：Auto-Compaction
```python
# 每轮 LLM 调用前检查是否需要压缩上下文
compact_result = await maybe_compact(
    messages,
    model=getattr(provider, "model", "") or settings.GEMINI_MODEL,
    tracking=compact_tracking,
    summarizer=_summarizer,
)
if compact_result is not None:
    messages = apply_compaction(list(messages), compact_result)
```

原 `maybe_compact` 调用位置在已删除的 `agent_node` 中，本次迁移到 `chat_service` 的主路径，使自动压缩对前端聊天路径正式生效。

#### 工具执行：`asyncio.create_task` + 进度队列 drain
```python
# 所有工具以 create_task 运行，同时 drain progress_queue → SSE
tool_task = asyncio.create_task(_execute_tool(tc.tool_name, tc.tool_args))
try:
    while not tool_task.done():
        try:
            ev = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
            yield _sse(ev)          # 转发 research_progress 事件
        except asyncio.TimeoutError:
            continue
    # drain 尾部事件
    while not progress_queue.empty():
        yield _sse(progress_queue.get_nowait())
    result = tool_task.result()
except asyncio.CancelledError:
    tool_task.cancel()
    raise
```

这使所有工具（包括普通工具，虽然它们不发 progress 事件）都以统一方式执行，未来若有其他长耗时工具也能自然接入进度推送。

#### 捕获 `run_target_discovery` 产出的 `report_file_id`
```python
if tc.tool_name == "run_target_discovery":
    try:
        payload = json.loads(result)
        fid = payload.get("report_file_id")
        if fid:
            produced_file_ids.append(str(fid))
    except Exception:
        pass
```

#### assistant 消息持久化时附带 `file_ids`
```python
await append_message(session_id, {
    "id": assistant_msg_id,
    "role": "assistant",
    "content": full_text,
    "ts": _now_iso(),
    "file_ids": produced_file_ids,   # ← 新增，前端可显示下载链接
})
```

---

### 4.4 `app/api/chat.py`

向 `chat_service.stream_chat()` 转发 `project_id`：

```python
chat_service.stream_chat(
    ...,
    project_id=str(payload.project_id) if payload.project_id else None,
)
```

---

### 4.5 `app/agent/prompts/target_discovery.py`

删除已无引用的 `INTENT_ROUTER_PROMPT` 常量（原用于 `agent.py` 中的 `intent_router_node`，该节点已随 `agent.py` 一并删除）。

---

### 4.6 `aidd-agent-front-react/API接口文档.md`

在 §8.2 SSE 事件定义表中新增 `research_progress` 行：

| 事件 | 触发时机 | `data` 字段 | 说明 |
|---|---|---|---|
| `research_progress` | 长耗时工具阶段进度 | `phase`, `target` | 仅 `run_target_discovery` 触发；`phase` 为子图节点名 |

并在 §8.3 之后新增 §8.3.1 靶点调研事件流示例。

---

## 5. 删除文件

| 文件 | 删除原因 |
|---|---|
| `app/agent/agent.py` | 包含 `build_agent` / `intent_router_node` / `target_discovery_node` / `tool_node` / `run_once` / `AgentState`，全部由新架构替代 |
| `app/agent/graph.py` | LangGraph Studio 入口（`langgraph dev` 专用），生产路径不再使用 |
| `app/agent/subagent.py` | 依赖 `build_agent`，随之删除 |
| `langgraph.json` | 指向已删除的 `graph.py:graph` |
| `.langgraph_api/` | LangGraph dev 运行时缓存，与代码一并清理 |

---

## 6. SSE 事件协议变化

新增事件类型：

```
data: {"event": "research_progress", "data": {"phase": "composition", "target": "TARDBP"}}
```

`phase` 取值（对应 `target_discovery_graph` 节点名）：
- `composition` — 蛋白质结构/序列（UniProt / PDB / AlphaFold）
- `literature` — 文献检索（PubMed / Semantic Scholar）
- `function` — 功能/疾病（OpenTargets / Monarch / GO）
- `pathway` — 通路（KEGG / Reactome / STRING）
- `drugs` — 药物（ChEMBL / PubChem / GtoPdb）
- `synthesize` — 最终整合（无工具调用，纯 LLM 聚合）

**已有事件不变**（向后兼容）。

---

## 7. 双轨报告机制

深度调研报告采用双轨策略，兼顾 LLM 摘要质量和上下文成本控制：

```
run_target_discovery 执行完毕
  ├─ 轨道 A（内联，当前轮次）
  │    ToolMessage 包含紧凑摘要（<500 tokens）
  │    → LLM 读取摘要 → 生成友好的中文总结回复
  │    → assistant message 携带 file_ids: [report_file_id]
  │    → 前端可展示下载按钮
  │
  └─ 轨道 B（持久化，后续轮次）
       完整 TargetReport JSON → S3 (application/json)
       SessionFile 记录 → PostgreSQL
       后续追问时 _load_file_context 可按需加载
       → 防止 5-15k token 报告污染后续每轮上下文
```

---

## 8. Auto-Compaction 迁移

原调用位置：`app/agent/agent.py` 中的 `agent_node`（已删除）

新调用位置：`app/services/chat_service.py` 的 ReAct 循环每轮开始前

```python
compact_tracking = CompactTrackingState()  # 函数级局部变量，per-conversation

while total_rounds < MAX_TOOL_ROUNDS:
    # ← 新增：每轮检查
    compact_result = await maybe_compact(messages, model=..., tracking=compact_tracking, summarizer=...)
    if compact_result is not None:
        messages = apply_compaction(list(messages), compact_result)
    ...
```

这使 Auto-Compaction（设计文档 §9）对前端聊天路径正式生效，无需 LangGraph 图介入。

---

## 9. 验证结果

```
$ AIDD_FORCE_FAKE_LLM=1 python smoke_test.py

TOOL RESULT: {
  "status": "ok",
  "target": "TDP-43",
  "counts": {"proteins": 0, "papers": 1, ...},
  "function_brief": "A test",
  "report_file_id": "fake-id",
  "report_filename": "TDP-43_target_report.json"
}

PROGRESS EVENTS: [
  ("research_progress", {"phase": "composition", "target": "TDP-43"}),
  ("research_progress", {"phase": "literature", "target": "TDP-43"}),
  ("research_progress", {"phase": "synthesize", "target": "TDP-43"})
]
```

```
$ python -c "import app.main; print('OK')"
OK

CORE: ['query_arxiv', 'query_pubmed', 'run_target_discovery']
```

```
$ timeout 8 python -c "... uvicorn startup ..."
startup-ok
```

---

## 10. 待完成（后续迭代）

| 优先级 | 事项 | 说明 |
|---|---|---|
| P1 | 前端适配 `research_progress` 事件 | 显示阶段化进度条；前端需新增 case 处理该事件类型 |
| P1 | 前端展示报告下载按钮 | assistant message 的 `file_ids` 可关联下载 API |
| P2 | 编写集成测试 | `tests/test_run_target_discovery_tool.py`、`tests/test_chat_service_target_discovery.py` |
| P2 | 已知子图问题修复 | `synthesize` 节点 400 错误（`reasoning_content` 回传）；`drugs`/`pathway` 超时优化 |
| P3 | `ChatRequest.project_id` 字段确认 | 若前端 payload 中不含 `project_id` 则回落到 DB 查询，确认无误 |
| P3 | README 更新 | 移除 `langgraph dev` 启动说明 |

---

## 11. 文件变更列表

```
新建
  app/services/chat_context.py
  app/services/target_report_service.py
  app/tools/deep_research.py

修改
  app/tools/registry.py
  app/tools/__init__.py
  app/services/chat_service.py
  app/api/chat.py
  app/agent/prompts/target_discovery.py
  aidd-agent-front-react/API接口文档.md

删除
  app/agent/agent.py
  app/agent/graph.py
  app/agent/subagent.py
  langgraph.json
  .langgraph_api/（目录及所有内容）
```
