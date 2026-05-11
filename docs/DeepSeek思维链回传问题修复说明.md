# DeepSeek Thinking Mode `reasoning_content` 回传问题修复说明

## 问题描述

在使用 DeepSeek V4（thinking 模式）作为 LLM 后端时，第一轮调用成功，但在同一 node 的第二次 LLM 调用（即历史消息中包含上一轮 assistant 消息时）会收到如下 400 错误：

```
openai.BadRequestError: Error code: 400 - {
  'error': {
    'message': 'The `reasoning_content` in the thinking mode must be passed back to the API.',
    'type': 'invalid_request_error',
    'param': None,
    'code': 'invalid_request_error'
  }
}
```

## 错误日志重现路径

```
literature node 第一次调用 → HTTP 200 OK
literature node 第二次调用（带历史 assistant 消息）→ HTTP 400 Bad Request
Node literature failed
```

## 根本原因

### DeepSeek Thinking Mode 协议要求

DeepSeek V4 的 thinking 模式会在每次 assistant 响应中返回一个额外字段 `reasoning_content`（即思维链内容）。**API 要求在多轮对话中，必须将上一轮 assistant 消息的 `reasoning_content` 原样回传**，否则拒绝请求。

格式示例（正确）：
```json
{
  "role": "assistant",
  "content": "最终回答内容",
  "reasoning_content": "这是思考过程……"
}
```

### 代码中的问题

**步骤 1 — `generate()` 处理响应时**（`llm_provider.py` ~L443）：

```python
reasoning = getattr(msg, "reasoning_content", "")
if reasoning:
    text = f"<thought>\n{reasoning}\n</thought>\n\n{text}"
```

→ `reasoning_content` 被合并进了 `text`，以 `<thought>…</thought>` 包裹。

**步骤 2 — 下一次调用时 `_to_openai_messages()` 拼装历史消息**（`llm_provider.py` ~L370）：

```python
elif isinstance(m, AIMessage):
    msg_dict = {"role": "assistant", "content": text or None}
    # ← 只还原了 content，没有 reasoning_content 字段
```

→ 整个 `<thought>…</thought>…` 都被塞进 `content`，`reasoning_content` 字段缺失。

→ DeepSeek API 检测到 thinking 模式历史中没有 `reasoning_content`，返回 400。

## 修复方案

在 `DeepSeekProvider` 中重写 `_to_openai_messages` 静态方法，在构建历史 assistant 消息时，将 `<thought>…</thought>` 块解析还原为独立的 `reasoning_content` 字段：

```python
@staticmethod
def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    out = OpenAICompatibleProvider._to_openai_messages(messages)
    for msg_dict in out:
        if msg_dict.get("role") != "assistant":
            continue
        content = msg_dict.get("content")
        if not isinstance(content, str):
            continue
        stripped = content.lstrip()
        if not stripped.startswith("<thought>"):
            continue
        end = stripped.find("</thought>")
        if end == -1:
            continue
        reasoning = stripped[len("<thought>"):end].strip()
        rest = stripped[end + len("</thought>"):].lstrip("\n").lstrip()
        msg_dict["reasoning_content"] = reasoning
        msg_dict["content"] = rest or None
    return out
```

修复后，发送给 DeepSeek 的 assistant 历史消息格式为：

```json
{
  "role": "assistant",
  "content": "最终回答",
  "reasoning_content": "思维链内容（从 <thought>…</thought> 中还原）"
}
```

完全符合 DeepSeek Thinking Mode 的多轮对话协议。

## 修改文件

| 文件 | 修改内容 |
|------|----------|
| `app/agent/llm_provider.py` | `DeepSeekProvider` 类中新增 `_to_openai_messages()` 静态方法重写 |

## 影响范围

- 仅影响 `DeepSeekProvider`，其他 provider（Gemini、Qwen、OpenAICompatibleProvider）行为不变。
- 多轮对话、工具调用场景均适用。
- 如果 assistant 消息不包含 `<thought>` 块（例如非 thinking 模式或第一轮）则逻辑直接跳过，无副作用。

## 临时绕过方案（不推荐）

若不需要思维链功能，可在初始化时禁用 thinking 模式：

```python
DeepSeekProvider(enable_thinking=False)
```

或在 `.env` 中切换到非思考型模型。但推荐使用上述根本修复，保留 thinking 能力。

## 参考

- [DeepSeek Thinking Mode 官方文档](https://api-docs.deepseek.com/guides/thinking_mode)
- 错误发生位置：`app/agent/target_discovery_graph.py` → `_run_node_loop()` → `provider.generate()`
