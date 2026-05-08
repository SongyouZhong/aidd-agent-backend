# DeepSeek 接入架构设计与优化方案

本文档详细分析了 AIDD Agent Platform 现有架构中思维链（Chain of Thought）的实现方式，并针对未来接入 DeepSeek（特别是具有原生推理能力的 DeepSeek-R1 模型）提出了架构解耦与优化的标准指南。

## 1. 系统现状分析

在目前的 `aidd-agent-backend` 架构中，大模型的思考过程完全依赖于**提示词强迫（Prompt-based Forcing）**机制。

### 1.1 当前实现机制

目前的架构通过以下三个环节实现模型思考和 UI 层的手风琴（"Thinking..."）折叠效果：

1. **Prompt 强约束** (`app/agent/prompts/templates.py`)：
   在系统提示词中硬性规定：`Please put your thinking process inside the <thought>...</thought> tags; put the final answer inside <answer>...</answer>.`
2. **强制预填 (Pre-fill)** (`app/agent/agent.py` & `app/agent/prompt_renderer.py`)：
   在与大模型交互前，后端会在历史消息的最后强制插入一条 `AIMessage(content="<thought>\n")`。这种机制迫使普通模型（如 Qwen, Gemini Flash）必须接续生成思考内容，防止其跳过思考直接输出结果。
3. **前端解析与流式路由** (`app/services/chat_service.py`)：
   后端的流式迭代器（`stream_chat`）在接收到大模型的纯文本流时，通过字符串匹配函数 `_in_thought_block(text)` 实时计算当前处于哪个标签内。如果在 `<thought>` 内，则发送 `thinking_delta` 事件；否则发送 `content_delta` 事件。

### 1.2 存在的问题

这套机制对于不支持原生推理的模型非常有效。然而，如果我们要接入 **DeepSeek-R1 (Reasoner)** 模型，这种机制会成为致命的阻碍：
- **破坏原生推理管线**：DeepSeek-R1 是基于强化学习（RL）训练的推理模型，它会在 OpenAI 兼容的 API 中通过专门的 `reasoning_content` 字段下发思维链。强加 `<thought>` 标签不仅多此一举，还会严重破坏其内置的思考逻辑，导致模型“变笨”或输出混乱的嵌套标签。
- **解析逻辑耦合**：后端流式响应极度依赖于特定的标签字符串格式，无法兼容拥有原生结构化 `thinking` 字段的现代模型。

---

## 2. 优化方案与架构改造

为了使平台能够完美兼容常规模型（Gemini/Qwen/DeepSeek-V3）以及原生推理模型（DeepSeek-R1），我们需要对系统进行**解耦改造**。核心思想是：**将现有的标签解析降级为兼容层，将原生 `reasoning_content` 提升为一等公民。**

### 2.1 提供商层级适配 (LLM Provider)

在 `app/agent/llm_provider.py` 中引入新的 `DeepSeekProvider`（或适配现有的兼容 OpenAI 的提供商）。在解析流式数据时，直接剥离并下发思考事件，无需再依赖后端字符串解析：

```python
# app/agent/llm_provider.py 概念代码
async def stream(self, messages, tools=None) -> AsyncGenerator[StreamChunk, None]:
    stream_response = await self._client.chat.completions.create(
        model=self.model,
        messages=self._to_openai_messages(messages),
        stream=True
    )
    async for chunk in stream_response:
        delta = chunk.choices[0].delta
        
        # 1. 优先提取原生思维链 (适配 DeepSeek-R1)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            yield StreamChunk(type="thinking_delta", content=delta.reasoning_content)
        
        # 2. 提取标准输出
        if delta.content:
            yield StreamChunk(type="content_delta", content=delta.content)
            
        # 3. 工具调用提取等...
```

### 2.2 动态拦截强制预填 (Pre-fill)

在向模型发送消息前，根据配置的模型类别，动态决定是否插入 `<thought>`。

```python
# app/agent/agent.py
def _prepare_messages_for_llm(state_messages: list, model_name: str):
    messages = list(state_messages)
    
    # 针对原生推理模型（如 deepseek-reasoner），跳过人工前缀
    is_reasoning_model = "reasoner" in model_name.lower() or "r1" in model_name.lower()
    
    if not is_reasoning_model:
        # 非推理模型，保留原有的 Fallback 机制
        messages.append(AIMessage(content="<thought>\n"))
        
    return messages
```

### 2.3 精简 System Prompt

DeepSeek-R1 官方强烈建议**去除所有关于“请一步步思考”、“请将思考过程放在特定标签内”的提示词（Zero-shot CoT）**。
我们需要更新 `app/agent/prompt_renderer.py`，加入动态剪裁逻辑：

```python
# app/agent/prompt_renderer.py
def render_system_prompt(model_name: str, ...) -> str:
    base_prompt = ... # 获取通用提示词
    
    if "reasoner" in model_name.lower():
        # 剥离所有关于格式和强迫思考的规则
        base_prompt = remove_formatting_rules(base_prompt)
        
    return base_prompt
```

### 2.4 流式服务层净化

重构 `app/services/chat_service.py`，削减基于正则或字符串检测的 `<thought>` 拦截逻辑：

```python
# app/services/chat_service.py
async for chunk in provider.stream(history):
    # 1. 如果 Provider 已经明确这是一个原生思考片段
    if chunk.type == "thinking_delta":
        yield ServerSentEvent(data=json.dumps({"type": "thinking", "content": chunk.content}))
        continue

    # 2. 兼容老逻辑：如果是不支持原生思考的模型，在内容中自行查找 <thought>
    if chunk.type == "content_delta":
        if _in_thought_block(current_full_text):
             # 提取标签内的内容并当作 thinking 发送
             yield ServerSentEvent(...)
        else:
             # 当作正常文本发送
             yield ServerSentEvent(...)
```

## 3. 预期收益

完成以上改造后：
1. **模型生态扩展**：系统不仅可以完美无损地接入 DeepSeek-R1，还为未来其他具备原生 `reasoning_content` 的模型（如 OpenAI o1/o3）铺平了道路。
2. **逻辑内聚**：思维链的判断逻辑被下沉到了最靠近 API 响应的 Provider 层，极大地简化了应用层 `chat_service.py` 的复杂流式处理代码。
3. **性能最大化**：不同模型将自动选择最符合其预训练目标的上下文格式，保证回复的逻辑性和工具调用的准确率。
