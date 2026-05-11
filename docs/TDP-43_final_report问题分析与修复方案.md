# TDP-43 final_report 问题分析与修复方案

> 分析对象：[logs/target_discovery/20260510T103839_TDP-43/final_report.json](../logs/target_discovery/20260510T103839_TDP-43/final_report.json)
> 涉及代码：[app/agent/target_discovery_graph.py](../app/agent/target_discovery_graph.py)、[app/agent/prompts/target_discovery.py](../app/agent/prompts/target_discovery.py)、[app/tools/disease.py](../app/tools/disease.py)、[app/tools/drug.py](../app/tools/drug.py)、[app/tools/pathway.py](../app/tools/pathway.py)
> 日期：2026-05-10

---

## 0. 现象速览

`final_report.json` 中关键字段缺失或异常：

| 字段 | 实际产出 | 期望 |
| --- | --- | --- |
| `papers` | `[]` | ≥ 3 篇代表性原文 |
| `small_molecule_drugs` | `[]` | ≥ 3 个（mitoxantrone、CHEMBL4635203 等已被工具返回） |
| `peptide_drugs` / `antibody_drugs` | `[]` | 至少给出"数据源不足"的结构化说明 |
| `pathways` | 含 1 条全 `null` 的 Reactome 行 | 不应出现空行 |
| `disease_associations[].score` | 部分为 `null`（来自 `"N/A (...)"` 字符串被丢弃） | 数值或 `null`（语义统一） |
| `notes` | 5 条超时/解析失败 + 1 条 `BadRequestError` | 仅必要的诊断信息 |

`notes` 中最严重的一条：

```
Node [drugs] fallback summarization failed:
BadRequestError('... An assistant message with \'tool_calls\' must be followed by tool messages
responding to each \'tool_call_id\'. (insufficient tool messages following tool_calls message) ...')
```

这条错误是导致 drugs 节点最终 0 输出的直接原因。

---

## 1. 问题清单（按严重度排序）

### P0-1 Drugs 节点 fallback 抛 `BadRequestError`，导致整个节点产出为空

**现象**
```
Node [drugs] timed out (>300s), attempted partial summarization.
Node [drugs] fallback summarization failed: BadRequestError(...
  "insufficient tool messages following tool_calls message" ...)
```

**根因**
[app/agent/target_discovery_graph.py](../app/agent/target_discovery_graph.py) 中 `_run_node_loop` 的循环结构是：
```python
ai_msg = AIMessage(content=last_text, tool_calls=[...])
messages.append(ai_msg)            # ← 先把 AI 消息（含 tool_calls）追加进去
if not resp.tool_calls:
    break
for tc in resp.tool_calls:         # ← 再串行执行工具，逐个 append ToolMessage
    tool_out = await _invoke_tool(...)
    messages.append(ToolMessage(...))
```
而外层 `_safe_node` 用 `asyncio.wait_for(_run_node_loop(...), timeout=300)` 强制取消。当 timeout 命中、`for tc in resp.tool_calls` 还没跑完时，`messages` 里就出现"AI 消息已带 N 个 tool_calls，但 ToolMessage 不足 N 条"的非法历史。

紧接着 `_safe_node` 的超时分支直接复用同一份 `messages` 去喂 fallback：
```python
messages.append(HumanMessage("Execution time limit reached. ... output <answer>"))
final_resp = await provider.generate(messages=messages, tools=None)
```
DeepSeek/OpenAI 协议层校验 `tool_calls ↔ tool` 一一对应，于是返回 400 直接报死。

**修复方案**

在 fallback 调用 LLM 前 **清洗消息历史**，工具函数：

```python
def _sanitize_for_summary(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure every AIMessage.tool_calls id has a matching ToolMessage after it.

    Strategy: walk forward, for each AIMessage with tool_calls collect the set
    of ids; gather following ToolMessage ids until the next AIMessage / end;
    for any missing id, insert a placeholder ToolMessage(content='[cancelled]').
    If the trailing message is an AIMessage with unfulfilled tool_calls and no
    ToolMessage follows, drop it.
    """
```

调用点（两处）：
1. `_run_node_loop` 内"safety net"那段（强制最后一次无工具汇总），替换 `messages.append(HumanMessage(...))` 之前一行。
2. `_safe_node` 的 `except asyncio.TimeoutError` 分支同上。

同时，建议把"先 append AI 消息再跑工具"改成 **原子提交**：

```python
# 收集所有 ToolMessage 后一次性 extend，失败/取消则整组回滚
tool_msgs: list[ToolMessage] = []
try:
    for tc in resp.tool_calls:
        out = await _invoke_tool(tc.name, tc.args)
        tool_msgs.append(ToolMessage(content=out, name=tc.name, tool_call_id=tc.id))
except asyncio.CancelledError:
    # 不写入 ai_msg，整步丢弃
    raise
messages.append(ai_msg)
messages.extend(tool_msgs)
```

这样即便被 `wait_for` 取消，`messages` 仍是合法的 ReAct 序列。

---

### P0-2 Drugs 工具 `query_gtopdb` 返回 503 不重试，被 LLM 当作"无数据"

**现象**
```json
{ "name": "query_gtopdb",
  "content": "GtoPdb query failed for 'TDP-43': retryable status 503" }
```
连续两次都是 503，节点没有自动重试，LLM 误以为 GtoPdb 无 TDP-43 条目。

**根因**
[app/tools/drug.py](../app/tools/drug.py) 中 `query_gtopdb` 把 5xx 直接拼成字符串返回，没有指数退避；[app/tools/pathway.py](../app/tools/pathway.py) 的 Reactome 同样问题（导致 final_report 里出现一条全 `null` 的 Reactome 行）。

**修复方案**

提取公共重试装饰器（或就地实现）：

```python
async def _retry(coro_factory, *, retries: int = 3, base: float = 1.0):
    last_exc = None
    for i in range(retries):
        try:
            return await coro_factory()
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                raise
            last_exc = e
            await asyncio.sleep(base * (2 ** i))
    raise last_exc
```

应用：`query_gtopdb`、`query_reactome`、`query_kegg`（KEGG 同样偶发 5xx）。

---

### P0-3 Literature 节点超时 → `papers: []`

**现象**
工具其实已经返回了关键文献（Neumann 2006 Science PMID:17023659、Ou 1995 J Virol PMID:7745706、Sreedharan 2008 Science PMID:18309045、Kabashi 2008 Nat Genet PMID:18372902、Chen-Plotkin 2010 Nat Rev Neurol PMID:20234357），但 LLM 仍在重复搜索导致 120 s 超时，fallback 也没解析出 JSON。

**根因（多重）**
1. 节点预算 120 s + `MAX_NODE_STEPS=10`，prompt 里没有"达到 N 条立刻收尾"的硬规则。LLM 不停做无意义的"按 PMID 反查自己已经看过的论文"。
2. literature 在 composition 之前跑（拓扑硬编码），拿不到 verified `gene_symbol="TARDBP"`，PubMed 查询全用自由文本"TDP-43 highly cited review"，召回质量低。
3. 工具调用没有 (tool_name, normalized_args) 级缓存，相同的 `query_pubmed` 不同表述被反复调用。

**修复方案**

A. **拓扑改写**（在 `build_target_discovery_graph` 中）：
```
START → composition → [literature, function, pathway, drugs] → synthesize → END
```
让 composition 作为唯一前置节点先把 UniProt+gene_symbol 跑出来，下游 4 节点扇出并行（`graph.add_edge("composition", "literature")` 等四条；`graph.add_edge("literature", "synthesize")` 等四条）。

B. **literature 的 `prior_context` 同时携带 `gene_symbol`**：
```python
context = "UniProt accession: Q13148\nGene symbol (use in PubMed as TARDBP[gene]): TARDBP"
```
prompt 内补一句："Prefer queries of the form `TARDBP[gene] AND ...`."

C. **`_run_node_loop` 增加参数级缓存**：
```python
seen: dict[tuple[str, str], str] = {}
key = (tc.name, json.dumps(tc.args, sort_keys=True))
if key in seen:
    tool_out = seen[key]
else:
    tool_out = await _invoke_tool(...)
    seen[key] = tool_out
```

D. **prompt 加硬性早停规则**（[app/agent/prompts/target_discovery.py](../app/agent/prompts/target_discovery.py) `LITERATURE_NODE_PROMPT`）：
> "Stop the loop and output `<answer>` as soon as you have ≥3 papers with PMID. Do NOT verify by re-querying the same PMIDs."

E. **节点参数收紧**：literature 改为 `max_steps=5, timeout=90`。

---

### P0-4 Function 节点超时 + `score` 字段类型混杂

**现象**
- 节点超时（>120 s），fallback 才出 JSON。
- raw_output 中两条 Monarch 关联写的是 `"score": "N/A (Monarch entity — disease defined by TDP-43 pathology)"`，synthesize 阶段被丢弃成 `null`，但 schema 设计本应是 `float | null`。

**根因**
1. QuickGO 调用三次（FPC / P / C 三 aspect）+ Monarch 多次组合，每次 2-3 s 累加超时。
2. QuickGO 返回的 `go_name` 全是 `null`（见 P1-5），LLM 反复纠结于"再查一次 GO 名"。
3. `FUNCTION_NODE_PROMPT` 没明确"score 必须是 0-1 浮点或 null，禁止字符串解释"。

**修复方案**
1. 在 prompt 中明确：`score MUST be a float in [0,1] or null. Do NOT put explanatory strings here; put narration in function_narrative.`
2. 在 synthesize 后置处理（[app/agent/target_discovery_graph.py](../app/agent/target_discovery_graph.py) `synthesize_node`）增加一段：
   ```python
   for d in report.get("disease_associations", []):
       s = d.get("score")
       if isinstance(s, str):
           try: d["score"] = float(s)
           except ValueError: d["score"] = None
   ```
3. 节点预算改 `max_steps=5, timeout=90`。
4. 配合 P1-5 的 QuickGO 名称补全，可省掉至少 1 轮调用。

---

### P1-5 `query_quickgo` 不返回 GO term 名称

**现象**
function.json 中所有 `"go_name": null`，LLM 只能凭记忆把 `GO:0008380` 翻译成"RNA splicing"，存在幻觉风险。

**根因**
[app/tools/disease.py](../app/tools/disease.py) 的 `query_quickgo` 只调了 `/QuickGO/services/annotation/search`，没有再去 `/QuickGO/services/ontology/go/terms/{id}` 拉 label。

**修复方案**
1. 收集所有 `go_id`，批量调一次 `GET /QuickGO/services/ontology/go/terms/{ids}`（QuickGO 支持逗号分隔），拿到 `name` 字段。
2. 同时按 `(go_id, evidence)` 去重（当前同一个 `GO:0005515` 出现 16 条 IPI 完全重复占用了输出额度）。
3. 单次响应 cap 到例如 40 条，防止把 prompt 撑爆。

---

### P1-6 Drugs 节点 `peptide_drugs` / `antibody_drugs` 数据源不足无结构化反馈

**现象**
final_report 中 `peptide_drugs: []` 与 `antibody_drugs: []`，但 ChEMBL 工具明确返回过：
```json
{ "note": "DRAMP/THPdb not wrapped (no public REST). Use literature search for AMPs." }
```
该提示被淹没，前端无法解释"为什么是空的"。

**修复方案**
- Synthesize prompt 增加字段：
  ```json
  "data_source_gaps": [
    {"category": "peptide_drugs", "reason": "DRAMP/THPdb not wrapped"},
    ...
  ]
  ```
- `drugs_node` 在节点输出基础上把 ChEMBL/peptide 工具的 `note` 显式传给 synthesize（写入 `sub_results["drugs"]["_notes"]`）。

---

### P1-7 Pathway 结果含全 `null` 的 Reactome 行

**现象**
```json
{ "source": "Reactome", "external_id": null, "name": null, "url": null, "interactors": [] }
```

**根因**
Reactome 工具 5xx 时仍返回了一条占位 dict；synthesize 没过滤。

**修复方案**
1. [app/tools/pathway.py](../app/tools/pathway.py) `query_reactome` 失败时显式 raise / 返回 `{"error": "..."}`，不要返回半成品。
2. `synthesize_node` 后置过滤：
   ```python
   report["pathways"] = [
       p for p in report.get("pathways", [])
       if p.get("name") or p.get("external_id")
   ]
   ```

---

### P2-8 单工具调用没有独立超时

**现象**
GtoPdb 503 后客户端等到底层 httpx 默认超时（可能 30+ s），单次 fan-out 中只要一个慢调用就拖爆整个节点预算。

**修复方案**
[app/agent/target_discovery_graph.py](../app/agent/target_discovery_graph.py) `_invoke_tool`：
```python
TOOL_TIMEOUT_DEFAULT = 30.0
async def _invoke_tool(name, args):
    impl = default_registry.get(name)
    if impl is None:
        return f"[error] tool '{name}' not loaded"
    try:
        return await asyncio.wait_for(_do_invoke(impl, args), timeout=TOOL_TIMEOUT_DEFAULT)
    except asyncio.TimeoutError:
        return f"[tool timeout: {name} >{TOOL_TIMEOUT_DEFAULT:.0f}s]"
```

---

### P2-9 全串行管道导致预算叠加

**现象**
当前 5 节点串行最坏 120+120+120+300+300 = ~21 min；本次实际跑了 3 个超时节点。

**修复方案**
见 P0-3 节 A：composition 之后并行 4 路扇出，最坏只剩 `max(node)` ≈ 2 min。

---

### P2-10 重复参数无去重

**现象**
drugs 节点连续调了 `target="Q13148"` 与 `target="TARDBP"`，命中同一个 `CHEMBL2362981`，浪费两轮。

**修复方案**
见 P0-3 节 C：在 `_run_node_loop` 内做 `(name, args)` 级缓存。

---

## 2. 修复落地顺序

| 优先级 | 修复项 | 涉及文件 | 风险 |
| --- | --- | --- | --- |
| P0 | 1. 消息历史清洗 + 原子提交 | `target_discovery_graph.py` (`_run_node_loop`, `_safe_node`) | 低，只动 fallback 路径 |
| P0 | 2. GtoPdb / Reactome / KEGG 重试 | `tools/drug.py`, `tools/pathway.py` | 低 |
| P0 | 3. 拓扑改写 + literature 注入 gene_symbol | `target_discovery_graph.py` | 中，需重测一遍多目标 |
| P0 | 4. score 类型校验 + prompt 收紧 | `target_discovery_graph.py`, `prompts/target_discovery.py` | 低 |
| P1 | 5. QuickGO 名称补全 + 去重 | `tools/disease.py` | 低 |
| P1 | 6. data_source_gaps 字段 | `target_discovery_graph.py`, prompt | 低 |
| P1 | 7. Pathway 空行过滤 | `target_discovery_graph.py` | 低 |
| P2 | 8. 单工具超时 | `target_discovery_graph.py` (`_invoke_tool`) | 低 |
| P2 | 9. 节点预算 / max_steps 收紧 | 同上 | 低 |
| P2 | 10. (name, args) 缓存 | `_run_node_loop` | 低 |

---

## 3. 验证方案

1. **回归单测**：mock provider 制造"AI tool_calls 后被 wait_for 取消"的场景，断言 `_safe_node` 返回有效 JSON，不再抛 `BadRequestError`。覆盖 P0-1。
2. **TDP-43 端到端重跑**：
   - `papers ≥ 3`（至少包含 PMID 17023659、7745706、18309045）
   - `small_molecule_drugs ≥ 1`（mitoxantrone + CHEMBL4635203）
   - `notes` 不再含 "fallback summarization failed"
   - `pathways` 不再含全 null 行
   - 总耗时 < 8 min
3. **EGFR 端到端重跑**：验证并行拓扑下 `prior_context` 仍正确传播。
4. **QuickGO 校验**：≥ 50% 的 GO 注释带 `go_name`。

---

## 4. 范围与假设

- **范围内**：仅改动上述 5 个文件 + 对应 prompt。
- **范围外**：换 LLM 提供商、引入 DRAMP/THPdb 新数据源、改前端、改 `app/api/targets.py` 暴露的 schema。
- **假设**：`default_registry` 的工具列表不变；DeepSeek 的 `tool_calls ↔ tool` 协议校验在可预见时间内不变。
