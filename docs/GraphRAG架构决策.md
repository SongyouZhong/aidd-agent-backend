# GraphRAG 架构决策记录（ADR）

- **日期**：2026-05-10  
- **状态**：已决定 — 保持方案 A（内嵌于 backend），待条件成熟后迁移至方案 B  

---

## 背景

项目中存在两个独立仓库：

| 仓库 | 职责 |
|------|------|
| `aidd-agent-GraphRAG` | **离线数据管线**：下载 WikiPathways GPML → 解析 → 写入 Neo4j。一次性脚本 + Neo4j 容器（docker-compose）。 |
| `aidd-agent-backend` | **在线 Agent 服务**：`app/tools/graph_rag.py` 通过 Bolt 直连 Neo4j，使用 `GraphCypherQAChain` 执行自然语言 → Cypher 查询。 |

问题的核心在于：**GraphRAG 的查询逻辑（Cypher 生成 + LLM 调用）目前写死在 backend 中，与 GraphRAG 项目的数据写入能力割裂。** 应当如何合理划分边界？

---

## 三种候选方案

### 方案 A：保持现状 — Tool 内嵌于 backend

```
agent-backend ──Bolt──> Neo4j（由 GraphRAG 项目部署）
   └─ tools/graph_rag.py  (LangChain GraphCypherQAChain)
```

**优点**
- 实现最简，无额外网络跳转
- LLM 与 agent 共用同一套配置（已通过 `get_graph_rag_llm()` 统一）
- Cypher Prompt 与 agent schema 紧耦合，schema 调优在同一仓库内完成
- 无跨进程序列化开销

**缺点**
- backend 需要 `langchain_neo4j` + Neo4j 驱动依赖
- 若多个消费者需要查询 GraphRAG，需各自重复实现

---

### 方案 B：GraphRAG 项目暴露 HTTP 查询 API（推荐的演进方向）

```
agent-backend ──HTTP POST /query──> graphrag-service ──Bolt──> Neo4j
                  (新增 FastAPI 服务，部署于 GraphRAG 项目)
```

**优点**
- 真正的职责分离：GraphRAG 项目同时负责**写入**和**查询**，成为知识图谱唯一所有者
- backend 彻底摆脱 `langchain_neo4j` / `neo4j-driver` 依赖
- 多消费方（前端直查、批处理、其他 Agent）可复用同一查询接口
- 可在查询层独立做缓存、限流、审计，不污染 agent 代码
- 独立部署与扩缩容，schema 变更不影响 backend 发布节奏

**缺点**
- 需要新增 FastAPI 服务、Dockerfile、k3s 部署清单
- LLM 配置需在两个项目间同步（通过统一的环境变量注入）
- 多一跳网络延迟，需设计合理的超时与重试策略

---

### 方案 C：拆出共享 Python 包

```
aidd-graphrag-py（PyPI 私服 / git submodule）
  ├─ build_graph.py
  └─ query.py（供 agent import）
```

**优点**：代码单一来源，进程内调用无网络开销  
**缺点**：需要维护内部包发布流程；backend 仍持有 Neo4j 依赖；升级耦合

---

## 决策

**当前采用方案 A。**

原因：

1. **Neo4j 已是独立服务**（GraphRAG 项目 docker-compose 管理）——数据所有权在物理上已隔离，schema 演化不依赖 backend。
2. **单一消费者**——目前只有一个 agent 使用 GraphRAG，过早拆服务属于 over-engineering。
3. **Cypher Prompt 仍在调优**——Prompt 模板（节点标签、关系类型、示例 Cypher）与 agent 工具描述强相关，同仓迭代效率更高。
4. **LLM 配置已统一**——`get_graph_rag_llm()` 使 graph_rag.py 与 agent 共享同一套 `LLM_PRIORITY` 配置，主要耦合问题已解决。

---

## 触发迁移至方案 B 的条件

满足以下任一条件时，应启动方案 B 的实施：

- [ ] 出现第二个 GraphRAG 消费方（如前端直接查询、离线批处理）
- [ ] 需要在查询层做独立的缓存 / 限流 / 审计
- [ ] backend 部署体积或启动时间因 `langchain_neo4j` 产生明显影响
- [ ] GraphRAG schema 变更频率超过 backend 发布节奏，需要解耦

---

## 已完成的相关修改

| 文件 | 变更 | 日期 |
|------|------|------|
| `app/agent/llm_provider.py` | 新增 `get_graph_rag_llm()`，GraphRAG 的 LLM 选择遵循 `LLM_PRIORITY`，不再硬编码 | 2026-05-10 |
| `app/tools/graph_rag.py` | 移除本地 `_build_graph_llm()`，改为调用 `get_graph_rag_llm()` | 2026-05-10 |
