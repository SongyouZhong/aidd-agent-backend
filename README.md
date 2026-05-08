# AIDD Agent Platform — Backend

> 基于 LangGraph ReAct 架构的 AI 驱动药物研发 (AIDD) 智能体后端，支持多轮对话、动态工具检索、专业子图委托（靶点发现）、GraphRAG 集成以及自动上下文压缩。

## 目录

- [架构概览](#架构概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速启动](#快速启动)
  - [1. 前置条件](#1-前置条件)
  - [2. 启动中间件](#2-启动中间件)
  - [3. 初始化数据库](#3-初始化数据库)
  - [4. 配置环境变量与大模型](#4-配置环境变量与大模型)
  - [5. 安装 Python 环境](#5-安装-python-环境)
  - [6. 启动后端服务](#6-启动后端服务)
- [API 使用指南：完整对话流程](#api-使用指南完整对话流程)
- [离线模式 & 在线模式](#离线模式--在线模式)
- [内置工具与知识库](#内置工具与知识库)
- [运行测试](#运行测试)
- [开发调试工具](#开发调试工具)
- [API 文档](#api-文档)

---

## 架构概览

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────────────────────────────┐
│   Frontend   │───▶│  FastAPI      │───▶│  LangGraph Multi-Agent System             │
│  (REST API)  │◀───│  + JWT Auth   │◀───│  ┌─────────┐    ┌──────────┐           │
└──────────────┘    └──────┬───────┘    │  │ Router  │───▶│ General  │           │
                           │            │  │  Node   │    │  Agent   │           │
                    ┌──────┴───────┐    │  └────┬────┘    └─────┬────┘           │
                    │  Storage      │    │       │               │                │
                    │  ┌─────────┐  │    │  ┌────▼──────────────▼────┐           │
                    │  │ Redis   │  │    │  │ Target Discovery       │ (Sub-Graph)│
                    │  │ (hot)   │  │    │  │ - Literature           │            │
                    │  ├─────────┤  │    │  │ - Composition          │            │
                    │  │SeaweedFS│  │    │  │ - Function             │            │
                    │  │ (cold)  │  │    │  │ - Pathway (GraphRAG)   │            │
                    │  ├─────────┤  │    │  │ - Drugs                │            │
                    │  │Postgres │  │    │  │ - Synthesize           │            │
                    │  │ (meta)  │  │    │  └─────────┬──────────────┘           │
                    │  ├─────────┤  │    │  ┌─────────▼──────────────┐           │
                    │  │ Neo4j   │  │    │  │ Finalize (Citations)   │           │
                    │  │(GraphRAG│  │    │  └────────────────────────┘           │
                    │  └─────────┘  │    └─────────────────────────────────────────┘
                    └──────────────┘
```

**核心特性：**
- **智能意图路由 (Router)**：根据用户提问，自动判断是进入通用对话 (General Agent) 还是启动特定领域的专属流水线 (如 Target Discovery 子图)。
- **Target Discovery 子图**：专门为靶点发现定制的工作流，强制执行按阶段的调研顺序（文献 → 蛋白构成 → 生物学功能 → 通路 → 药物），确保生成符合特定 Schema (5 大维度) 的专业报告，且通过节点级可配置的容错与重试机制（自动处理 API 超时），提高报告成功率。
- **动态工具检索 (Hot-loading)**：通过 `tool_search` 在庞大的工具库中按需挂载 Deferred 工具，保持上下文整洁。
- **GraphRAG 集成**：深度整合 Neo4j 知识图谱（如 WikiPathways 数据），让大模型可以通过自然语言在包含复杂调控机制的图数据库中进行查询。
- **外部知识接入**：通过接入 Semantic Scholar，实现了高达 192k 上下文窗口的海量学术文献自动聚合和内容精读。
- **自动上下文压缩 (Auto-Compaction)**：解决长对话灾难，采用两级策略 — Session Memory 快速压缩 + LLM 深度摘要。
- **混合大模型支持**：无缝支持云端大模型 (Gemini API) 或基于 vLLM 部署的本地超大开源模型 (如 Llama-3.3-70B)。
- **混合存储架构**：Redis 热缓存 + SeaweedFS 冷归档 + PostgreSQL 元数据。

---

## 技术栈

| 层级 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangGraph + LangChain Core |
| LLM 提供者 | Google Gemini (`google-genai`) / 本地 vLLM (Llama-3.3-70B) / 离线 FakeLLM |
| 关系型数据库 | PostgreSQL 16 (用户/会话元数据) |
| 图数据库 | Neo4j (WikiPathways GraphRAG) |
| 缓存 | Redis 7 (消息热缓存) |
| 对象存储 | SeaweedFS S3 (消息冷归档 + 原始工具输出) |
| 迁移 | Alembic |
| 认证 | JWT (python-jose + passlib/bcrypt) |
| 开发调试 | LangGraph Studio + LangSmith |

---

## 项目结构

```
aidd-agent-backend/
├── app/
│   ├── main.py                    # FastAPI 应用入口
│   ├── core/                      # Pydantic Settings & Auth/Exceptions
│   ├── api/                       # REST APIs (/auth, /sessions, etc.)
│   ├── agent/
│   │   ├── agent.py               # 通用 ReAct 图定义
│   │   ├── target_discovery_graph.py # 靶点发现特定子图 (5-Phase 流水线)
│   │   ├── graph.py               # LangGraph Studio 入口（包含 Router）
│   │   ├── llm_provider.py        # Gemini / OpenAI (vLLM) / FakeLLM 适配器
│   │   ├── subagent.py            # 深度研究子代理工具
│   │   ├── context_manager.py     # 自动压缩 (Auto-Compaction)
│   │   └── prompts/               # System Prompts 模板 (包含 Target Discovery)
│   ├── tools/
│   │   ├── registry.py            # Core/Deferred 工具注册表
│   │   ├── search_tool.py         # 动态工具发现 (tool_search)
│   │   ├── literature.py          # PubMed, arXiv
│   │   ├── semantic_scholar.py    # Semantic Scholar 集成
│   │   ├── structure.py           # UniProt, PDB, AlphaFold, InterPro
│   │   ├── disease.py             # OpenTargets, Monarch, QuickGO
│   │   ├── pathway.py             # KEGG, Reactome, STRING
│   │   ├── graph_rag.py           # WikiPathways Neo4j 知识图谱查询
│   │   ├── drug.py                # ChEMBL, PubChem, GtoPdb
│   │   ├── peptide.py             # ChEMBL Peptides
│   │   ├── schemas.py             # 领域知识的 Pydantic 模型 (防 API 臃肿)
│   │   └── base.py                # REST 异步基类
│   ├── storage/                   # Redis + SeaweedFS 客户端
│   ├── models/                    # SQLAlchemy ORM
│   ├── schemas/                   # Pydantic (HTTP Layer)
│   ├── services/                  # Business Logic
│   └── db/                        # Database Sessions
├── scripts/                       # 端到端冒烟测试 (如 test_tdp43_ad.py)
├── docs/                          # 设计与实施文档
├── langgraph.json                 # Studio 配置文件
└── docker-compose.yml             # 环境依赖 (PG+Redis+S3+Neo4j)
```

---

## 快速启动

### 1. 前置条件

- **Docker & Docker Compose**
- **Conda / Mamba**

### 2. 启动中间件

```bash
cd aidd-agent-backend
docker compose up -d
```

确认启动的服务：`aidd-postgres`, `aidd-redis`, SeaweedFS 组件。如果您配置了 Neo4j 服务，也应确保它正在运行。

### 3. 初始化数据库

```bash
# 初始化 SeaweedFS 存储桶
PYTHONPATH=. python scripts/init_seaweedfs_bucket.py

# 运行数据库结构迁移
alembic upgrade head
```

### 4. 配置环境变量与大模型

```bash
cp .env.example .env
```

**编辑 `.env` 选择你的 LLM 驱动引擎：**

- **选项 A：使用云端 Gemini（推荐日常开发）**
  ```env
  GEMINI_API_KEY=your-gemini-key
  ```

- **选项 B：使用本地开源大模型（如 Llama-3.3-70B）**
  需要在另一台机器或当前环境运行 vLLM 提供 OpenAI 兼容的 API：
  ```bash
  # 假设你在 aidd-agent-model 目录下启动了 serve.sh
  # bash serve.sh 启动了 Llama-3.3-70B 服务
  ```
  然后在 `.env` 中配置：
  ```env
  OPENAI_API_KEY=dummy
  OPENAI_API_BASE=http://localhost:8000/v1
  LLM_PROVIDER=openai
  ```

### 5. 安装 Python 环境

```bash
mamba env create -f environment.yml
mamba activate aidd-agent
```

### 6. 启动后端服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 API 文档：[http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

---

## API 使用指南：完整对话流程

详见原 API 规范。流程简述：
1. `POST /api/v1/auth/register` (注册)
2. `POST /api/v1/auth/login` (登录换 Token)
3. `POST /api/v1/sessions` (创建对话)
4. 发送消息，后端会通过意图识别自动将常规闲聊路由给 General Agent，将特定靶点分析（如 "分析 TDP-43"）路由给 Target Discovery Pipeline。

---

## 离线模式 & 在线模式

通过 `AIDD_FORCE_FAKE_LLM=1` 可强制走 FakeLLM 路径进行快速离线单元测试。

---

## 内置工具与知识库

系统按照 `Core` (常驻) 和 `Deferred` (需搜索后动态挂载) 机制管理工具：

| 领域 | 工具名称 | 分类 | 核心功能 |
|---|---|---|---|
| 基础检索 | `tool_search` | 常驻 | 在库中检索需要的 Deferred 工具 |
| **文献检索** | `query_pubmed` | Core | 搜索 PubMed 生物医学文献 |
| | `query_arxiv` | Core | 搜索 arXiv 预印本 |
| | `query_semantic_scholar_search` | Deferred | 搜索 Semantic Scholar 全球文献库 |
| | `query_semantic_scholar_paper` | Deferred | 提取论文详细摘要及核心结论 |
| | `query_semantic_scholar_citations` | Deferred | 获取论文关联的上下游引用参考文献 |
| **蛋白质与结构** | `query_uniprot` | Deferred | 查询 UniProt 蛋白质序列与功能特征 |
| | `query_pdb` & `query_pdb_identifiers` | Deferred | 检索 RCSB 晶体结构并匹配最优高分辨率构象 |
| | `query_alphafold` | Deferred | 查找 AlphaFold 预测的靶点结构模型 |
| | `query_interpro` | Deferred | 查询 InterPro 结构域和蛋白质家族分类 |
| **疾病与功能** | `query_opentarget` | Deferred | 获取 OpenTargets 中基因-疾病关联度评分 |
| | `query_monarch` | Deferred | 查询 Monarch 疾病表型 |
| | `query_quickgo` | Deferred | 获取 QuickGO 的基因本体学 (GO) 注释 |
| **通路与网络** | `query_kegg` | Deferred | KEGG 代谢和信号通路分析 |
| | `query_reactome` | Deferred | Reactome 通路事件查询 |
| | `query_stringdb` | Deferred | STRING 蛋白质-蛋白质相互作用 (PPI) |
| | **`query_wikipathways_graph`** | Deferred | **基于 Neo4j GraphRAG 的复杂基因调控网络查询** |
| **药物研发** | `query_chembl_target_activities` | Deferred | ChEMBL 查询高亲和力（如 IC50 < 1uM）小分子抑制剂 |
| | `query_pubchem` | Deferred | PubChem 化合物验证及 SMILES 结构查询 |
| | `query_gtopdb` | Deferred | IUPHAR / GuideToPharmacology 靶点-配体数据 |
| | `query_chembl_peptides` | Deferred | 查询针对靶点的多肽药物序列及临床进展 |

> **自动容错重试**: 各节点工具调用配置了独立的超时控制（如药物节点配置 300s 以上超时），防止缓慢的外部数据库查询导致整个流水线崩溃。

---

## 运行测试

推荐运行基于特定靶点的端到端集成测试，观察 Agent 路由和子图处理逻辑：

```bash
# 运行 TARDBP/TDP-43 的全面靶点发现评估
PYTHONPATH=. python scripts/test_tdp43_ad.py
```

---

## 开发调试工具

### LangGraph Studio

实时跟踪大模型的思考过程和工具调用堆栈：

```bash
# 安装 CLI
pip install langgraph-cli

# 在根目录启动 Studio
langgraph dev --no-reload
```
浏览器访问 `http://localhost:2024`。支持：
- 节点实时输入输出查看
- JSON State 修改重放 (Human-in-the-loop)
- 查看 `messages` 中所有完整的 Prompt 和响应

### LangSmith — 性能与 Token 监控

在 `.env` 中加入：
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=aidd-agent
```
Agent 每次运行的数据将自动汇总至云端，方便计算 Llama-3.3-70B 或 Gemini 的 Token 成本及系统推理时延。

---

## API 文档

启动后访问：[http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

## License

Internal project — AIDD Agent Platform.
