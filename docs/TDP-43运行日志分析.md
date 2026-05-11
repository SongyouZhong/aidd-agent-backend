# TDP-43 靶点发现 — 运行日志分析与测试评估

> 运行时间：2026-05-08T13:21:19 ~ 13:26:46（约 5m27s）
> 日志目录：`logs/target_discovery/20260508T132121_TDP-43/`

---

## 一、测试期望 vs 实际输出

**测试输入**：`请帮我做TDP-43 靶点分析：原始发现论文、蛋白质结构、与阿尔兹海默症/ALS 的疾病关联、作用通路、以及目前有哪些针对该靶点的药物`

| # | 期望项 | 满足程度 | 评级 |
|---|--------|---------|------|
| 1 | 靶点来源（原始论文链接） | ✅ 完全满足 | ⭐⭐⭐ |
| 2 | 蛋白质构成（序列 + PDB code） | ⚠️ 部分满足 | ⭐⭐ |
| 3 | 靶点如何影响病症（bio function） | ⚠️ 部分满足 | ⭐⭐ |
| 4 | 通过什么 pathway 发生作用 | ✅ 基本满足 | ⭐⭐½ |
| 5 | 有效药物（SMILES / 序列） | ⚠️ 部分满足 | ⭐⭐ |

**整体完成度：约 60–65%**

### 1️⃣ 靶点来源（原始论文链接）— ✅ 完全满足

提供了 4 篇关键论文，包含 title、year、PMID、DOI 和 PubMed 链接：

| 论文 | 年份 | PMID |
|------|------|------|
| 原始发现论文（HIV-1 TAR DNA binding） | 1995 | 7608134 |
| FTLD/ALS 中的 TDP-43 鉴定 | 2006 | 17023659 |
| TDP-43 突变与家族性 ALS | 2008 | 18309045 |
| TDP-43 & FUS/TLS 综述 | 2010 | 20400460 |

> 💡 超出预期——不仅提供了 1995 年的原始发现论文，还附带了疾病关联的里程碑论文。

### 2️⃣ 蛋白质构成（序列 + PDB code）— ⚠️ 部分满足

| 维度 | 状态 | 详情 |
|------|------|------|
| 蛋白数量 | ✅ | 报告了 1 个蛋白 (Q13148) |
| 氨基酸序列 | ✅ | 提供了完整的 414 aa 序列 |
| PDB code | ⚠️ **不完整** | 仅列出 `4IUF`（实际有 25 个） |
| 蛋白域注释 | ✅ | 列出了 5 个 InterPro 结构域 |
| AlphaFold | ✅ | 提供了 AlphaFold ID |

**主要缺口**：`query_pdb_identifiers` 实际返回了 25 个 PDB ID，但 Prompt 约束 LLM 只保留了 1 个。

### 3️⃣ 靶点如何影响病症（bio function）— ⚠️ 部分满足

- ✅ 提供了 TDP-43 的功能概述（RNA 代谢、转录调控等）
- ✅ 引用了 GO 术语
- ✅ 列出了 5 条疾病关联（OpenTargets + Monarch）
- ❌ **没有阿尔茨海默症（Alzheimer's Disease）的疾病关联**
- ❌ 功能叙述缺乏具体致病机制描述

### 4️⃣ 通过什么 pathway 发生作用 — ✅ 基本满足

提供了 3 条 KEGG 通路（含 URL 和互作蛋白），Reactome 返回 404（无数据）。

### 5️⃣ 有效药物 — ⚠️ 部分满足

| 类型 | 状态 |
|------|------|
| 小分子（含 SMILES） | ✅ 3 个（CHEMBL4635203, MITOXANTRONE, THIOCTIC ACID） |
| 多肽（含序列） | ❌ 空数组 |
| 抗体（含序列） | ❌ 空数组 |

---

## 二、执行时间线

```
13:21:19 ─ Run started
13:21:21 ─ intent_router → Gemini (2s) ✅
13:21:21 ─ literature starts
13:23:21 ─ literature ⏰ TIMEOUT (120s)
13:23:37 ─ literature fallback summarization complete
13:23:39 ─ composition starts
13:24:04 ─ composition complete ✅ (~25s)
13:24:13 ─ function starts
13:24:32 ─ function complete ✅ (~19s)
13:24:36 ─ pathway starts
13:25:17 ─ pathway complete ✅ (~41s)
13:25:25 ─ drugs starts
13:26:32 ─ drugs complete ✅ (~67s)
13:26:32 ─ synthesize starts
13:26:46 ─ synthesize complete ✅ (~14s)
─────────────────────────────
Total: ~5m27s
```

---

## 三、问题详析

### 问题 1：Literature 节点超时 (120s) 🔴 P0

**现象**：第一次 Gemini 调用花了整整 2 分钟 (13:21:21 → 13:23:21)。LLM 返回后立即调用 PubMed，但此时已触发 120s 超时。literature 节点只完成了 1 次 PubMed 搜索。

**根因**：`NODE_TIMEOUT_SECONDS = 120.0` 对于需要多次 LLM + API 交互的节点来说不够。

**修复建议**：
- 增加 `NODE_TIMEOUT_SECONDS` 至 180-240s
- 或优化 prompt：减少首次 LLM 调用的复杂度

---

### 问题 2：PDB 结果被 Prompt 约束为仅 1 个 🔴 P0

**现象**：`query_pdb_identifiers` 返回了 25 个 PDB IDs：
```json
["1WF0", "2CQG", "2N2C", "2N3X", "2N4G", "2N4H", "2N4P", "4BS2", "4IUF", "4Y00", "4Y0F", "5MDI", "5MRG", ...]
```
但 LLM 只选了 `4IUF`（分辨率 2.75 Å），最终报告 `pdb_ids: ["4IUF"]`。

**根因**：`COMPOSITION_NODE_PROMPT` 中写了：
> `CRITICAL CONSTRAINT: You MUST EXACTLY select ONLY the TOP 1 highest resolution PDB ID.`

**修复建议**：修改 prompt 为"只对 TOP 1 调用 `query_pdb` 获取详情，但在最终 JSON 的 `pdb_ids` 中列出所有 `query_pdb_identifiers` 返回的 ID"。

---

### 问题 3：缺少阿尔茨海默症 (AD) 疾病关联 🟡 P1

**现象**：OpenTargets 返回的 top 疾病关联中没有 Alzheimer's（只有 ALS + FTD），Monarch 也未补充。

**根因**：
1. AD 和 TDP-43 的关联分数较低，可能未排进 top 25
2. 用户原始提问中的疾病名称（"阿尔兹海默症"）未传递给 function_node

**修复建议**：将用户原始提问中的疾病关键词传入 function 节点的 `prior_context`。

---

### 问题 4：Neo4j 知识图谱查询返回空结果 🟢 P2

**现象**：
```
Generated Cypher: MATCH (p:Pathway)-[:CONTAINS]->(g) WHERE g.name = 'TARDBP' RETURN p.name
Full Context: []
```
执行了两次，都返回空。

**根因**：LLM 没有先调用 `query_graph_schema` 了解 schema 就直接查询了。Cypher 中的 label/relationship 可能不正确。

---

### 问题 5：Drug 节点 API 404 🟡 P1

#### 5a. PubChem 查询用了错误的名称
```
❌ compound/name/TDP-43   → 404  （蛋白质名，非化合物）
❌ compound/name/NI-205   → 404  （ASO 药物代号）
❌ compound/name/BIIB105  → 404  （ASO 药物代号）
```

LLM 把蛋白质名和 ASO 药物代号当小分子化合物查 PubChem。

#### 5b. IUPHAR/GtoPdb 404
TDP-43 不是传统药物靶标（受体/酶/离子通道），IUPHAR 未收录，3 次查询全部 404。

**修复建议**：改进 `query_pubchem` 的 tool description，明确"仅用于小分子化合物名称"。

---

### 问题 6：ChEMBL 活性查询策略单一 🟢 P2

**现象**：查询了 KI 和 KD 但结果可能为空，没有尝试 EC50 或降低 `pchembl_min` 阈值。

**修复建议**：Prompt 中建议 "如果首次查询结果不足 3 个化合物，尝试降低 pchembl_min 到 5.0 或查询 EC50"。

---

## 四、修复优先级总结

| 优先级 | 问题 | 影响 | 修复难度 |
|--------|------|------|----------|
| 🔴 P0 | Literature 超时 (120s 不够) | 论文数据不完整 | 低 — 改 timeout 值 |
| 🔴 P0 | PDB IDs 被 prompt 限制为 1 个 | 蛋白质结构信息丢失 | 低 — 改 prompt |
| 🟡 P1 | 用户疾病关键词未传入 function 节点 | 漏掉 AD 关联 | 中 — 改架构传参 |
| 🟡 P1 | PubChem 被错误调用 | 浪费调用 + 错误日志 | 低 — 改 tool description |
| 🟢 P2 | Neo4j 图谱空结果 | 通路信息不全 | 高 — 需检查数据 |
| 🟢 P2 | ChEMBL 查询策略单一 | 药物覆盖不足 | 低 — 改 prompt |
