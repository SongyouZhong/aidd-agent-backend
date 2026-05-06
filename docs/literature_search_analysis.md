# AIDD Agent Backend：文献检索工具机制分析报告

本项目 (`aidd-agent-backend`) 集成了多个文献检索核心工具。由于大模型存在严格的上下文（Context Window）限制，系统在底层对检索的数据拉取量和输出内容进行了深度的管控和压缩。

本文档详细总结了各个文献搜索工具的配置、拉取数量逻辑以及搜索结果的压缩与提取机制。

---

## 1. 核心检索工具与数量设置

项目主要集成了三大文献检索平台，均设置了**硬性最大返回数量上限（Semantic Scholar 最多 20 篇，PubMed 和 arXiv 最多 100 篇）**。

### 1.1 Semantic Scholar (语义学者)
Semantic Scholar 是功能最丰富、检索逻辑最深入的工具集（位于 `app/tools/semantic_scholar.py`），具备“超额检索”以保证排序质量的能力。

*   **常规搜索 (`query_semantic_scholar_search`)**
    *   **参数：** 关键词 (`query`)、年份 (`year`)、是否按引用量排序 (`sort_by_citations`，默认 `False`)、期望返回数量 (`max_results`，默认 5 篇)。
    *   **拉取数量：**
        *   当 `sort_by_citations=False` 时：API 拉取数量即为 `max_results` (1-20 篇)。
        *   当 `sort_by_citations=True` 时：**底层 API 会一次性拉取 100 篇论文**。本地按引用量（citationCount）倒序排列后，截取前 `max_results` 篇（最多 20 篇）返回。
*   **引文追溯 (`query_semantic_scholar_citations`)**
    *   **功能：** 查找引用了某篇特定文章的后续论文（正向引用）。
    *   **拉取数量：** **底层 API 会一次性拉取 50 篇引文**。本地进行复杂的打分排序（优先排“高影响力引用”，其次排引文自身的被引数），最后截取前 `max_results` 篇（默认 10 篇，最多 20 篇）返回。
*   **详情获取 (`query_semantic_scholar_paper`)**
    *   **功能：** 通过 `paper_id` 精确获取 **1 篇**文献的深度元数据（含完整摘要和作者等）。

### 1.2 PubMed (生物医学核心库)
专注于生物医学领域的文献检索（位于 `app/tools/literature.py`）。

*   **常规搜索 (`query_pubmed`)**
    *   **参数：** 关键词 (`query`)、期望返回数量 (`max_papers`，默认 50 篇)。
    *   **拉取数量：** 使用 NCBI 的 E-utilities 接口，`retmax` 设置等于 `max_papers`。精确拉取并返回 **最多 100 篇论文**（默认 50 篇）。

### 1.3 arXiv (预印本库)
主要用于物理、计算机以及计算生物/化学的预印本检索（位于 `app/tools/literature.py`）。

*   **常规搜索 (`query_arxiv`)**
    *   **参数：** 关键词 (`query`)、期望返回数量 (`max_papers`，默认 50 篇)。
    *   **拉取数量：** 默认按 `Relevance` (相关度) 排序，直接向 API 请求 `max_papers` 的数量，**最多返回 100 篇论文**（默认 50 篇）。

---

## 2. 数据的提取与压缩机制

项目**并未**使用外部 LLM 对搜索结果进行总结，而是采用了一套定制化的预处理流水线（Preprocessing Pipeline）进行高效的物理提取与压缩。

### 2.1 字段级提取与基础裁剪 (Extraction & Basic Pruning)
在工具内部直接进行 JSON 解析和垃圾数据过滤，提升信息密度。

*   **高密度 Markdown 组装：** 通过内置函数（如 `_format_paper` / `_format_papers`），仅提取**标题、作者、年份、引用数、URL和摘要**，将其拼接为紧凑的 Markdown 格式列表返回。
*   **长文本强制截断：** 例如 Semantic Scholar 的工具中，如果单篇论文摘要过长（> 2000 字符），系统会直接保留前 2000 个字符并在末尾追加 `...`，进行基础裁剪。

### 2.2 全局 Token 级压缩与保护 (Token Cap Compression)
系统中最核心的截断机制，用于保护大模型的上下文窗口不受污染或超载。定义在 `app/tools/preprocess.py`。

*   **全局拦截器 (`@guarded_tool`)：** 所有的文献检索工具都被 `@guarded_tool(max_tokens=MAX_TOOL_TOKENS)` 装饰器包裹，对最终的返回结果进行拦截处理。
*   **硬性天花板限制：** 单个工具的返回文本绝对不能超过 `MAX_TOOL_TOKENS`（系统中硬编码为 **40000 Tokens**，约 120000 个字符）。
*   **粗估与优雅截断：** 
    *   **快速估算：** 采用性能开销极低的粗估算法（`1 Token ≈ 3 字符`）计算文本长度。
    *   **语义边界截断：** 当输出超长时，系统会自动触发 `cap_tokens` 函数。它不会从词汇中间直接腰斩，而是向回寻找最近的**段落边界（`\n\n`）**或**句子边界（`. `）**进行截断，尽可能保留最后一句话的完整语义。
    *   **截断溯源提示：** 在被截断处强行追加提示语：`…[truncated by preprocessing pipeline; raw output stored as sidechain]`，让 Agent 和用户明确知道信息存在遗漏。

---

> **💡 核心总结**
> 
> AIDD Agent 的检索系统采取的是 **"后端超额拉取 + 本地启发式排序 + 高密度信息提取 + 强制物理截断"** 的策略组合。在面对大批量文献时，牺牲了长尾数据的完整性，以此换取系统极快的响应速度和大模型上下文的安全与稳定。
