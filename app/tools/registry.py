"""Tool registry with core/deferred separation + keyword search.

Implements design doc §7.3.1 (动态工具检索 / Hot-loading):

    * CORE tools live in the System Prompt at all times.
    * DEFERRED tools are searched via ``tool_search`` and dynamically
      mounted into the next round of context.

A LangGraph node can call ``default_registry.bind_active(state)`` to
materialise the concrete ``BaseTool`` list to expose to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.tools.database import query_chembl, query_uniprot
from app.tools.disease import query_monarch, query_opentarget, query_quickgo
from app.tools.drug import (
    query_chembl_target_activities,
    query_gtopdb,
    query_pubchem,
)
from app.tools.literature import query_arxiv, query_pubmed
from app.tools.semantic_scholar import (
    query_semantic_scholar_citations,
    query_semantic_scholar_paper,
    query_semantic_scholar_search,
)
from app.tools.pathway import query_kegg, query_reactome, query_stringdb
from app.tools.graph_rag import query_wikipathways_graph
from app.tools.peptide import query_chembl_peptides
from app.tools.structure import (
    query_alphafold,
    query_interpro,
    query_pdb,
    query_pdb_identifiers,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# Categorisation drives whether the tool is in the default System Prompt.
CORE_TOOL_NAMES = {"query_pubmed", "query_arxiv"}
DEFERRED_TOOL_NAMES = {
    "query_semantic_scholar_search",
    "query_semantic_scholar_paper",
    "query_semantic_scholar_citations",
    "query_uniprot",
    "query_chembl",
    # Phase B — structure
    "query_pdb",
    "query_pdb_identifiers",
    "query_alphafold",
    "query_interpro",
    # Phase B — disease / function
    "query_opentarget",
    "query_monarch",
    "query_quickgo",
    # Phase B — pathway
    "query_kegg",
    "query_reactome",
    "query_stringdb",
    "query_wikipathways_graph",
    # Phase B — drug
    "query_chembl_target_activities",
    "query_pubchem",
    "query_gtopdb",
    # Phase B — peptide
    "query_chembl_peptides",
}


@dataclass
class ToolEntry:
    name: str
    description: str
    keywords: list[str]
    category: str  # "core" | "deferred"
    impl: "BaseTool"


@dataclass
class ToolRegistry:
    """In-process registry. One per app; not thread-shared state."""

    entries: dict[str, ToolEntry] = field(default_factory=dict)

    # --- registration --------------------------------------------------

    def register(
        self,
        impl: "BaseTool",
        *,
        category: str,
        keywords: list[str] | None = None,
    ) -> None:
        name = impl.name
        self.entries[name] = ToolEntry(
            name=name,
            description=(impl.description or "").strip(),
            keywords=[k.lower() for k in (keywords or [])],
            category=category,
            impl=impl,
        )

    # --- introspection -------------------------------------------------

    def core_tools(self) -> list["BaseTool"]:
        return [e.impl for e in self.entries.values() if e.category == "core"]

    def deferred_tools(self) -> list["BaseTool"]:
        return [e.impl for e in self.entries.values() if e.category == "deferred"]

    def get(self, name: str) -> "BaseTool | None":
        e = self.entries.get(name)
        return e.impl if e else None

    # --- search --------------------------------------------------------

    def search(self, query: str, *, top_k: int = 3) -> list[ToolEntry]:
        """Tiny scoring: keyword hits + token overlap with description."""
        q_tokens = {t for t in query.lower().split() if len(t) >= 3}
        if not q_tokens:
            return []
        scored: list[tuple[float, ToolEntry]] = []
        for entry in self.entries.values():
            if entry.category != "deferred":
                continue
            score = 0.0
            for kw in entry.keywords:
                if any(kw in tok or tok in kw for tok in q_tokens):
                    score += 2.0
            desc_tokens = {t.lower().strip(",.()") for t in entry.description.split()}
            score += len(q_tokens & desc_tokens) * 0.5
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # --- runtime context ----------------------------------------------

    def bind_active(self, hot_loaded: set[str] | None = None) -> list["BaseTool"]:
        """Return the tool list to expose to the LLM for the next turn.

        ``hot_loaded`` is the set of deferred tool names previously
        surfaced via ``tool_search`` and approved for mounting.
        """
        active = self.core_tools()
        if hot_loaded:
            for name in hot_loaded:
                e = self.entries.get(name)
                if e and e.category == "deferred":
                    active.append(e.impl)
        return active


def _build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        query_pubmed,
        category="core",
        keywords=["pubmed", "literature", "paper", "biomedical", "clinical"],
    )
    reg.register(
        query_arxiv,
        category="core",
        keywords=["arxiv", "preprint", "paper", "literature"],
    )
    # --- Semantic Scholar --------------------------------------------
    reg.register(
        query_semantic_scholar_search,
        category="deferred",
        keywords=["semantic scholar", "literature", "paper", "search", "文献", "检索", "论文"],
    )
    reg.register(
        query_semantic_scholar_paper,
        category="deferred",
        keywords=["semantic scholar", "paper details", "文献详情", "论文详情"],
    )
    reg.register(
        query_semantic_scholar_citations,
        category="deferred",
        keywords=["semantic scholar", "citation", "reference", "引用", "参考文献"],
    )
    reg.register(
        query_uniprot,
        category="deferred",
        keywords=[
            "uniprot", "protein", "sequence", "gene",
            "靶点", "蛋白", "蛋白质", "domain",
        ],
    )
    reg.register(
        query_chembl,
        category="deferred",
        keywords=[
            "chembl", "compound", "molecule", "smiles",
            "ic50", "bioactivity", "drug", "化合物", "活性",
        ],
    )
    # --- Phase B: structure ------------------------------------------
    reg.register(
        query_pdb,
        category="deferred",
        keywords=["pdb", "structure", "crystal", "rcsb", "结构", "晶体"],
    )
    reg.register(
        query_pdb_identifiers,
        category="deferred",
        keywords=["pdb", "uniprot", "xref", "search", "结构搜索"],
    )
    reg.register(
        query_alphafold,
        category="deferred",
        keywords=["alphafold", "predicted structure", "model", "预测结构"],
    )
    reg.register(
        query_interpro,
        category="deferred",
        keywords=["interpro", "domain", "结构域", "family"],
    )
    # --- Phase B: disease / function ---------------------------------
    reg.register(
        query_opentarget,
        category="deferred",
        keywords=[
            "opentargets", "target", "disease", "association",
            "靶点", "疾病", "关联",
        ],
    )
    reg.register(
        query_monarch,
        category="deferred",
        keywords=["monarch", "phenotype", "disease", "表型", "ontology"],
    )
    reg.register(
        query_quickgo,
        category="deferred",
        keywords=["go", "gene ontology", "quickgo", "功能", "function"],
    )
    # --- Phase B: pathway --------------------------------------------
    reg.register(
        query_kegg,
        category="deferred",
        keywords=["kegg", "pathway", "通路", "metabolic", "signaling"],
    )
    reg.register(
        query_reactome,
        category="deferred",
        keywords=["reactome", "pathway", "通路", "signaling"],
    )
    reg.register(
        query_stringdb,
        category="deferred",
        keywords=["string", "ppi", "interaction", "network", "相互作用"],
    )
    reg.register(
        query_wikipathways_graph,
        category="deferred",
        keywords=["wikipathways", "graph", "network", "pathway", "知识图谱", "网络", "graphrag"],
    )
    # --- Phase B: drug -----------------------------------------------
    reg.register(
        query_chembl_target_activities,
        category="deferred",
        keywords=[
            "chembl", "inhibitor", "ic50", "ki", "kd",
            "bioactivity", "drug", "化合物", "抑制剂", "活性",
        ],
    )
    reg.register(
        query_pubchem,
        category="deferred",
        keywords=["pubchem", "compound", "smiles", "cid", "化合物"],
    )
    reg.register(
        query_gtopdb,
        category="deferred",
        keywords=[
            "iuphar", "gtopdb", "guidetopharmacology", "ligand",
            "配体", "drug", "peptide ligand",
        ],
    )
    # --- Phase B: peptide --------------------------------------------
    reg.register(
        query_chembl_peptides,
        category="deferred",
        keywords=[
            "peptide", "多肽", "therapeutic peptide", "oligopeptide",
            "chembl peptide",
        ],
    )
    return reg


default_registry = _build_default_registry()
