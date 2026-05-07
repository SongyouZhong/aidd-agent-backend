"""Domain database tools (DEFERRED — hot-loaded on demand).

These are intentionally NOT included in the default System Prompt. They
become callable only after the model invokes ``tool_search`` and the
registry hot-mounts the matching schema (design doc §7.3.1).
"""

from __future__ import annotations

import logging
import re

from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool
from app.tools.schemas import Molecule, Protein

logger = logging.getLogger(__name__)

# UniProt accession format (covers both 6- and 10-character forms):
# [OPQ][0-9][A-Z0-9]{3}[0-9]  (old format, e.g. P00533)
# [A-NR-Z][0-9][A-Z0-9]{3}[0-9]  (new 6-char, e.g. Q13148)
# [A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z][A-Z0-9]{3}[0-9]  (new 10-char)
_UNIPROT_ACC_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]([A-Z][A-Z0-9]{3}[0-9])?)$"
)


# --- UniProt -----------------------------------------------------------

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


def _uniprot_to_protein(raw: dict) -> Protein:
    name = None
    proten = raw.get("proteinDescription") or {}
    rec = proten.get("recommendedName") or {}
    full = rec.get("fullName") or {}
    if isinstance(full, dict):
        name = full.get("value")

    organism = (raw.get("organism") or {}).get("scientificName")
    seq_block = raw.get("sequence") or {}
    seq_len = seq_block.get("length")
    seq_value = seq_block.get("value")  # full AA sequence (Phase A1 fix)

    gene = None
    for g in raw.get("genes", []) or []:
        gn = (g.get("geneName") or {}).get("value")
        if gn:
            gene = gn
            break

    function_summary = None
    for c in raw.get("comments", []) or []:
        if c.get("commentType") == "FUNCTION":
            texts = c.get("texts") or []
            if texts:
                function_summary = texts[0].get("value")
                break

    keywords = [k.get("name") for k in raw.get("keywords", []) or [] if k.get("name")]

    # Cross-references
    pdb_ids: list[str] = []
    interpro_domains: list[str] = []
    alphafold_id: str | None = None
    for x in raw.get("uniProtKBCrossReferences", []) or []:
        db = x.get("database")
        xid = x.get("id")
        if not xid:
            continue
        if db == "PDB":
            pdb_ids.append(xid)
        elif db == "InterPro":
            # Try to read the human-readable description from the properties.
            desc = None
            for p in x.get("properties", []) or []:
                if p.get("key") in ("EntryName", "DomainName"):
                    desc = p.get("value")
                    break
            interpro_domains.append(f"{xid} ({desc})" if desc else xid)
        elif db == "AlphaFoldDB":
            alphafold_id = xid

    accession = raw.get("primaryAccession") or raw.get("accession", "")
    if not alphafold_id and accession:
        # AlphaFold ID == UniProt accession by convention.
        alphafold_id = accession

    return Protein(
        accession=accession,
        name=name,
        gene=gene,
        organism=organism,
        sequence_length=seq_len,
        sequence=seq_value,
        function_summary=function_summary,
        keywords=keywords[:10],
        pdb_ids=pdb_ids[:25],
        alphafold_id=alphafold_id,
        interpro_domains=interpro_domains[:15],
    )


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_uniprot(query: str, max_results: int = 5) -> str:
    """Search the UniProt protein knowledgebase.

    Args:
        query: Either a UniProt accession (e.g. ``P00533``) or a free-form
               search expression (gene name, organism, etc.).
        max_results: Cap results when ``query`` is a search expression.

    Returns a compact markdown table per protein with sequence length,
    function summary and keywords (the heavy raw JSON is dropped).
    """
    q = query.strip()

    if _UNIPROT_ACC_RE.match(q):
        # Looks like a UniProt accession — direct entry lookup.
        url = f"{UNIPROT_BASE}/{q}"
        params = None
    else:
        url = f"{UNIPROT_BASE}/search"
        params = {
            "query": q,
            "format": "json",
            "size": max(1, min(int(max_results), 20)),
        }

    try:
        data = await query_rest_api(url, params=params)
    except Exception as exc:
        logger.exception("UniProt query failed")
        return f"UniProt query failed: {exc}"

    items = data.get("results") if isinstance(data, dict) and "results" in data else [data]
    proteins = [_uniprot_to_protein(it) for it in items if isinstance(it, dict)]

    if not proteins:
        return "No UniProt entries found."

    lines = ["| Accession | Gene | Organism | Length | Function |",
             "|---|---|---|---|---|"]
    for p in proteins:
        fn = (p.function_summary or "").replace("\n", " ").strip()
        if len(fn) > 200:
            fn = fn[:200] + "…"
        lines.append(f"| {p.accession} | {p.gene or '-'} | {p.organism or '-'} | "
                     f"{p.sequence_length or '-'} | {fn or '-'} |")
    
    result = "\n".join(lines)
    for p in proteins:
        if p.sequence:
            result += f"\n\n> Sequence for {p.accession}:\n{p.sequence}"
            
    return result


# --- ChEMBL ------------------------------------------------------------

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def _chembl_to_molecule(raw: dict) -> Molecule:
    structures = raw.get("molecule_structures") or {}
    mol_type = (raw.get("molecule_type") or "").lower()
    if "protein" in mol_type or "antibody" in mol_type:
        modality = "antibody"
    elif "peptide" in mol_type or "oligopeptide" in mol_type:
        modality = "peptide"
    elif "small molecule" in mol_type or mol_type == "":
        modality = "small_molecule"
    else:
        modality = "other"

    return Molecule(
        molecule_chembl_id=raw.get("molecule_chembl_id", ""),
        pref_name=raw.get("pref_name"),
        canonical_smiles=structures.get("canonical_smiles"),
        max_phase=raw.get("max_phase"),
        standard_inchi_key=structures.get("standard_inchi_key"),
        modality=modality,
    )


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_chembl(query: str, max_results: int = 10) -> str:
    """Query ChEMBL for compound bioactivity data.

    Args:
        query: A ChEMBL ID (e.g. ``CHEMBL25``) or a molecule name keyword.
        max_results: Cap when doing a name search.

    Only ``molecule_chembl_id``, ``pref_name``, ``canonical_smiles``,
    ``max_phase`` and ``standard_inchi_key`` survive the pruning step
    (TC-2.2.1).
    """
    q = query.strip()
    max_results = max(1, min(int(max_results), 25))

    try:
        if q.upper().startswith("CHEMBL"):
            data = await query_rest_api(f"{CHEMBL_BASE}/molecule/{q.upper()}.json")
            items = [data]
        else:
            data = await query_rest_api(
                f"{CHEMBL_BASE}/molecule/search.json",
                params={"q": q, "limit": max_results},
            )
            items = data.get("molecules", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.exception("ChEMBL query failed")
        return f"ChEMBL query failed: {exc}"

    mols = [_chembl_to_molecule(it) for it in items if isinstance(it, dict)]
    if not mols:
        return "No ChEMBL molecules found."

    lines = ["| ChEMBL ID | Name | Max Phase | SMILES |", "|---|---|---|---|"]
    for m in mols:
        smi = (m.canonical_smiles or "")
        if len(smi) > 80:
            smi = smi[:80] + "…"
        lines.append(
            f"| {m.molecule_chembl_id} | {m.pref_name or '-'} | "
            f"{m.max_phase if m.max_phase is not None else '-'} | {smi or '-'} |"
        )
    return "\n".join(lines)
