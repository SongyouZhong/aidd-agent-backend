"""Render a TargetReport dict (output of target_discovery_graph) to Markdown.

The rendered Markdown is the user-facing artifact: a human-readable
deep-research report with sections, GFM tables, fenced code blocks for
biological sequences (FASTA) and SMILES strings, and inline links to
source databases.

Only depends on the standard library — no external markdown libraries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Bilingual label lookup. Add new labels here to support additional
# sections without scattering string literals across the renderer.
_LABELS: dict[str, dict[str, str]] = {
    "English": {
        "report_title":         "Target Deep Analysis Report",
        "gene_symbol":          "Gene Symbol",
        "organism":             "Organism",
        "generated":            "Generated",
        "sec_function":         "## 1. Biological Function & Disease Mechanism",
        "sec_disease":          "## 2. Disease Associations",
        "disease_col":          "| Disease | Source | Score | Link |",
        "disease_link":         "View",
        "sec_pathway":          "## 3. Signaling Pathways",
        "pathway_col":          "| Pathway | Source | External ID | Key Interactors |",
        "sec_drugs":            "## 4. Targeted Drugs",
        "sec_drug_sm":          "### 4.1 Small Molecule Drugs",
        "drug_sm_col":          "| ChEMBL ID | Name | Clinical Phase | Activity (example) |",
        "sec_drug_pep":         "### 4.2 Peptide Drugs",
        "sec_drug_ab":          "### 4.3 Antibody Drugs",
        "sec_protein":          "## 5. Protein Structural Composition",
        "protein_gene":         "Gene",
        "protein_length":       "Sequence Length",
        "protein_pdb":          "Representative PDB",
        "protein_pdb_more":     "total: {n}",
        "protein_domains":      "InterPro Domains",
        "protein_seq":          "Amino Acid Sequence",
        "sec_literature":       "## 6. Key Literature",
        "sec_gaps":             "## 7. Data Source Gaps",
        "sec_notes":            "## 8. Pipeline Notes",
    },
    "Chinese": {
        "report_title":         "靶点深度分析报告",
        "gene_symbol":          "基因符号",
        "organism":             "物种",
        "generated":            "生成时间",
        "sec_function":         "## 一、生物学功能与疾病机制",
        "sec_disease":          "## 二、疾病关联",
        "disease_col":          "| 疾病 | 来源 | 关联评分 | 链接 |",
        "disease_link":         "查看",
        "sec_pathway":          "## 三、信号通路",
        "pathway_col":          "| 通路名称 | 来源 | 外部 ID | 关键互作 |",
        "sec_drugs":            "## 四、靶向药物",
        "sec_drug_sm":          "### 4.1 小分子药物",
        "drug_sm_col":          "| ChEMBL ID | 名称 | 临床阶段 | 活性 (示例) |",
        "sec_drug_pep":         "### 4.2 多肽药物",
        "sec_drug_ab":          "### 4.3 抗体药物",
        "sec_protein":          "## 五、蛋白质结构组成",
        "protein_gene":         "基因",
        "protein_length":       "序列长度",
        "protein_pdb":          "代表 PDB 结构",
        "protein_pdb_more":     "共 {n} 个",
        "protein_domains":      "InterPro 结构域",
        "protein_seq":          "氨基酸序列",
        "sec_literature":       "## 六、关键文献",
        "sec_gaps":             "## 七、数据源缺口",
        "sec_notes":            "## 八、流水线备注",
    },
}


def _as_dict(item: Any) -> dict[str, Any]:
    """Return item if it is a dict, otherwise an empty dict.

    Guards every list-of-dicts loop in the renderer against LLM outputs that
    accidentally emit strings or other scalar types instead of objects.
    """
    return item if isinstance(item, dict) else {}


def render_target_report_md(report: dict[str, Any], target_query: str, language: str = "English") -> str:
    """Render a deep-research TargetReport dict into Markdown.

    Sections:
      1. Header (target meta + generation timestamp)
      2. Function narrative
      3. Disease associations (table)
      4. Pathways (table)
      5. Drugs (small molecules / peptides / antibodies — separate tables)
      6. Protein composition (cards + sequence as fasta code block)
      7. Literature (numbered list with links)
      8. Notes / data-source gaps
    """
    out: list[str] = []
    lbl = _LABELS.get(language, _LABELS["English"])
    target_raw = report.get("target")
    target: dict[str, Any] = target_raw if isinstance(target_raw, dict) else {}
    name = target.get("name") or (target_raw if isinstance(target_raw, str) else None) or target_query
    gene = target.get("gene_symbol") or "—"
    uniprot_ids = target.get("uniprot_ids") or []
    organism = target.get("organism") or "Homo sapiens"
    description = (target.get("description") or "").strip()

    # 1) Header --------------------------------------------------------
    out.append(f"# 🧬 {name} {lbl['report_title']}")
    out.append("")
    out.append(
        f"> **{lbl['gene_symbol']}**: `{gene}` &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**{lbl['organism']}**: {organism} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**UniProt**: {', '.join(f'`{u}`' for u in uniprot_ids) or '—'}"
    )
    out.append("")
    out.append(f"*{lbl['generated']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    out.append("")
    if description:
        out.append(description)
        out.append("")

    # 2) Function narrative -------------------------------------------
    fn = (report.get("function_narrative") or "").strip()
    if fn:
        out.append(lbl["sec_function"])
        out.append("")
        out.append(fn)
        out.append("")

    # 3) Disease associations -----------------------------------------
    diseases = report.get("disease_associations") or []
    if diseases:
        out.append(lbl["sec_disease"])
        out.append("")
        out.append(lbl["disease_col"])
        out.append("|---|---|---:|---|")
        for _d in diseases:
            d = _as_dict(_d)
            dname = (d.get("disease_name") or "—").replace("|", "\\|")
            src = d.get("source") or "—"
            score = d.get("score")
            score_s = f"{float(score):.3f}" if isinstance(score, (int, float)) else "—"
            url = d.get("url") or ""
            link = f"[{lbl['disease_link']}]({url})" if url else "—"
            out.append(f"| {dname} | {src} | {score_s} | {link} |")
        out.append("")

    # 4) Pathways ------------------------------------------------------
    pathways = report.get("pathways") or []
    if pathways:
        out.append(lbl["sec_pathway"])
        out.append("")
        out.append(lbl["pathway_col"])
        out.append("|---|---|---|---|")
        for _p in pathways:
            p = _as_dict(_p)
            pname = (p.get("name") or "—").replace("|", "\\|")
            src = p.get("source") or "—"
            ext_id = p.get("external_id") or "—"
            url = p.get("url")
            if url and ext_id != "—":
                ext_id = f"[{ext_id}]({url})"
            interactors = p.get("interactors") or []
            inter_s = ", ".join(f"`{i}`" for i in interactors[:8]) or "—"
            out.append(f"| {pname} | {src} | {ext_id} | {inter_s} |")
        out.append("")

    # 5) Drugs ---------------------------------------------------------
    sm = report.get("small_molecule_drugs") or []
    pep = report.get("peptide_drugs") or []
    ab = report.get("antibody_drugs") or []
    if sm or pep or ab:
        out.append(lbl["sec_drugs"])
        out.append("")

    if sm:
        out.append(lbl["sec_drug_sm"])
        out.append("")
        out.append(lbl["drug_sm_col"])
        out.append("|---|---|---|---|")
        for _d in sm:
            d = _as_dict(_d)
            cid = d.get("molecule_chembl_id") or "—"
            pname = d.get("pref_name") or "—"
            phase = d.get("max_phase")
            phase_s = str(phase) if phase is not None else "—"
            acts = d.get("activities") or []
            if acts:
                a = _as_dict(acts[0])
                act_s = f"{a.get('type', '?')} = {a.get('value_nM', a.get('value_nm', '?'))} nM"
            else:
                act_s = "—"
            out.append(f"| `{cid}` | {pname} | {phase_s} | {act_s} |")
        out.append("")
        # SMILES code block per drug (renderable as language="smiles")
        for _d in sm:
            d = _as_dict(_d)
            smi = d.get("canonical_smiles")
            if smi:
                cid = d.get("molecule_chembl_id") or ""
                pname = d.get("pref_name") or cid
                out.append(f"**{pname}** SMILES:")
                out.append("")
                out.append("```smiles")
                out.append(str(smi))
                out.append("```")
                out.append("")
        # If everything was a stub note, surface it
        if all(not _as_dict(d).get("canonical_smiles") and _as_dict(d).get("note") for d in sm):
            for _d in sm:
                d = _as_dict(_d)
                if d.get("note"):
                    out.append(f"> ℹ {d['note']}")
                    out.append("")

    if pep:
        out.append(lbl["sec_drug_pep"])
        out.append("")
        for _d in pep:
            d = _as_dict(_d)
            note = d.get("note")
            if note:
                out.append(f"> ℹ {note}")
                out.append("")
                continue
            cid = d.get("molecule_chembl_id") or "—"
            pname = d.get("pref_name") or cid
            seq = d.get("peptide_sequence")
            out.append(f"- **{pname}** (`{cid}`)")
            if seq:
                out.append("")
                out.append("```sequence")
                out.append(str(seq))
                out.append("```")
                out.append("")

    if ab:
        out.append(lbl["sec_drug_ab"])
        out.append("")
        for _d in ab:
            d = _as_dict(_d)
            note = d.get("note")
            if note:
                out.append(f"> ℹ {note}")
                out.append("")
                continue
            cid = d.get("molecule_chembl_id") or "—"
            pname = d.get("pref_name") or cid
            seq = d.get("peptide_sequence")
            out.append(f"- **{pname}** (`{cid}`)")
            if seq:
                out.append("")
                out.append("```sequence")
                out.append(str(seq))
                out.append("```")
                out.append("")

    # 6) Protein composition ------------------------------------------
    proteins = report.get("proteins") or []
    if proteins:
        out.append(lbl["sec_protein"])
        out.append("")
        for _p in proteins:
            p = _as_dict(_p)
            acc = p.get("accession") or "—"
            pname = p.get("name") or acc
            pgene = p.get("gene") or "—"
            length = p.get("sequence_length") or "—"
            af = p.get("alphafold_id")
            pdbs = p.get("pdb_ids") or []
            domains = p.get("interpro_domains") or []
            out.append(f"### {pname} (`{acc}`)")
            out.append("")
            out.append(f"- **{lbl['protein_gene']}**: `{pgene}`")
            out.append(f"- **{lbl['protein_length']}**: {length} aa")
            if af:
                out.append(f"- **AlphaFold**: [{af}](https://alphafold.ebi.ac.uk/entry/{af})")
            if pdbs:
                pdb_links = ", ".join(
                    f"[{pid}](https://www.rcsb.org/structure/{pid})" for pid in pdbs[:10]
                )
                _more_tpl = lbl["protein_pdb_more"]
                more = f" *({_more_tpl.format(n=len(pdbs))})*" if len(pdbs) > 10 else ""
                out.append(f"- **{lbl['protein_pdb']}**: {pdb_links}{more}")
            if domains:
                out.append(f"- **{lbl['protein_domains']}**:")
                for _dom in domains:
                    dom = _as_dict(_dom)
                    did = dom.get("interpro_id") or "—"
                    dname = dom.get("name") or "—"
                    dtype = dom.get("type") or ""
                    type_s = f" *({dtype})*" if dtype else ""
                    out.append(f"  - [`{did}`](https://www.ebi.ac.uk/interpro/entry/InterPro/{did}/) {dname}{type_s}")
            seq = p.get("sequence")
            if seq:
                out.append("")
                out.append(f"**{lbl['protein_seq']}**:")
                out.append("")
                out.append("```fasta")
                out.append(f">{acc}|{pgene}")
                # Wrap to 60 chars per FASTA convention
                for i in range(0, len(seq), 60):
                    out.append(seq[i : i + 60])
                out.append("```")
            out.append("")

    # 7) Literature ---------------------------------------------------
    papers = report.get("papers") or []
    if papers:
        out.append(lbl["sec_literature"])
        out.append("")
        for i, _p in enumerate(papers, 1):
            p = _as_dict(_p)
            title = (p.get("title") or "—").strip()
            year = p.get("year")
            year_s = f" ({year})" if year else ""
            url = p.get("url") or ""
            doi = p.get("doi")
            pmid = p.get("pmid")
            ids: list[str] = []
            if pmid:
                ids.append(f"[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
            if doi:
                ids.append(f"[DOI:{doi}](https://doi.org/{doi})")
            id_s = " &nbsp;|&nbsp; ".join(ids)
            head = f"**{i}. [{title}]({url})**" if url else f"**{i}. {title}**"
            out.append(f"{head}{year_s}")
            if id_s:
                out.append("")
                out.append(id_s)
            summary = (p.get("summary") or "").strip()
            if summary:
                out.append("")
                out.append(f"> {summary}")
            out.append("")

    # 8) Notes / gaps -------------------------------------------------
    gaps = report.get("data_source_gaps") or []
    if gaps:
        out.append(lbl["sec_gaps"])
        out.append("")
        for _g in gaps:
            g = _as_dict(_g)
            cat = g.get("category") or "—"
            reason = g.get("reason") or "—"
            out.append(f"- **{cat}**: {reason}")
        out.append("")

    notes = report.get("notes") or []
    if notes:
        out.append(lbl["sec_notes"])
        out.append("")
        for n in notes:
            out.append(f"- {n}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"
