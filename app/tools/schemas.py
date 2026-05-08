"""Pydantic schemas used for hard pruning (design doc §7.2.1 step 1).

Every external-API response is funneled through one of these models so
that the LLM only ever sees a curated, minimal field set — never the raw
JSON bloat returned by PubMed / UniProt / ChEMBL etc.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
    """A single literature record (PubMed / arXiv / Scholar)."""

    model_config = ConfigDict(extra="ignore")

    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None
    url: str | None = None

    def to_markdown(self) -> str:
        head = self.title.strip()
        meta_bits = []
        if self.year:
            meta_bits.append(str(self.year))
        if self.journal:
            meta_bits.append(self.journal)
        if self.pmid:
            meta_bits.append(f"PMID:{self.pmid}")
        if self.doi:
            meta_bits.append(f"DOI:{self.doi}")
        meta = " · ".join(meta_bits)
        body = (self.abstract or "").strip()
        link = self.url or ""
        return f"### {head}\n{meta}\n\n{body}\n{link}".strip()


class Subunit(BaseModel):
    """One polymeric subunit of a target complex."""

    model_config = ConfigDict(extra="ignore")

    accession: str
    name: str | None = None
    chain_id: str | None = None  # PDB chain when known
    copies: int = 1


class Isoform(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accession: str  # e.g. P00533-2
    name: str | None = None
    is_canonical: bool = False
    sequence_length: int | None = None


class Protein(BaseModel):
    """Curated UniProt projection — only the science-critical fields survive."""

    model_config = ConfigDict(extra="ignore")

    accession: str
    name: str | None = None
    gene: str | None = None
    organism: str | None = None
    sequence_length: int | None = None
    sequence: str | None = None  # full amino-acid string (Phase A2)
    function_summary: str | None = None
    keywords: list[str] = Field(default_factory=list)

    # Cross-references (Phase A2)
    pdb_ids: list[str] = Field(default_factory=list)
    alphafold_id: str | None = None
    interpro_domains: list[str] = Field(default_factory=list)
    subunits: list[Subunit] = Field(default_factory=list)
    isoforms: list[Isoform] = Field(default_factory=list)


class Activity(BaseModel):
    """A single bioassay measurement against a target."""

    model_config = ConfigDict(extra="ignore")

    target_id: str | None = None  # ChEMBL target id, UniProt acc, etc.
    type: str | None = None  # IC50 / Ki / Kd / EC50 ...
    value_nm: float | None = None
    units: str | None = None
    assay_description: str | None = None
    pchembl: float | None = None


class Molecule(BaseModel):
    """Curated ChEMBL/PubChem/GtoPdb projection."""

    model_config = ConfigDict(extra="ignore")

    molecule_chembl_id: str
    pref_name: str | None = None
    canonical_smiles: str | None = None
    max_phase: float | None = None
    standard_inchi_key: str | None = None

    # Phase A3: modality + peptide + activities + MoA
    modality: Literal["small_molecule", "peptide", "antibody", "other"] = "small_molecule"
    peptide_sequence: str | None = None
    mechanism_of_action: str | None = None
    activities: list[Activity] = Field(default_factory=list)


# --- Target / Pathway / Disease aggregates ----------------------------


class Pathway(BaseModel):
    """A signalling / metabolic pathway from KEGG, Reactome, etc."""

    model_config = ConfigDict(extra="ignore")

    source: Literal["KEGG", "Reactome", "WikiPathways", "Other"]
    external_id: str
    name: str
    description: str | None = None
    url: str | None = None
    interactors: list[str] = Field(default_factory=list)


class DiseaseAssociation(BaseModel):
    """A target-disease association (OpenTargets / Monarch / DisGeNET)."""

    model_config = ConfigDict(extra="ignore")

    source: Literal["OpenTargets", "Monarch", "DisGeNET", "Other"]
    disease_id: str  # EFO/MONDO/...
    disease_name: str
    score: float | None = None
    evidence_summary: str | None = None
    url: str | None = None


class Target(BaseModel):
    """A drug target — usually one gene/protein family with optional subunits."""

    model_config = ConfigDict(extra="ignore")

    name: str
    gene_symbol: str | None = None
    uniprot_ids: list[str] = Field(default_factory=list)
    organism: str = "Homo sapiens"
    description: str | None = None


class TargetReport(BaseModel):
    """Structured 5-section answer produced by the target-discovery sub-graph."""

    model_config = ConfigDict(extra="ignore")

    target: Target
    # Section 1: source literature
    papers: list[Paper] = Field(default_factory=list)
    # Section 2: protein composition
    proteins: list[Protein] = Field(default_factory=list)
    # Section 3: bio function / disease mechanism
    disease_associations: list[DiseaseAssociation] = Field(default_factory=list)
    function_narrative: str | None = None
    # Section 4: pathways
    pathways: list[Pathway] = Field(default_factory=list)
    # Section 5: effective drugs
    small_molecule_drugs: list[Molecule] = Field(default_factory=list)
    peptide_drugs: list[Molecule] = Field(default_factory=list)
    antibody_drugs: list[Molecule] = Field(default_factory=list)

    # Notes from each node (failure / partial coverage)
    notes: list[str] = Field(default_factory=list)
