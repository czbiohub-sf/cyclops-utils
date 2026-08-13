"""Gene super-category resolver.

Maps gene names to high-level cell biology categories using three approaches:

- **chad**: CHAD v5 hierarchical clusters only (~19% coverage, 8 categories)
- **chad_boosted**: CHAD + Reactome/GO keyword matching + regex + Harmonizome
  overrides (~98% coverage, 8 categories)
- **reactome_toplevel**: Reactome top-level pathways (78% coverage, 29 categories,
  multi-mapping — genes can belong to multiple categories)
- **reactome_cell_biology**: reactome_toplevel filtered to 17 cell-biology-relevant
  categories (excludes tissue/organism-specific and catch-all disease categories;
  see configs/reactome_cell_biology_filter.md for full rationale)

Used by reporter_radar_stage and potentially other attribution analyses.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

import yaml
from cyclops_utils.paths import BASE_PATH

logger = logging.getLogger(__name__)

# Available annotation sources
ALL_SOURCES = frozenset({"chad", "chad_boosted", "reactome_toplevel", "reactome_cell_biology"})

# Categories included in reactome_cell_biology (see configs/reactome_cell_biology_filter.md)
REACTOME_CELL_BIOLOGY_CATEGORIES = frozenset({
    "Autophagy",
    "Cell Cycle",
    "Cell-Cell communication",
    "Cellular responses to stimuli",
    "Chromatin organization",
    "DNA Repair",
    "DNA Replication",
    "Gene expression (Transcription)",
    "Metabolism",
    "Metabolism of proteins",
    "Metabolism of RNA",
    "Organelle biogenesis and maintenance",
    "Programmed Cell Death",
    "Protein localization",
    "Signal Transduction",
    "Transport of small molecules",
    "Vesicle-mediated transport",
})

DEFAULT_CHAD_PATH = Path(
    f"{BASE_PATH}/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml"
)
DEFAULT_GENE_PANEL_PATH = Path(
    f"{BASE_PATH}/configs/annotated_gene_panel_July2025.csv"
)
DEFAULT_REACTOME_CATEGORIES_PATH = Path(
    f"{BASE_PATH}/configs/ontologies/reactome/panel_gene_reactome_categories.tsv"
)


def parse_sources(sources_str: str) -> FrozenSet[str]:
    """Parse a comma-separated sources string into a validated frozenset."""
    parsed = frozenset(s.strip().lower() for s in sources_str.split(",") if s.strip())
    unknown = parsed - ALL_SOURCES
    if unknown:
        raise ValueError(
            f"Unknown source(s): {unknown}. Valid: {sorted(ALL_SOURCES)}"
        )
    if not parsed:
        raise ValueError(f"At least one source required. Valid: {sorted(ALL_SOURCES)}")
    return parsed


def sources_label(sources: FrozenSet[str]) -> str:
    """Return a filesystem-safe label for a set of sources, e.g. 'chad_boosted'."""
    return "_".join(sorted(sources))


def is_reactome_toplevel_mode(sources: FrozenSet[str]) -> bool:
    """Check if sources indicate any Reactome multi-mapping mode."""
    return bool(sources & {"reactome_toplevel", "reactome_cell_biology"})


# ---------------------------------------------------------------------------
# Harmonizome overrides (gap-filler for poorly annotated genes)
# ---------------------------------------------------------------------------

_HARMONIZOME_OVERRIDES: Dict[str, str] = {
    # -----------------------------------------------------------------------
    # Translation — cytoplasmic ribosome biogenesis, tRNA modification,
    # rRNA processing, translation initiation factors
    # -----------------------------------------------------------------------
    "ADAT3":   "Translation",      # tRNA adenosine deaminase; modifies anticodon wobble position (A34→I) to expand decoding
    "NOL10":   "Translation",      # Nucleolar protein required for 18S rRNA processing and 40S ribosome biogenesis
    "NOP9":    "Translation",      # PUM-HD RNA-binding protein; processes 18S pre-rRNA during 40S subunit assembly
    "TSR2":    "Translation",      # Chaperone that escorts RPS26 to pre-40S ribosome during small subunit biogenesis
    "KRI1":    "Translation",      # Required for 18S rRNA processing and 40S ribosomal subunit biogenesis
    "C12orf45": "Translation",     # Associates with ribosome biogenesis machinery; implicated in pre-rRNA processing for 40S maturation
    "ZCCHC9":  "Translation",      # Zinc-knuckle protein; associates with pre-ribosomal complex during 18S rRNA maturation
    "PRKRA":   "Translation",      # PACT: activates PKR (EIF2AK2) which phosphorylates eIF2α to globally suppress translation initiation
    "YAE1":    "Translation",      # CIA-pathway component required for tRNA modification (Elongator complex maturation via iron-sulfur cluster assembly)

    # -----------------------------------------------------------------------
    # Gene Expression — transcription, splicing, mRNA processing, chromatin,
    # RNA export, mRNA decay, nuclear transport of Pol II
    # -----------------------------------------------------------------------
    "ANP32B":  "Gene Expression",  # Acidic nuclear phosphoprotein; associates with chromatin and regulates histone acetylation/RNA Pol II transcription
    "ASF1B":   "Gene Expression",  # Histone H3.1/H3.3 chaperone; deposits histones during replication-coupled chromatin assembly
    "CCDC174": "Gene Expression",  # Associates with exon junction complex; required for pre-mRNA splicing and mRNA export
    "RBM26":   "Gene Expression",  # RNA-binding protein; involved in mRNA 3′-end processing and nonsense-mediated decay
    "SCAF4":   "Gene Expression",  # SR-related CTD-associated factor; links RNA Pol II CTD to pre-mRNA 3′-end cleavage/polyadenylation
    "PATL1":   "Gene Expression",  # P-body decapping activator; functions in mRNA decay and translational repression in processing bodies
    "ZNRD1":   "Gene Expression",  # RBQ-1 subunit of RNA Pol I/II accessory factor; participates in transcriptional regulation
    "BTF3L4":  "Gene Expression",  # Nascent polypeptide-associated complex beta subunit-like; transcription factor interacting with RNA Pol II
    "CCDC59":  "Gene Expression",  # Associates with spliceosome and pre-mRNA splicing factors; implicated in RNA processing
    "PRRC2A":  "Gene Expression",  # BAT2; large RNA-binding protein functioning in pre-mRNA splicing and m6A-mediated mRNA stabilization
    "MSANTD4": "Gene Expression",  # Myb/SANT domain protein; transcriptional regulation via chromatin-associated DNA binding
    "TIPARP":  "Gene Expression",  # PARP7; mono-ART transcriptional co-repressor of AHR; modifies RNA Pol II — not a DNA repair enzyme
    "GPN2":    "Gene Expression",  # GPN-loop GTPase required for nuclear import of RNA Polymerase II; GTPase but function is Pol II biogenesis
    "CSE1L":   "Gene Expression",  # Exportin for importin-α; recycles nuclear import machinery — primary role is nucleocytoplasmic transport

    # -----------------------------------------------------------------------
    # Cell Cycle & DNA — cell cycle checkpoints, DNA replication,
    # DNA repair, mitosis, chromosome stability
    # -----------------------------------------------------------------------
    "TTI1":    "Cell Cycle & DNA", # TTT complex (TELO2-TTI1-TTI2) stabilizes ATM/ATR/DNA-PKcs; required for DNA damage checkpoint signaling
    "TTI2":    "Cell Cycle & DNA", # TTT complex subunit alongside TTI1; obligate partner for ATM/ATR/DNA-PKcs stabilization
    "TSPAN31": "Cell Cycle & DNA", # Amplified at 12q14 with CDK4; co-amplified with cell cycle oncogenes and modulates cell proliferation
    "DNASE1L1":"Cell Cycle & DNA", # DNase I-family endonuclease; cleaves chromatin DNA during apoptosis downstream of caspase activation

    # -----------------------------------------------------------------------
    # Signaling — kinase cascades, receptor signaling, ion channels,
    # GTPases in signal transduction
    # -----------------------------------------------------------------------
    "PTPRH":   "Signaling",        # Receptor-type PTP-H; dephosphorylates RTK substrates and modulates EGF/integrin signaling
    "CD5":     "Signaling",        # Type I transmembrane receptor on T/B1 cells; modulates TCR/BCR signaling thresholds via SHP-1/SHP-2
    "GABRR3":  "Signaling",        # GABA-A rho-3 receptor; ligand-gated Cl⁻ channel mediating inhibitory neurotransmitter signaling
    "KCNG1":   "Signaling",        # Kv6.1 voltage-gated K⁺ channel modifier subunit; modulates membrane excitability via Kv2 heteromers
    "KCNG4":   "Signaling",        # Kv6.4 voltage-gated K⁺ channel modifier; modulates Kv2.1 activity and neuronal excitability signaling
    "TM2D2":   "Signaling",        # Ortholog of Drosophila Almondex; implicated in Notch and beta-amyloid precursor signaling pathways
    "DIPK1B":  "Signaling",        # FAM69B; divergent ER-localized active kinase; characterized as ER signaling kinase, not proteostasis factor
    "PARP12":  "Signaling",        # ARTD12; mono-ADP-ribosyltransferase in antiviral/stress signaling and stress granule assembly — not DNA repair

    # -----------------------------------------------------------------------
    # Membrane Trafficking — vesicle transport, Arf GTPases, lysosome
    # positioning, ER protein insertion, endosomal recycling
    # -----------------------------------------------------------------------
    "BORCS7":  "Membrane Trafficking", # BORC subunit; tethers lysosomes to kinesin for anterograde lysosome/late endosome positioning
    "BORCS8":  "Membrane Trafficking", # BORC subunit (MEF2BNB); required for lysosomal trafficking along microtubules
    "WRB":     "Membrane Trafficking", # GET/TRC pathway receptor (WRB-CAML complex); inserts tail-anchored proteins into ER membrane
    "RGPD8":   "Membrane Trafficking", # GRIP domain targets to trans-Golgi network; RANBP2-related protein involved in vesicular trafficking
    "NBEAL1":  "Membrane Trafficking", # BEACH domain protein; late endosome/lysosome membrane dynamics, related to LYST trafficking regulators
    "AGAP4":   "Membrane Trafficking", # Arf GAP with GLD domain; regulates Arf1/Arf6 GTPase cycling in vesicle coat assembly — not kinase signaling
    "AGAP9":   "Membrane Trafficking", # Arf GAP family; primary role is Arf GTPase cycling during vesicle budding, not receptor signaling

    # -----------------------------------------------------------------------
    # Metabolism — mitochondria, ETC, biosynthetic enzymes, transporters,
    # small-molecule metabolism, mitochondrial translation factors
    # -----------------------------------------------------------------------
    "SLC38A2": "Metabolism",       # SNAT2; primary plasma membrane glutamine/neutral amino acid transporter driving mTORC1 sensing
    "SLC38A6": "Metabolism",       # SNAT6; vesicular glutamine transporter supporting neurotransmitter and amino acid metabolism
    "SLC5A3":  "Metabolism",       # SMIT1; sodium/myo-inositol cotransporter for phosphatidylinositol biosynthesis and osmoregulation
    "CTPS2":   "Metabolism",       # CTP synthase 2; catalyzes de novo CTP synthesis from UTP in pyrimidine nucleotide metabolism
    "PPCS":    "Metabolism",       # Phosphopantothenoylcysteine synthetase; second step of coenzyme A biosynthesis from pantothenate
    "MICOS13": "Metabolism",       # QIL1; MICOS complex subunit maintaining mitochondrial inner membrane cristae for oxidative phosphorylation
    "CA7":     "Metabolism",       # Cytosolic carbonic anhydrase VII; CO₂⇌HCO₃⁻ interconversion in pH regulation and metabolic buffering
    "CA13":    "Metabolism",       # Cytosolic carbonic anhydrase XIII; CO₂ hydration in metabolic and acid-base regulatory pathways
    "MGAM":    "Metabolism",       # Maltase-glucoamylase; intestinal brush-border enzyme cleaving terminal glucose in carbohydrate catabolism
    "ABHD16A": "Metabolism",       # α/β-hydrolase phosphatidylserine lipase; generates lysophosphatidylserine bioactive lipid metabolite
    "COMTD1":  "Metabolism",       # COMT-domain methyltransferase; predicted catecholamine/small-molecule methylation metabolic enzyme
    "METTL7A": "Metabolism",       # ER/lipid-droplet thiol methyltransferase; methylates H₂S and small thiols as metabolic detoxification
    "METTL7B": "Metabolism",       # Paralog of METTL7A; shared thiol methyltransferase activity on metabolic substrates at ER membrane
    "UGT3A2":  "Metabolism",       # UDP glycosyltransferase 3A2; transfers sugar moieties to steroids/bile acids in phase II metabolism
    "C6orf203": "Metabolism",      # MTRES1; mitochondrial RNA-binding protein processing mt-rRNAs for mitochondrial ribosome assembly — mitochondrial biology, not cytoplasmic translation

    # -----------------------------------------------------------------------
    # Protein Homeostasis — ER quality control, ubiquitin-proteasome,
    # chaperones, glycosylation, unfolded protein response
    # -----------------------------------------------------------------------
    "EMC1":    "Protein Homeostasis", # Largest EMC subunit; co-translational insertase for tail-anchored and multi-pass TM proteins into ER
    "EMC2":    "Protein Homeostasis", # TPR-repeat EMC subunit (TTC35); required for ER protein quality control and TM domain insertion
    "EMC3":    "Protein Homeostasis", # Catalytic insertase domain of EMC; directly mediates hydrophobic TM segment insertion into lipid bilayer
    "TM7SF3":  "Protein Homeostasis", # 7-TM ER protein; suppresses ER stress-induced UPR activation and protects against UPR-mediated apoptosis
    "NUDCD1":  "Protein Homeostasis", # NudC chaperone family; Hsp90 co-chaperone stabilizing cytoplasmic dynein and other client proteins
    "COMMD4":  "Protein Homeostasis", # COMM domain protein in CCC complex; regulates ubiquitin-dependent NF-κB degradation via proteasome pathway
    "URM1":    "Protein Homeostasis", # Ubiquitin-related modifier 1; mammalian urmylation (ubiquitin-like conjugation) pathway — predominant role is protein modification not tRNA thiolation

    # -----------------------------------------------------------------------
    # Cytoskeleton & Morphology — actin, focal adhesion, keratins,
    # microtubule organization, cell shape, cilia
    # -----------------------------------------------------------------------
    "RAI14":   "Cytoskeleton & Morphology", # Actin-binding protein (NORPEG/ankycorbin); organizes F-actin at adherens junctions and regulates epithelial morphology
    "KIAA1211":"Cytoskeleton & Morphology", # Centrosome/basal body protein required for ciliogenesis and cytoskeletal organization during morphogenesis
    "NHSL1":   "Cytoskeleton & Morphology", # NHS-like 1; activates WAVE regulatory complex to stimulate Arp2/3-dependent actin polymerization and lamellipodia
    "MYBPHL":  "Cytoskeleton & Morphology", # Myosin-binding protein H-like; modulates actomyosin filament organization and sarcomere thick filament architecture
    "KRTAP2-4":"Cytoskeleton & Morphology", # Keratin-associated protein; cross-links keratin intermediate filaments in hair shaft cortex
    "KRTAP4-2":"Cytoskeleton & Morphology", # High-sulfur KRTAP; interacts with type II keratin filaments defining hair fiber mechanical properties
    "SPRR2B":  "Cytoskeleton & Morphology", # Small proline-rich protein; cornified envelope component cross-linked to keratin during epidermal terminal differentiation
    "LCE3A":   "Cytoskeleton & Morphology", # Late cornified envelope protein; structural component providing mechanical integrity to keratinized epidermis
    "CCDC74B": "Cytoskeleton & Morphology", # CCDC74B; localizes to cilia/flagella axonemes; required for axonemal microtubule organization and ciliary motility
    "TMEM8A":  "Cytoskeleton & Morphology", # Transmembrane protein regulating actin cytoskeleton organization, cell morphology and migration — not a metabolic enzyme
}


# ---------------------------------------------------------------------------
# CHAD hierarchy helpers
# ---------------------------------------------------------------------------

def _load_chad_hierarchy(chad_path: Path) -> dict:
    """Load the CHAD v5 hierarchy YAML."""
    with open(chad_path) as f:
        return yaml.safe_load(f) or {}


def _build_name_index(chad: dict) -> Dict[str, dict]:
    """Build cluster name -> cluster dict index from CHAD hierarchy."""
    idx: Dict[str, dict] = {}
    for _id, cluster in chad.items():
        if isinstance(cluster, dict) and "name" in cluster:
            idx[cluster["name"]] = cluster
    return idx


def _collect_genes_recursive(
    cluster_name: str,
    name_index: Dict[str, dict],
    visited: Optional[Set[str]] = None,
) -> Set[str]:
    """Recursively collect all genes under a named cluster via components."""
    if visited is None:
        visited = set()
    if cluster_name in visited:
        return set()
    visited.add(cluster_name)

    cluster = name_index.get(cluster_name)
    if cluster is None:
        return set()

    genes = set(cluster.get("genes", []))
    for component in cluster.get("components", []):
        genes |= _collect_genes_recursive(component, name_index, visited)
    return genes


def _load_gene_annotations(panel_path: Path) -> Dict[str, str]:
    """Load Reactome + GO annotations from the gene panel CSV.

    Returns dict: gene_name -> concatenated lowercase annotation string.
    """
    annotations: Dict[str, str] = {}
    if not panel_path.exists():
        logger.warning(f"Gene panel not found: {panel_path}")
        return annotations

    with open(panel_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene = row.get("Gene.name", "").strip()
            if not gene:
                continue
            react = row.get("In_REACT_pathways", "") or ""
            go = row.get("In_go_pathways", "") or ""
            annotations[gene] = (react + " " + go).lower()

    return annotations


# ---------------------------------------------------------------------------
# Reactome top-level loader
# ---------------------------------------------------------------------------

def _load_reactome_toplevel(
    reactome_path: Path,
) -> Dict[str, List[str]]:
    """Load Reactome top-level pathway assignments from pre-computed TSV.

    The TSV has columns: gene_name, ncbi_id, reactome_categories
    where reactome_categories is pipe-separated.

    Returns dict: gene_name -> list of Reactome top-level pathway names.
    """
    gene_to_cats: Dict[str, List[str]] = {}
    if not reactome_path.exists():
        logger.warning(f"Reactome categories file not found: {reactome_path}")
        return gene_to_cats

    with open(reactome_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("gene_name", "").strip()
            cats_str = row.get("reactome_categories", "").strip()
            if not gene:
                continue
            if cats_str:
                gene_to_cats[gene] = [c.strip() for c in cats_str.split("|") if c.strip()]

    return gene_to_cats


# ---------------------------------------------------------------------------
# Builder: chad / chad_boosted (8 curated categories, single-mapping)
# ---------------------------------------------------------------------------

def build_gene_supercategory_map(
    mapping_config: dict,
    chad_path: Optional[Path] = None,
    gene_panel_path: Optional[Path] = None,
    boosted: bool = False,
) -> Dict[str, str]:
    """Build gene -> super-category mapping using CHAD with optional boosting.

    Parameters
    ----------
    mapping_config : dict
        Parsed gene_supercategory_mapping.yaml with ``super_categories`` key.
    chad_path : Path, optional
        Override for the CHAD hierarchy YAML path.
    gene_panel_path : Path, optional
        Override for the annotated gene panel CSV path.
    boosted : bool
        If True, augment CHAD with Reactome/GO keyword matching, regex patterns,
        and Harmonizome overrides for ~98% coverage. If False, CHAD clusters only
        (~19% coverage).

    Returns
    -------
    dict
        gene_name -> category_name mapping.
    """
    chad_path = chad_path or Path(
        mapping_config.get("chad_hierarchy_path", str(DEFAULT_CHAD_PATH))
    )
    gene_panel_path = gene_panel_path or Path(
        mapping_config.get("gene_panel_path", str(DEFAULT_GENE_PANEL_PATH))
    )

    chad = _load_chad_hierarchy(chad_path)
    name_index = _build_name_index(chad)
    gene_annotations = _load_gene_annotations(gene_panel_path)

    gene_to_cat: Dict[str, str] = {}
    categories = mapping_config.get("super_categories", {})

    # Collect all known genes
    all_genes: Set[str] = set()
    for _id, cluster in chad.items():
        if isinstance(cluster, dict):
            all_genes.update(cluster.get("genes", []))
    all_genes.update(gene_annotations.keys())

    # CHAD cluster-based assignment (always applied)
    for cat_name, cat_def in categories.items():
        for cluster_name in cat_def.get("chad_clusters", []):
            for gene in _collect_genes_recursive(cluster_name, name_index):
                if gene not in gene_to_cat:
                    gene_to_cat[gene] = cat_name
    chad_assigned = len(gene_to_cat)

    if not boosted:
        _log_summary("chad", gene_to_cat, all_genes, chad=chad_assigned)
        return gene_to_cat

    # Boost: Reactome + GO pathway keyword matching
    before = len(gene_to_cat)
    for gene, ann in gene_annotations.items():
        if gene in gene_to_cat or not ann.strip():
            continue
        best_cat = None
        best_score = 0
        for cat_name, cat_def in categories.items():
            keywords = cat_def.get("pathway_keywords", [])
            score = sum(1 for kw in keywords if kw.lower() in ann)
            if score > best_score:
                best_score = score
                best_cat = cat_name
        if best_cat and best_score >= 1:
            gene_to_cat[gene] = best_cat
    keyword_assigned = len(gene_to_cat) - before

    # Boost: Regex pattern matching on gene names
    before = len(gene_to_cat)
    compiled: Dict[str, list] = {}
    for cat_name, cat_def in categories.items():
        compiled[cat_name] = [
            re.compile(p) for p in cat_def.get("gene_patterns", [])
        ]
    for gene in all_genes:
        if gene in gene_to_cat:
            continue
        for cat_name, rxs in compiled.items():
            if any(rx.search(gene) for rx in rxs):
                gene_to_cat[gene] = cat_name
                break
    regex_assigned = len(gene_to_cat) - before

    # Boost: Harmonizome-derived overrides
    before = len(gene_to_cat)
    for gene in all_genes:
        if gene not in gene_to_cat and gene in _HARMONIZOME_OVERRIDES:
            gene_to_cat[gene] = _HARMONIZOME_OVERRIDES[gene]
    harmonizome_assigned = len(gene_to_cat) - before

    _log_summary(
        "chad_boosted", gene_to_cat, all_genes,
        chad=chad_assigned, keywords=keyword_assigned,
        regex=regex_assigned, harmonizome=harmonizome_assigned,
    )
    return gene_to_cat


def _log_summary(
    label: str,
    gene_to_cat: Dict[str, str],
    all_genes: Set[str],
    **counts: int,
) -> None:
    """Log gene category assignment summary."""
    cat_counts: Dict[str, int] = {}
    for cat in gene_to_cat.values():
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    unassigned = len(all_genes) - len(gene_to_cat)

    parts = [f"{k}={v}" for k, v in counts.items() if v is not None]
    logger.info(
        f"Gene super-category mapping [{label}]: "
        f"{len(gene_to_cat)}/{len(all_genes)} genes assigned "
        f"({', '.join(parts)})"
    )
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat:<30} {n:>4} genes")
    if unassigned > 0:
        logger.info(f"  {'Other (unassigned)':<30} {unassigned:>4} genes")


# ---------------------------------------------------------------------------
# Builder: reactome_toplevel (29 categories, multi-mapping)
# ---------------------------------------------------------------------------

def build_reactome_toplevel_map(
    reactome_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Build gene -> [categories] mapping using Reactome top-level pathways.

    Uses Reactome's own 29 top-level biological process categories.
    Genes can belong to multiple categories (multi-mapping).

    Parameters
    ----------
    reactome_path : Path, optional
        Path to the pre-computed panel_gene_reactome_categories.tsv.

    Returns
    -------
    dict
        gene_name -> list of category names. Unmapped genes are not included.
    """
    reactome_path = reactome_path or DEFAULT_REACTOME_CATEGORIES_PATH
    gene_to_cats = _load_reactome_toplevel(reactome_path)

    n_genes = len(gene_to_cats)
    cat_counts: Dict[str, int] = {}
    for cats in gene_to_cats.values():
        for cat in cats:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    logger.info(
        f"Reactome top-level mapping: {n_genes} genes assigned "
        f"across {len(cat_counts)} categories "
        f"(avg {sum(len(c) for c in gene_to_cats.values()) / max(n_genes, 1):.1f} "
        f"categories/gene)"
    )
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat:<40} {n:>4} genes")

    return gene_to_cats


def build_reactome_cell_biology_map(
    reactome_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Build gene -> [categories] mapping using cell-biology-relevant Reactome categories.

    Same as build_reactome_toplevel_map but filtered to the 17 categories
    directly relevant to OPS cell biology (excludes tissue/organism-specific
    and catch-all disease categories). See configs/reactome_cell_biology_filter.md.
    """
    gene_to_cats_full = build_reactome_toplevel_map(reactome_path)
    gene_to_cats: Dict[str, List[str]] = {}
    for gene, cats in gene_to_cats_full.items():
        filtered = [c for c in cats if c in REACTOME_CELL_BIOLOGY_CATEGORIES]
        if filtered:
            gene_to_cats[gene] = filtered

    n_genes = len(gene_to_cats)
    logger.info(
        f"Reactome cell-biology mapping: {n_genes} genes assigned "
        f"across {len(REACTOME_CELL_BIOLOGY_CATEGORIES)} categories "
        f"({n_genes - len(gene_to_cats_full)} genes dropped by category filter)"
    )
    return gene_to_cats


# ---------------------------------------------------------------------------
# Assignment helpers (used by the stage at scoring time)
# ---------------------------------------------------------------------------

def assign_genes_to_categories(
    gene_names: List[str],
    gene_to_cat: Dict[str, str],
    mapping_config: dict,
    boosted: bool = False,
) -> Dict[str, str]:
    """Assign a list of gene names to categories (single-mapping mode).

    Uses the pre-built mapping first. For genes not in the map and boosted=True,
    falls back to regex and Harmonizome.

    Returns dict: gene_name -> category_name. Unmatched genes map to "Other".
    """
    result: Dict[str, str] = {}
    categories = mapping_config.get("super_categories", {})

    # Precompile patterns per category (only if boosted)
    compiled_patterns: Dict[str, list] = {}
    if boosted:
        for cat_name, cat_def in categories.items():
            compiled_patterns[cat_name] = [
                re.compile(p) for p in cat_def.get("gene_patterns", [])
            ]

    for gene in gene_names:
        if gene in gene_to_cat:
            result[gene] = gene_to_cat[gene]
            continue

        matched = False
        if boosted:
            for cat_name, rxs in compiled_patterns.items():
                if any(rx.search(gene) for rx in rxs):
                    result[gene] = cat_name
                    matched = True
                    break
            if not matched and gene in _HARMONIZOME_OVERRIDES:
                result[gene] = _HARMONIZOME_OVERRIDES[gene]
                matched = True
        if not matched:
            result[gene] = "Other"

    return result


def assign_genes_to_categories_multi(
    gene_names: List[str],
    gene_to_cats: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Assign genes using multi-mapping (Reactome top-level mode).

    Returns dict: gene_name -> list of category names. Unmatched genes map to ["Other"].
    """
    result: Dict[str, List[str]] = {}
    for gene in gene_names:
        if gene in gene_to_cats:
            result[gene] = gene_to_cats[gene]
        else:
            result[gene] = ["Other"]
    return result
