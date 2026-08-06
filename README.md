# Protocol-aware partner ranking — diagnostic evaluation resource

**Release:** `diagnostic-eval`  
**GitHub:** https://github.com/Xindi-Wang1004/protocol-aware-partner-ranking-eval  
**Companion to:** *Protocol-aware benchmarking of constrained biological partner ranking: a case study in virus–host PPI* (Briefings in Bioinformatics, Problem Solving Protocol).

This repository provides the **frozen diagnostic retrieve-then-rerank stack** used in the manuscript (not a new SOTA predictor).

## Contents

| Path | Description |
|------|-------------|
| `checkpoints/` | Retrieval (`gnn_tau007_longrun`) and classification (`esm3_frozen`) diagnostic checkpoints |
| `embed_cache/` | Precomputed HVIDB-2104 virus gallery embeddings |
| `data/` | Processed HVIDB train/val/test and IntAct cross-test pair files; `SV.fasta` |
| `scripts/` | Evaluation / bootstrap / baseline scripts used for Tables 3–5 and supplements |
| `models/`, `config/` | Minimal Python modules to instantiate/load checkpoints |
| `PUSH.md` | How to publish to GitHub + Zenodo |
| `analysis_results/` | Published JSON/NPZ underlying manuscript tables and figures |
| `MANIFEST.json` | Exact file list and sizes |

## Not included (by design)

- ESM3 **base** weights (download separately under EvolutionaryScale / Hayes *et al.* 2025 license)
- Training runs, LoRA fine-tuning experiments, exploratory smoke outputs, drafts
- Unrelated documentation and plots from the development tree

## Data availability (BIB)

- Upstream sources: [HVIDB](https://doi.org/10.1093/bib/bbaa425), [IntAct](https://doi.org/10.1093/nar/gkt1115)
- Large binaries (checkpoints + `train_protein_pairs.pkl`): [Zenodo](https://doi.org/10.5281/zenodo.21826320) (`10.5281/zenodo.21826320`)
- Code + lightweight processed files: this GitHub repository (release `diagnostic-eval`)
- Code license: MIT · Processed data: CC BY 4.0

## Reproduce

See `REPRODUCE.md`.

## Citation

Wang X, Li Y, Hon C. Protocol-aware benchmarking of constrained biological partner ranking: a case study in virus–host PPI. *Briefings in Bioinformatics* (submitted).

## Large files

Checkpoints (~5.3 GB each) and `train_protein_pairs.pkl` (~386 MB) are on Zenodo:

- **DOI:** https://doi.org/10.5281/zenodo.21826320
- Record: https://zenodo.org/records/21826320

Place downloaded files under `checkpoints/` and `data/processed/` as named in `checkpoints/DOWNLOAD.txt` and `data/processed/DOWNLOAD.txt`.

