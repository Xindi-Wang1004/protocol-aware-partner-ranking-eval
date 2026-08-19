# Protocol-aware partner ranking — diagnostic evaluation resource

**Releases:** `diagnostic-eval` (frozen PPI stack) · `partner-rank-eval-v0.1.0` (reporting CLI)  
**GitHub:** https://github.com/Xindi-Wang1004/protocol-aware-partner-ranking-eval  
**Companion to:** *Protocol-aware benchmarking of constrained biological partner ranking: a case study in virus–host PPI*.

This repository provides the **frozen diagnostic retrieve-then-rerank stack** used in the manuscript (not a new SOTA predictor), plus **`partner_rank_eval/`**, a small reporting CLI that turns per-query ranks into Recall@T / oracle@K / Hit@K, the definitional identity residual, and a fillable reporting-checklist JSON (manuscript Table 6).

## Contents

| Path | Description |
|------|-------------|
| `partner_rank_eval/` | Reporting CLI (`pip install -e .`; `partner-rank-eval metrics\|simulate\|checklist`). See `partner_rank_eval/TUTORIAL.md`. |
| `checkpoints/` | Retrieval (`gnn_tau007_longrun`) and classification (`esm3_frozen`) diagnostic checkpoints |
| `embed_cache/` | Precomputed HVIDB-2104 virus gallery embeddings |
| `data/` | Processed HVIDB train/val/test and IntAct cross-test pair files; `SV.fasta` |
| `scripts/` | Evaluation / bootstrap / baseline scripts used for Tables 3–6 and supplements |
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
- Code + lightweight processed files: this GitHub repository (releases `diagnostic-eval` and `partner-rank-eval-v0.1.0`)
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

