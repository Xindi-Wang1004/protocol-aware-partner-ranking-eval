# Reproduce

## Environment

```bash
conda activate protein   # Python 3.10 + PyTorch + CUDA (same as server 14)
```

## ESM3 base weights

Not shipped. Place ESM3 open weights where the project loaders expect them (same layout as training).

## Layout for re-run scripts

Scripts expect a project root with original relative paths. From this release:

```bash
export PROJECT_ROOT=$(pwd)
mkdir -p retrieval_task/gnn_tau007_longrun classification_task/esm3_frozen analysis_results/embed_cache
ln -sf ../../checkpoints/retrieval_gnn_tau007_longrun_best_model.pt retrieval_task/gnn_tau007_longrun/best_model.pt
ln -sf ../../checkpoints/classification_esm3_frozen_best_model.pt classification_task/esm3_frozen/best_model.pt
cp embed_cache/*.pt analysis_results/embed_cache/
# data/ is already at project root in this release
```

Core modules may be present as `scripts/__pycache__/*.cpython-310.pyc` (no `.py` source for some eval helpers). Ensure `scripts/` is on `PYTHONPATH` or run from project root as on server 14.

## Recompute primary cluster bootstrap (Table 3)

```bash
python scripts/bib_cluster_and_baselines.py --dataset hvidb_test --device cuda:0
```

Outputs write under `analysis_results/`. Compare to shipped JSON for numerical agreement.

## Notes

- Diagnostic stack: frozen ESM3 embeddings with trainable projection heads; inactive LoRA modules retained for upstream code compatibility (see Supplementary Table S2 / manuscript).
- Replace-rerank full-sequence passes are slow; some NPZ fields may be zeros if computed with `--skip-replace`.


## BindingDB second-source (Articles TSV)

Provenance used in the manuscript:

- URL: https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202608_tsv.zip
- Downloaded: 18 August 2026
- SHA256: `2529b1c572aa7b298e57355e251c9a9572b82dd35f97259d5e3866ee69bfada5`
- Licence: CC-BY (BindingDB-curated Articles only; not the All dump)

```bash
python scripts/download_bindingdb_articles.py --out data/bindingdb/BindingDB_BindingDB_Articles_202608_tsv.zip
python scripts/bindingdb_partner_distribution.py --tsv data/bindingdb/BindingDB_BindingDB_Articles_202608_tsv.zip --out analysis_results/bindingdb/
python scripts/bindingdb_weak_retrieve_rerank.py --evalset analysis_results/bindingdb/evalset.json --out analysis_results/bindingdb/weak_retrieve_rerank.json
```

Pair-subset saturation diagnostics (Supplementary Table S15):

```bash
python transfer_bioinformatics/scripts/compute_pair_subset_saturation.py
# or from a checkout that contains transfer_bioinformatics/analysis outputs
```
