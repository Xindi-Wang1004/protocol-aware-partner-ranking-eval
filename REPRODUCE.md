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

- Diagnostic stack: frozen ESM3; LoRA adapters present in architecture but **not trained** (zero-effect).
- Replace-rerank full-sequence passes are slow; some NPZ fields may be zeros if computed with `--skip-replace`.
