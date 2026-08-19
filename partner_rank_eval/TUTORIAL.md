# Tutorial: partner-rank-eval

This is a reporting tool, not a predictor. It takes **already computed ranks** and returns Recall@T, oracle@K, end-to-end Hit@K, the definitional identity residual, and a fillable reporting-checklist JSON (manuscript Table 6).

Commands, flags, and JSON keys are frozen in `CLI_INTERFACE.md` (v0.1). Do not rename them.

## 1. Install

```bash
python -m pip install -e ".[dev]"
partner-rank-eval --version
```

## 2. Simulate the identity and gallery nesting

```bash
partner-rank-eval simulate --n 2000 --gallery 1000 --seed 0 --out examples/simulate.json
```

`identity_grid` covers designed retrieval coverage \(r \in \{0.05, 0.2, 0.5, 0.8\}\) and rerank quality \(q \in \{0.5, 0.9\}\). On each cell, `identity_abs_error` is 0: with one labelled true partner and \(K \le T\),

\[
\mathrm{Hit}@K = \mathrm{Recall}@T \times \mathrm{oracle}@K
\]

by definition, not as an empirical finding.

`gallery_nesting` keeps the same underlying scores and shrinks the evaluation gallery (1000 → 200 → 20 → 10) while always keeping the true partner. Headline Hit@K is **monotonically non-decreasing as the gallery shrinks** (equivalently, non-increasing as the gallery grows). That is the same direction as the manuscript: a fixed large gallery yields a lower Hit@K than a pair-subset gallery.

Optional figure (needs `pip install -e ".[plot]"`):

```bash
PYTHONPATH=src python examples/plot_simulate.py
```

## 3. Score your own ranks

One row = one labelled query. Ranks are **1-based**. Leave `rerank_rank` empty when the true partner missed the top-\(T\) shortlist.

CSV (`examples/toy_ranks.csv`):

```text
query_id,true_id,retrieval_rank,rerank_rank,gallery_size,in_gallery
q0,t,2,1,1000,true
q2,t,80,,1000,true
```

JSONL is accepted with the same fields (`examples/toy_ranks.jsonl`).

```bash
partner-rank-eval metrics \
  --rankings examples/toy_ranks.csv \
  --T 20 --K 10 \
  --protocol fixed-gallery \
  --gallery "toy gallery n=1000" \
  --checklist \
  --out examples/metrics.json
```

Output blocks:

| block | what it is |
|---|---|
| `metrics` | `recall@T`, `end_to_end_hit@K`, `oracle@K`, `recall_x_oracle`, `identity_abs_error` (reported even when 0), `n_oracle_subset` |
| `decomposition` | names the retrieval gate vs conditional rerank factors |
| `checklist` | reporting-checklist JSON; `--protocol` and `--gallery` fill the corresponding items |

`K > T` is an error (exit code 2): the identity is undefined if reranking could promote a retrieval miss into Hit@K.

On the toy file, 6/8 queries are retrieved at \(T=20\) and 5 of those 6 are hits at \(K=10\), so Hit@10 = 0.625 = 0.75 × 0.833… and `identity_abs_error` is 0.

## 4. Empty or filled checklist

```bash
partner-rank-eval checklist --out examples/checklist_template.json
partner-rank-eval checklist --metrics examples/metrics.json --out /tmp/filled_checklist.json
```

Ten items; the CLI name is `checklist`, not a table number, so manuscript renumbering does not break the interface. Numeric items can be auto-filled from `metrics`; protocol/gallery/model-role still need a human statement.

## 5. Replay Table 4 (HVIDB test)

Frozen aggregates, no large binaries:

```bash
python examples/replay_table4_identity.py
```

All four arms (GNN, popularity, kNN, ESM3 cosine) have residual 0. The cosine-arm oracle \(n=25\) is a conditional diagnostic only.

## Python API

```python
from partner_rank_eval import (
    QueryRecord,
    decompose,
    metrics_from_ranks,
    simulate_retrieve_rerank,
)

m = simulate_retrieve_rerank(recall_at_T=0.2, oracle_hit=0.9, seed=0)
assert abs(m["end_to_end_hit@K"] - m["recall_x_oracle"]) < 1e-12
print(decompose(m))
```

## What this tool does not do

It does not train a retriever, does not claim DTI or PPI SOTA, and does not ship BindingDB ligand tables (those remain under BindingDB licences). Pass your own rank file.
