# partner-rank-eval

Protocol-aware metrics for **constrained biological partner ranking**.
This is the software artifact for the reporting checklist (manuscript Table 6).
It does not train or promote a predictor.

Walkthrough: [TUTORIAL.md](TUTORIAL.md). Frozen flags and JSON keys: [CLI_INTERFACE.md](CLI_INTERFACE.md).

## Identity (not an empirical discovery)

On one query set, with \(K \le T\):

\[
\mathrm{Hit}@K_{\text{end-to-end}} = \mathrm{Recall}@T \times \mathrm{oracle}@K
\]

where `oracle@K` is Hit@K **conditional** on the true partner appearing in the top-\(T\) retrieval shortlist. Headline Hit@K is monotonically non-decreasing as the evaluation gallery shrinks (equivalently, non-increasing as the gallery grows). Supplementary Note 1 states the lemma; Supplementary Figure S4 is `examples/plot_simulate.py`.

Frozen HVIDB-test Table 4 arms have residual 0 (GNN, popularity, kNN, ESM3 cosine):

```bash
python examples/replay_table4_identity.py
```

## Install

```bash
pip install -e ".[dev,plot]"
partner-rank-eval --version
```

## CLI

```bash
# 1) identity grid + gallery nesting
partner-rank-eval simulate --n 2000 --out examples/simulate.json

# 2) metrics from per-query ranks
partner-rank-eval metrics --rankings examples/toy_ranks.csv --T 20 --K 10 \
  --protocol fixed-gallery --gallery "toy n=1000" --checklist --out examples/metrics.json

# 3) empty reporting-checklist template
partner-rank-eval checklist --out examples/checklist_template.json
```

CSV or JSONL columns: `query_id,true_id,retrieval_rank,rerank_rank,gallery_size,in_gallery`.
Ranks are **1-based**. Leave `rerank_rank` empty when the true partner missed the shortlist.
`K > T` exits with code 2.

## Python

```python
from partner_rank_eval import simulate_retrieve_rerank, decompose

m = simulate_retrieve_rerank(recall_at_T=0.2, oracle_hit=0.9, seed=0)
print(m["end_to_end_hit@K"], m["recall_x_oracle"], decompose(m))
```

## Tests

```bash
pytest
```

## Licence

MIT. Do not redistribute BindingDB merged tables through this package; the CLI consumes ranks you already computed.
