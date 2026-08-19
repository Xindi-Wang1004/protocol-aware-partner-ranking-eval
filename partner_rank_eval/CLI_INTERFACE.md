# CLI interface lock (v0.1)

Frozen 2026-08-19. Implementation may grow next week; **do not rename these commands, flags, or JSON keys** without a version bump.

The deliverable is a reporting tool, not a predictor.

## Commands

```
partner-rank-eval metrics    --rankings FILE [--T 20] [--K 10] [--protocol NAME] [--gallery TEXT] [--checklist] [--out FILE]
partner-rank-eval simulate   [--n 2000] [--gallery 1000] [--seed 0] [--out FILE]
partner-rank-eval checklist  [--metrics FILE] [--out FILE]
```

Defaults: `T=20`, `K=10`. `K > T` is an error (identity undefined; CLI exit code 2).
`partner-rank-eval --version` prints `partner-rank-eval 0.1.0`.

## Input (`metrics`)

`--rankings` is CSV or JSONL. Required columns:

| column | meaning |
|---|---|
| `query_id` | query identifier |
| `true_id` | labelled true partner |
| `retrieval_rank` | 1-based rank in the stated evaluation gallery |
| `rerank_rank` | 1-based rank on the top-T shortlist; empty if not retrieved |
| `gallery_size` | optional; size of the stated evaluation gallery |
| `in_gallery` | optional; true if the labelled partner is eligible |

One row = one labelled query. Gallery membership is a column, not a separate file, in v0.1. A later `--queries` / `--gallery` pair may be added as a convenience loader; it must compile to the same row schema.

## Output (`metrics --checklist`)

JSON object with three blocks:

1. `metrics` — `n_queries`, `reachability`, `recall@T`, `end_to_end_hit@K`, `oracle@K`, `recall_x_oracle`, `identity_abs_error`, `n_oracle_subset`, `T`, `K`
2. `decomposition` — `headline_hit@K`, `retrieval_coverage_recall@T`, `conditional_rerank_oracle@K`, `gate_factor_missed_by_retrieval`
3. `checklist` — reporting-checklist template (`items[].id/item/value/reported`, `n_items`, `n_reported`); this is manuscript Table 6, but the CLI name is not a table number

Identity residual must be reported even when it is 0.

## Output (`simulate`)

`identity_grid` over `r ∈ {0.05, 0.2, 0.5, 0.8}` and `q ∈ {0.5, 0.9}`; `gallery_nesting` with the same underlying scores and shrinking galleries. Headline Hit@K is monotonically non-decreasing as the gallery shrinks (equivalently, non-increasing as the gallery grows).

## Output (`checklist`)

Empty reporting-checklist JSON if `--metrics` is omitted; filled from a metrics JSON otherwise.
