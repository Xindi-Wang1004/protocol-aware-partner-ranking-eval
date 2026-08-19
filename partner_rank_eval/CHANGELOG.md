# Changelog

## 0.1.0 — 2026-08-19

First locked CLI (`CLI_INTERFACE.md`):

- `partner-rank-eval metrics` — Recall@T, oracle@K, Hit@K, identity residual, optional checklist
- `partner-rank-eval simulate` — identity grid \(r \times q\) and nested-gallery monotonicity
- `partner-rank-eval checklist` — empty or filled reporting-checklist JSON (manuscript Table 6; CLI name is not a table number)

JSON output uses `null` rather than `NaN`. `K > T` exits with code 2.
