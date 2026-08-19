# Release notes

Do **not** upload BindingDB `evalset.json` or the merged TSV. Ship scripts + summary metrics only.

## GitHub

Target repo: https://github.com/Xindi-Wang1004/protocol-aware-partner-ranking-eval

If this package is nested as `partner_rank_eval/`, copy `.github/workflows/test.yml` to the repo root and set `defaults.run.working-directory: partner_rank_eval`. If this directory is the repository root, the workflow already runs from `.`.

```bash
git add partner_rank_eval   # or the repo root files, if this directory is the root
git commit -m "Add partner-rank-eval reporting CLI."
git tag -a partner-rank-eval -m "partner-rank-eval"
git push origin HEAD
git push origin partner-rank-eval
```

Create a GitHub release from that tag. Existing diagnostic scripts stay on release `diagnostic-eval`.

## Zenodo

The manuscript now cites https://doi.org/10.5281/zenodo.22009119.
After the GitHub tag is public, refresh that deposit (new version) **or** mint a software-specific DOI from the GitHub release.

Upload the source tree of this package (LICENSE, tests, examples, no `.venv`, no BindingDB dumps).
Record the new DOI in Data/Code availability before submission.

## Local sdist check

```bash
python -m pip install build
python -m build
python -m pytest
```
