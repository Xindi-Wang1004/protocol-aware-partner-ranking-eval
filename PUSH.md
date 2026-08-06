# How to publish (run on server 14)

Release: `~/bib_release/protocol-aware-partner-ranking-eval` (~11 GB).

GitHub rejects files >100 MB. Use **GitHub for code + small JSON** and **Zenodo for checkpoints + large pkl**.

## A. GitHub (slim)

```bash
cd ~/bib_release
rm -rf github_repo
rsync -a --exclude 'checkpoints/*.pt' --exclude 'data/processed/train_protein_pairs.pkl' \
  protocol-aware-partner-ranking-eval/ github_repo/
cd github_repo
printf 'Binaries on Zenodo — see README.\n' > checkpoints/DOWNLOAD.txt
printf 'train_protein_pairs.pkl on Zenodo (~386 MB).\n' > data/processed/DOWNLOAD.txt
git init && git add . && git commit -m "diagnostic-eval: code, embeds, results"
# create repo on GitHub, then:
# git remote add origin git@github.com:YOUR_USER/protocol-aware-partner-ranking-eval.git
# git branch -M main && git push -u origin main
# git tag diagnostic-eval && git push origin diagnostic-eval
# Pages: Settings → main → /docs
```

## B. Zenodo (full binaries)

```bash
cd ~/bib_release
tar -cvf protocol-aware-partner-ranking-eval_v1.0.tar protocol-aware-partner-ranking-eval
# Upload tar to Zenodo → DOI → put DOI in manuscript Data availability
```

Manuscript URLs after publish:
- GitHub: https://github.com/YOUR_USER/protocol-aware-partner-ranking-eval
- Zenodo: https://doi.org/10.5281/zenodo.XXXX
