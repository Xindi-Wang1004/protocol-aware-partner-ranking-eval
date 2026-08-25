#!/usr/bin/env python3
"""Download BindingDB curated Articles TSV and verify SHA256.

Provenance used in the manuscript:
  URL: https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202608_tsv.zip
  Downloaded: 18 August 2026
  SHA256: 2529b1c572aa7b298e57355e251c9a9572b82dd35f97259d5e3866ee69bfada5
  Licence: CC-BY (BindingDB-curated Articles; not the All dump with ChEMBL imports)
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202608_tsv.zip"
EXPECTED_SHA256 = "2529b1c572aa7b298e57355e251c9a9572b82dd35f97259d5e3866ee69bfada5"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Output zip path")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        print(f"Downloading {args.url} -> {args.out}")
        urllib.request.urlretrieve(args.url, args.out)
    digest = sha256_file(args.out)
    print(f"SHA256 {digest}")
    if digest != args.expected_sha256:
        raise SystemExit(
            f"SHA256 mismatch:\n  got  {digest}\n  want {args.expected_sha256}"
        )
    print("OK: checksum matches manuscript provenance.")


if __name__ == "__main__":
    main()
