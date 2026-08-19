# Frozen Table 4 identity check (HVIDB test, n=3000, T=20, K=10, α=0.7).
# Hit@10 = Recall@20 × oracle@20; residual is definitional, not a finding.
# Source: constrained_2104_sensitivity_baselines_extended_hvidb_test.json
#         + exp1_esm3_cosine_hvidb_test.json (2026-08-18).

from __future__ import annotations

import json
from pathlib import Path

ARMS = Path(__file__).with_name("table4_hvidb_identity.json")


def check(name: str, recall: float, oracle: float, e2e: float) -> None:
    prod = recall * oracle
    err = abs(e2e - prod)
    print(
        f"{name:32s} R@20={recall:.6f}  oracle={oracle:.6f}  "
        f"R×q={prod:.6f}  Hit@10={e2e:.6f}  |err|={err:.2e}"
    )
    if err > 1e-12:
        raise SystemExit(f"identity failed: {name}")


def main() -> None:
    payload = json.loads(ARMS.read_text())
    for row in payload["arms"]:
        check(row["name"], row["recall@20"], row["oracle@K"], row["end_to_end_hit@K"])
    print("OK: identity holds on all frozen Table 4 HVIDB-test arms")


if __name__ == "__main__":
    main()
