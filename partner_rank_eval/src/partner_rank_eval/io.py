"""Load per-query rankings from CSV or JSONL."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from .metrics import QueryRecord


def load_records(path: str | Path) -> List[QueryRecord]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return list(_from_jsonl(path))
    if path.suffix.lower() in {".json"}:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return [_record_from_dict(row) for row in payload]
        raise ValueError("JSON rankings must be a list of query objects")
    return list(_from_csv(path))


def _from_csv(path: Path) -> Iterable[QueryRecord]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"query_id", "true_id", "retrieval_rank"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must have columns query_id,true_id,retrieval_rank "
                "[rerank_rank,gallery_size,in_gallery]"
            )
        for row in reader:
            yield _record_from_dict(row)


def _from_jsonl(path: Path) -> Iterable[QueryRecord]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield _record_from_dict(json.loads(line))


def _record_from_dict(row: dict) -> QueryRecord:
    rerank = row.get("rerank_rank")
    if rerank == "" or rerank is None:
        rerank_i = None
    else:
        rerank_i = int(rerank)
    gal = row.get("gallery_size")
    gal_i = int(gal) if gal not in (None, "") else None
    in_g = row.get("in_gallery", True)
    if isinstance(in_g, str):
        in_g = in_g.strip().lower() in {"1", "true", "yes"}
    return QueryRecord(
        query_id=str(row["query_id"]),
        true_id=str(row["true_id"]),
        retrieval_rank=int(row["retrieval_rank"]),
        rerank_rank=rerank_i,
        gallery_size=gal_i,
        in_gallery=bool(in_g),
    )
