"""
Turn a Hugging Face function-calling chat dataset into scenario candidates (reference only).

Default: khaimk/ecommerce-function-calling-datasets-korean (no license stated: use locally, do not redistribute).
The source tool calls are kept as `source_calls` to help labeling against our own tools.

uv run --group hf python -m playground.dataset.import_hf --out playground/data/candidates_hf.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def as_messages(row: dict) -> list[dict]:
    for k in ("messages", "conversations", "conversation", "dialog"):
        v = row.get(k)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                continue
        if isinstance(v, list):
            return v
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="khaimk/ecommerce-function-calling-datasets-korean")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("playground/data/candidates_hf.jsonl"))
    ap.add_argument("--max", type=int, default=500)
    a = ap.parse_args()
    from datasets import load_dataset  # pyrefly: ignore[missing-import]

    ds = load_dataset(a.dataset, split=a.split)
    print("columns:", ds.column_names)
    rows = []
    for i, row in enumerate(ds):
        msgs = as_messages(row)
        users = [m.get("content") for m in msgs if m.get("role") in ("user", "human") and m.get("content")]
        calls = []
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        if not users:
            continue
        rows.append(
            {
                "id": f"HF{i + 1:04d}",
                "source": f"hf:{a.dataset}#{i}",
                "category": None,
                "turns": [{"text": u} for u in users],
                "context": msgs,
                "source_calls": calls,
                "expect": None,
            }
        )
        if len(rows) >= a.max:
            break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} candidates -> {a.out}")


if __name__ == "__main__":
    main()
