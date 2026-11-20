"""
Draft scenario labels from imported real dialogs with an LLM; a human reviews every draft.

For each candidate the model (1) decides whether it fits our domain, (2) rewrites the caller
turns minimally so names, phones and products exist in our mock data while keeping the
caller's real wording, fillers and disfluencies, and (3) drafts `expect` against our tools.

uv run python -m playground.dataset.label --in playground/data/candidates_aihub.jsonl \
    --out playground/data/scenarios_draft.jsonl
Review: set "review": "ok" (or fix the row) and append accepted rows to scenarios.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..config import DOMAIN, load_dotenv, load_manual, load_scenarios, load_tools
from ..llm import chat

INSTRUCTIONS = """You build evaluation scenarios for a Korean phone customer-service agent.
Given a real customer-service dialog, output JSON with:
- "usable": false if it cannot map to our shop (shipping status, cancel, return/refund, product info,
  transfer to a human, callback, opening hours); true otherwise.
- "category": short Korean label, e.g. 배송조회, 취소가능, 취소불가_배송중, 반품가능, 반품불가_개봉, 상품문의, 의학질문, 부작용신고, 상담원요청, FAQ, 콜백.
- "turns": the customer's turns only, in order. Keep the original wording, fillers (음, 어, 그) and spoken numbers,
  but replace names, phone numbers, order IDs and products with ones from MOCK DATA so tools can resolve them.
  Add a turn only if the customer must answer a question the agent will certainly ask (e.g. name and phone).
- "expect": {"calls": [{"name", "args"} or {"one_of": [...]}], "allow": [tool names that are fine but optional],
  "forbid": [tool names that must not run], "handoff": true/false}. Use only our tool names; put only the
  args that matter (ids, phone digits, opened). Follow the manual strictly (verification first, consent before
  cancel/return, no return after 7 days or if opened).
- "notes": one sentence on what this scenario tests."""


def build_prompt(c: dict) -> list[dict]:
    mock = json.loads((DOMAIN / "data.json").read_text(encoding="utf-8"))
    ctx = c.get("context") or [{"role": "customer", "text": t["text"]} for t in c["turns"]]
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "system", "content": "MANUAL:\n" + load_manual()},
        {
            "role": "system",
            "content": "TOOLS:\n"
            + json.dumps(
                [{"name": t["name"], "required": t["parameters"]["required"]} for t in load_tools()], ensure_ascii=False
            ),
        },
        {"role": "system", "content": "MOCK DATA:\n" + json.dumps(mock, ensure_ascii=False)},
        {
            "role": "user",
            "content": json.dumps({"dialog": ctx, "source_calls": c.get("source_calls")}, ensure_ascii=False),
        },
    ]


async def label_one(c: dict, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        res = await chat(build_prompt(c), response_format={"type": "json_object"})
    try:
        d = json.loads(res.text)
    except json.JSONDecodeError:
        print("bad JSON for", c["id"])
        return None
    if not d.get("usable"):
        return None
    return {
        "id": c["id"],
        "source": c["source"],
        "category": d.get("category"),
        "turns": d.get("turns") or [t["text"] for t in c["turns"]],
        "original_turns": [t["text"] for t in c["turns"]],
        "expect": d.get("expect") or {},
        "notes": d.get("notes"),
        "review": "pending",
    }


async def run(inp: Path, out: Path, limit: int, concurrency: int) -> None:
    cands = load_scenarios(inp)[:limit]
    sem = asyncio.Semaphore(concurrency)
    rows = [r for r in await asyncio.gather(*(label_one(c, sem) for c in cands)) if r]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(cands)} candidates, {len(rows)} usable drafts -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("playground/data/scenarios_draft.jsonl"))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    load_dotenv()
    asyncio.run(run(a.inp, a.out, a.limit, a.concurrency))


if __name__ == "__main__":
    main()
