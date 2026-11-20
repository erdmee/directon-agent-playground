"""
Turn AI Hub call-center JSON (e.g. 민원(콜센터) 질의-응답, 상담 음성) into scenario candidates.

AI Hub schemas differ per dataset, so the key names are flags with common defaults.
Run once with --inspect to print the keys of the first records, then set the flags.

uv run python -m playground.dataset.import_aihub --src ~/aihub/쇼핑 --inspect
uv run python -m playground.dataset.import_aihub --src ~/aihub/쇼핑 --out playground/data/candidates_aihub.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_KEYWORDS = "배송,택배,주문,취소,반품,환불,교환,운송장,도착,결제,상품,재고,섭취,영양제"


def records(obj, path: str = ""):
    """Yield every flat dict that looks like one utterance record, wherever it sits in the JSON."""
    if isinstance(obj, dict):
        if any(isinstance(v, str) for v in obj.values()) and not any(isinstance(v, list) for v in obj.values()):
            yield obj
        for v in obj.values():
            yield from records(v, path)
    elif isinstance(obj, list):
        for v in obj:
            yield from records(v, path)


def pick(rec: dict, keys: list[str]) -> str | None:
    for k in keys:
        if rec.get(k) not in (None, ""):
            return str(rec[k])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder with AI Hub label JSON files")
    ap.add_argument("--out", type=Path, default=Path("playground/data/candidates_aihub.jsonl"))
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dialog-key", default="대화셋일련번호,dialog_id,dialogue_id,conversation_id,id")
    ap.add_argument("--speaker-key", default="화자,speaker,speaker_type,role")
    ap.add_argument("--customer", default="고객,customer,client,C,user")
    ap.add_argument("--text-key", default="문장,발화,text,utterance,transcription,고객질문(요청),고객답변")
    ap.add_argument("--audio-key", default="파일명,audio,wav,file_name,filename")
    ap.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="keep dialogs whose customer turns contain one")
    ap.add_argument("--max", type=int, default=500)
    a = ap.parse_args()
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731

    files = sorted(a.src.rglob("*.json"))
    if a.inspect:
        for f in files[:3]:
            recs = list(records(json.loads(f.read_text(encoding="utf-8"))))
            print(f"== {f} ({len(recs)} records)")
            for r in recs[:3]:
                print(json.dumps(r, ensure_ascii=False)[:400])
        return

    dkeys, skeys, tkeys, akeys = split(a.dialog_key), split(a.speaker_key), split(a.text_key), split(a.audio_key)
    customers = {c.lower() for c in split(a.customer)}
    keywords = split(a.keywords)
    dialogs: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        for r in records(json.loads(f.read_text(encoding="utf-8"))):
            text = pick(r, tkeys)
            if not text:
                continue
            did = f"{f.stem}:{pick(r, dkeys) or 'all'}"
            speaker = (pick(r, skeys) or "").lower()
            dialogs[did].append(
                {
                    "role": "customer" if speaker in customers else (speaker or "agent"),
                    "text": text.strip(),
                    "audio": pick(r, akeys),
                }
            )

    rows = []
    for did, utts in dialogs.items():
        cust = [u for u in utts if u["role"] == "customer"]
        if not cust or not any(k in u["text"] for u in cust for k in keywords):
            continue
        rows.append(
            {
                "id": f"AH{len(rows) + 1:04d}",
                "source": f"aihub:{did}",
                "category": None,
                "turns": [{"text": u["text"], **({"audio_ref": u["audio"]} if u["audio"] else {})} for u in cust],
                "context": utts,
                "expect": None,
            }
        )
        if len(rows) >= a.max:
            break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(files)} files, {len(dialogs)} dialogs, {len(rows)} kept -> {a.out}")


if __name__ == "__main__":
    main()
