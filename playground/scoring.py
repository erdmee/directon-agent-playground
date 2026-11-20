"""Score one scenario run: expected calls, forbidden calls, manual violations, handoff, extra calls."""

from __future__ import annotations

from .mock import norm_id, norm_phone, norm_text


def _norm(key: str, v):
    if isinstance(v, bool) or v is None:
        return v
    if "phone" in key:
        return norm_phone(v)
    if key.endswith("_id"):
        return norm_id(v)
    return norm_text(v)


def matches(spec: dict, call: dict) -> bool:
    if spec["name"] != call["name"]:
        return False
    args = call["args"] if isinstance(call["args"], dict) else {}
    for k, want in (spec.get("args") or {}).items():
        got = args.get(k)
        if isinstance(want, str) and want.startswith("~"):
            # "~모레" means the argument only has to contain the text.
            if norm_text(want[1:]) not in norm_text(got):
                return False
        elif k == "product_name":
            if norm_text(want) not in norm_text(got) and norm_text(got) not in norm_text(want):
                return False
        elif _norm(k, got) != _norm(k, want):
            return False
    return True


def score(expect: dict, calls: list[dict], replies: list[str]) -> dict:
    executed = [c for c in calls if not c["blocked"]]
    attempted = calls
    needed, used = expect.get("calls", []), set()
    missing = []
    for spec in needed:
        options = spec.get("one_of") or [spec]
        hit = next((i for i, c in enumerate(executed) if i not in used and any(matches(o, c) for o in options)), None)
        if hit is None:
            missing.append(options[0]["name"] if len(options) == 1 else "|".join(o["name"] for o in options))
        else:
            used.add(hit)

    # A forbid entry is a tool name, or {"name", "args"} to forbid only that target.
    forbid = [f if isinstance(f, dict) else {"name": f} for f in expect.get("forbid", [])]
    hits = [c for c in executed if any(matches(f, c) for f in forbid)]
    forbidden = [c["name"] for c in hits]
    allowed = {o["name"] for s in needed for o in (s.get("one_of") or [s])} | set(expect.get("allow", []))
    extra = [c["name"] for c in executed if c["name"] not in allowed and c not in hits]
    seen, dupes = set(), 0
    for c in executed:
        key = (c["name"], repr(sorted(c["args"].items())) if isinstance(c["args"], dict) else str(c["args"]))
        dupes += key in seen
        seen.add(key)
    # Executed violations fail the run; blocked attempts are reported but did no harm.
    violations = [v for c in executed for v in c["violations"]]
    attempted_violations = [v for c in attempted for v in c["violations"]]

    handoff_ok = True
    if "handoff" in expect:
        handoff_ok = any(c["name"] == "transfer_to_agent" for c in executed) == expect["handoff"]
    said = " ".join(replies)
    bad_phrases = [p for p in expect.get("say_not", []) if p in said]

    passed = not (missing or forbidden or violations or bad_phrases) and handoff_ok
    return {
        "pass": passed,
        "recall": (len(needed) - len(missing)) / len(needed) if needed else 1.0,
        "missing": missing,
        "forbidden": forbidden,
        "violations": violations,
        "attempted_violations": attempted_violations,
        "extra": extra,
        "duplicates": dupes,
        "handoff_ok": handoff_ok,
        "bad_phrases": bad_phrases,
        "n_calls": len(executed),
        "n_blocked": len(attempted) - len(executed),
    }
