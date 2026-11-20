"""Tool execution for one session: in-process mock or an external Mock API, plus a call log."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

import httpx

from .config import env
from .mock import MockBackend
from .policy import PolicyTracker


@dataclass
class CallRecord:
    name: str
    args: dict
    result: dict
    ms: float
    turn: int
    violations: list[str] = field(default_factory=list)
    blocked: bool = False


class ToolRunner:
    """
    Strategies call `run`; the evaluator reads `calls`.

    guard=True turns policy violations into a `blocked` result instead of executing,
    so you can compare an unguarded agent against the same agent behind a rule guard.
    """

    def __init__(self, now: str, tools_url: str | None = None, guard: bool = False, session_id: str | None = None):
        self.backend = MockBackend(now)
        self.policy = PolicyTracker(self.backend)
        self.tools_url = (tools_url if tools_url is not None else env("TOOLS_URL")) or None
        self.guard = guard
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.calls: list[CallRecord] = []
        self.turn = 0
        self.last_user = ""
        self._http: httpx.AsyncClient | None = None

    def new_turn(self, user_text: str) -> None:
        self.turn += 1
        self.last_user = user_text

    async def run(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        violations = self.policy.check(name, args, self.last_user)
        if self.guard and violations:
            result = {"status": "blocked", "message": "; ".join(violations)}
            blocked = True
        else:
            result = await self._execute(name, args)
            blocked = False
            self.policy.observe(name, args, result)
        self.calls.append(
            CallRecord(name, args, result, (time.perf_counter() - t0) * 1000, self.turn, violations, blocked)
        )
        return result

    async def _execute(self, name: str, args: dict) -> dict:
        if not self.tools_url:
            return self.backend.call(name, args)
        # External Mock API: the local backend still mirrors state for the policy checker.
        self._http = self._http or httpx.AsyncClient(timeout=10)
        r = await self._http.post(
            f"{self.tools_url.rstrip('/')}/tools/{name}", json=args, headers={"X-Session-Id": self.session_id}
        )
        self.backend.call(name, args)
        try:
            return r.json()
        except ValueError:
            return {"status": "error", "message": f"HTTP {r.status_code}"}

    def log(self) -> list[dict]:
        return [asdict(c) for c in self.calls]

    async def aclose(self) -> None:
        if self._http:
            await self._http.aclose()
