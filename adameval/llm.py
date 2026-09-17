"""Model client, cost accounting, and a zero-cost mock for development.

Nothing in this module is allowed to be imported by adameval.gates. The gate
layer never calls a model; keeping the dependency one-directional is what makes
that claim checkable rather than aspirational.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

MODEL = "claude-opus-5"

# $ per million tokens. Cache write is 1.25x input, cache read 0.1x input.
PRICES = {
    "claude-opus-5":   {"in": 5.00, "out": 25.00, "cache_w": 6.25, "cache_r": 0.50},
    "claude-sonnet-5": {"in": 2.00, "out": 10.00, "cache_w": 2.50, "cache_r": 0.20},
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(self.input_tokens + o.input_tokens,
                     self.output_tokens + o.output_tokens,
                     self.cache_read + o.cache_read,
                     self.cache_write + o.cache_write)

    def cost(self, model: str = MODEL) -> float:
        p = PRICES.get(model, PRICES[MODEL])
        return (self.input_tokens * p["in"] + self.output_tokens * p["out"]
                + self.cache_read * p["cache_r"] + self.cache_write * p["cache_w"]) / 1e6


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    """Hard spend ceiling. The grid aborts rather than overrunning."""
    budget_usd: float
    total: Usage = field(default_factory=Usage)
    model: str = MODEL
    calls: int = 0

    @property
    def spent(self) -> float:
        return self.total.cost(self.model)

    def charge(self, u: Usage) -> None:
        self.total = self.total + u
        self.calls += 1
        if self.spent > self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.spent:.2f} of ${self.budget_usd:.2f} budget "
                f"over {self.calls} calls")


class Client(Protocol):
    def complete(self, system: list[dict], messages: list[dict]) -> tuple[str, Usage]: ...


class AnthropicClient:
    def __init__(self, model: str = MODEL, max_tokens: int = 16000):
        import anthropic
        self._c = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: list[dict], messages: list[dict]) -> tuple[str, Usage]:
        with self._c.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"model declined: {getattr(msg, 'stop_details', None)}")
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        return text, Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )


class MockClient:
    """Returns canned responses so the runner, repair loop, retry cap, CLI and
    provenance dump can be exercised end to end for $0."""

    def __init__(self, responses: list[str] | Callable[[int], str]):
        self._r, self.n = responses, 0

    def complete(self, system: list[dict], messages: list[dict]) -> tuple[str, Usage]:
        r = self._r(self.n) if callable(self._r) else self._r[min(self.n, len(self._r) - 1)]
        self.n += 1
        return r, Usage(input_tokens=3500, output_tokens=1200)


CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text)
    return max(blocks, key=len).strip() if blocks else None
