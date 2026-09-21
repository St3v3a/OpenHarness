"""Local request-size policy for Astra's elevated-price boundary."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextCostPolicy:
    advisory: int = 180_000
    compact: int = 220_000
    hard: int = 250_000
    margin: int = 22_000
    allow_elevated: bool = False

    @classmethod
    def from_env(cls):
        policy = cls(
            int(os.getenv("OPENHARNESS_CONTEXT_ADVISORY_TOKENS", "180000")),
            int(os.getenv("OPENHARNESS_AUTO_COMPACT_THRESHOLD_TOKENS", "220000")),
            int(os.getenv("OPENHARNESS_CONTEXT_HARD_GUARD_TOKENS", "250000")),
            int(os.getenv("OPENHARNESS_CONTEXT_ESTIMATE_MARGIN_TOKENS", "22000")),
            os.getenv("OPENHARNESS_ALLOW_ELEVATED_PRICE_BAND", "disabled") == "enabled",
        )
        if not 0 < policy.advisory < policy.compact < policy.hard < 272_000 or policy.margin < 0:
            raise ValueError("Context thresholds must satisfy 0 < advisory < compact < hard < 272000; margin >= 0")
        if os.getenv("OPENHARNESS_ALLOW_ELEVATED_PRICE_BAND", "disabled") not in {"enabled", "disabled"}:
            raise ValueError("Invalid elevated-price override")
        return policy
