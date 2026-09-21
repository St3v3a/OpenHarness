"""Provider usage and additive, per-request cost estimates (USD)."""
from __future__ import annotations

import logging
import math
import os
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class UsageSnapshot(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_uncached_cost_usd: float = 0.0
    elevated_price_turns: int = 0
    cache_details_missing_turns: int = 0

    @model_validator(mode="after")
    def validate_counts(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.cached_input_tokens + self.cache_write_input_tokens > self.input_tokens:
            logger.warning("Invalid cache usage counts; charging input as cache writes")
            self.cached_input_tokens = 0
            self.cache_write_input_tokens = self.input_tokens
        self.reasoning_output_tokens = min(self.reasoning_output_tokens, self.output_tokens)
        return self

    @property
    def ordinary_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens - self.cache_write_input_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def price_astra(self) -> "UsageSnapshot":
        """Price one request, never an aggregated session. Rates dated 2026-09-21."""
        input_rate = float(os.getenv("INVESTIGATION_COST_PER_M_INPUT") or 10)
        output_rate = float(os.getenv("INVESTIGATION_COST_PER_M_OUTPUT") or 50)
        if not all(math.isfinite(rate) and rate >= 0 for rate in (input_rate, output_rate)):
            raise ValueError("Pricing rates must be finite and nonnegative")
        elevated = self.input_tokens > 272_000
        input_multiplier, output_multiplier = (2, 1.5) if elevated else (1, 1)
        self.estimated_cost_usd = (
            (input_rate * (self.ordinary_input_tokens * (1.25 if self.cache_details_missing_turns else 1) + self.cached_input_tokens * .1 + self.cache_write_input_tokens * 1.25)) * input_multiplier
            + self.output_tokens * output_rate * output_multiplier
        ) / 1_000_000
        self.estimated_uncached_cost_usd = (
            self.input_tokens * input_rate * input_multiplier + self.output_tokens * output_rate * output_multiplier
        ) / 1_000_000
        self.elevated_price_turns = int(elevated)
        return self
