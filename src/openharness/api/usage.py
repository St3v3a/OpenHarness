"""Provider usage and additive, per-request cost estimates (USD)."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

# Input tokens above this are billed in the long-context band.
LONG_CONTEXT_THRESHOLD = 272_000


@dataclass(frozen=True)
class Band:
    """USD per 1,000,000 tokens for one context band."""
    input: float
    cached_input: float
    cache_write: float
    output: float


@dataclass(frozen=True)
class ModelRates:
    short: Band
    long: Band


# Session-budget prices in USD (the budget and estimated_cost_usd are USD).
# Billing is separate: the EUR rows in model_pricing are not read here.
# Source for both: the Azure OpenAI pricing page
# (https://azure.microsoft.com/en-us/pricing/details/azure-openai/), Global
# rows, native USD; receipt output/2026-10-05-gpt-6.1-sol-pricing-page-source.json.
PRICED_MODELS: dict[str, ModelRates] = {
    # Unchanged from the 2026-09-21 rates (10 / 50, cached 0.1x, write 1.25x,
    # long band 2x input / 1.5x output), which the page's astra rows match.
    "gpt-6-astra": ModelRates(short=Band(10.00, 1.00, 12.50, 50.00),
                              long=Band(20.00, 2.00, 25.00, 75.00)),
    # "GPT-6.1 Sol (short/long context) Global". The 272k boundary is
    # astra's; the page does not state sol's.
    "gpt-6.1-sol": ModelRates(short=Band(2.00, 0.10, 2.50, 10.00),
                              long=Band(4.00, 0.20, 5.00, 15.00)),
}

# Fallback for models not in PRICED_MODELS (SessionBudget's flat path).
CONSERVATIVE_INPUT_RATE, CONSERVATIVE_OUTPUT_RATE = 10.0, 50.0

_RATE_ENV = {"input": "INVESTIGATION_COST_PER_M_INPUT", "output": "INVESTIGATION_COST_PER_M_OUTPUT"}


def env_rate(name: str) -> float | None:
    """A per-1M rate override from the environment, or None when unset.

    Non-numeric or 0 falls back (None) with a warning; NaN, infinity or a
    negative value raises ValueError so a bad deploy fails at startup.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; ignoring it", name, raw)
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if value == 0:
        logger.warning("%s=0 would price every token at zero; ignoring it", name)
        return None
    return value


def scaled_band(band: Band, input_rate: float | None, output_rate: float | None) -> Band:
    """``band`` with input/output replaced by the given rates. Cached and
    cache-write input keep their ratio to input, so an override keeps the
    model's cache structure."""
    factor = (input_rate / band.input) if input_rate is not None else 1.0
    return Band(
        input=band.input * factor,
        cached_input=band.cached_input * factor,
        cache_write=band.cache_write * factor,
        output=output_rate if output_rate is not None else band.output,
    )


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
        """Price one request, never an aggregated session."""
        return self.price_for_model("gpt-6-astra")

    def price_for_model(
        self, model: str, *, input_rate: float | None = None, output_rate: float | None = None,
    ) -> "UsageSnapshot":
        """Price one request on ``model``'s PRICED_MODELS rates, in USD.

        ``input_rate``/``output_rate`` (else INVESTIGATION_COST_PER_M_INPUT /
        _OUTPUT) replace the short-band base rates; the long band scales by
        the same factor. Raises KeyError for a model not in PRICED_MODELS.
        """
        rates = PRICED_MODELS[model]
        if input_rate is None:
            input_rate = env_rate(_RATE_ENV["input"])
        if output_rate is None:
            output_rate = env_rate(_RATE_ENV["output"])
        for rate in (input_rate, output_rate):
            if rate is not None and (not math.isfinite(rate) or rate < 0):
                raise ValueError("Pricing rates must be finite and nonnegative")
        in_factor = input_rate / rates.short.input if input_rate is not None else None
        out_factor = output_rate / rates.short.output if output_rate is not None else None
        elevated = self.input_tokens > LONG_CONTEXT_THRESHOLD
        band = rates.long if elevated else rates.short
        band = scaled_band(
            band,
            band.input * in_factor if in_factor is not None else None,
            band.output * out_factor if out_factor is not None else None,
        )
        # Missing cache details: charge ordinary input at the cache-write rate
        # (conservative).
        ordinary_rate = band.cache_write if self.cache_details_missing_turns else band.input
        self.estimated_cost_usd = (
            self.ordinary_input_tokens * ordinary_rate
            + self.cached_input_tokens * band.cached_input
            + self.cache_write_input_tokens * band.cache_write
            + self.output_tokens * band.output
        ) / 1_000_000
        self.estimated_uncached_cost_usd = (
            self.input_tokens * band.input + self.output_tokens * band.output
        ) / 1_000_000
        self.elevated_price_turns = int(elevated)
        return self
