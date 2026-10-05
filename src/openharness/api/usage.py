"""Provider usage and additive, per-request cost estimates (USD)."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

# Conservative per-1M-token fallback, used for gpt-6-astra (rates dated
# 2026-09-21) and for any priced model before its rates take effect.
CONSERVATIVE_INPUT_RATE, CONSERVATIVE_OUTPUT_RATE = 10.0, 50.0


@dataclass(frozen=True)
class ModelRates:
    """Short-context per-1M-token rates for one model version. Cache reads and
    writes are priced as a ratio of ``input`` (so an INVESTIGATION_COST_PER_M_*
    override keeps the model's cache structure). Above 272,000 input tokens the
    long-context band charges 2x input / 1.5x output. ``effective_from``
    mirrors the dev1 model_pricing row; before a model's first version the
    conservative fallback applies."""
    input: float
    output: float
    cache_read_ratio: float = 0.1
    cache_write_ratio: float = 1.25
    effective_from: date | None = None


# Versions per model, oldest first; the latest with effective_from <= today wins.
PRICED_MODELS: dict[str, tuple[ModelRates, ...]] = {
    "gpt-6-astra": (ModelRates(CONSERVATIVE_INPUT_RATE, CONSERVATIVE_OUTPUT_RATE),),
    "gpt-6.1-sol": (
        # 2026-10-06 only: Azure Retail EUR westeurope Global Standard short-
        # context "6-sol" meters (cache read 0.176, write 2.1999), on the
        # assumption 6.1-sol bills like 6-sol. Receipt
        # output/2026-10-05-sol-westeurope-eur-source.json.
        ModelRates(1.7599, 8.7997, 0.1, 1.25, effective_from=date(2026, 10, 6)),
        # From 2026-10-07: Azure OpenAI pricing page
        # (https://azure.microsoft.com/en-us/pricing/details/azure-openai/),
        # "GPT-6.1 Sol (short context) Global", europe-west, USD 2.00 input /
        # 0.10 cached input / 2.50 cache writes / 10.00 output per 1M, converted
        # at the ECB reference rate for 2026-10-02 (1.1225 USD per EUR) to EUR
        # 1.781737 / 0.089087 / 2.227171 / 8.908686. Cached input is 0.05x input
        # (astra is 0.1x). The page's long-context row (4.00 / 15.00) matches the
        # 2x / 1.5x band; the 272k boundary itself is astra's and is not stated
        # for sol. Receipt output/2026-10-05-gpt-6.1-sol-pricing-page-source.json.
        ModelRates(1.781737, 8.908686, 0.05, 1.25, effective_from=date(2026, 10, 7)),
    ),
}


def rates_for(model: str, today: date | None = None) -> ModelRates:
    """The model's rates in force on ``today`` (UTC), else the conservative fallback."""
    day = today or _utc_today()
    current = None
    for version in PRICED_MODELS.get(model, ()):
        if version.effective_from is None or version.effective_from <= day:
            current = version
    return current or ModelRates(CONSERVATIVE_INPUT_RATE, CONSERVATIVE_OUTPUT_RATE)


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


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
        return self.price_for_model("gpt-6-astra")

    def price_for_model(self, model: str, *, today: date | None = None) -> "UsageSnapshot":
        """Price one request on ``model``'s PRICED_MODELS rates.

        INVESTIGATION_COST_PER_M_INPUT/OUTPUT, when set, override the rates for
        every model. A model not in PRICED_MODELS, or priced but not yet
        effective, uses the conservative fallback.
        """
        rates = rates_for(model, today)
        input_rate = float(os.getenv("INVESTIGATION_COST_PER_M_INPUT") or rates.input)
        output_rate = float(os.getenv("INVESTIGATION_COST_PER_M_OUTPUT") or rates.output)
        if not all(math.isfinite(rate) and rate >= 0 for rate in (input_rate, output_rate)):
            raise ValueError("Pricing rates must be finite and nonnegative")
        elevated = self.input_tokens > 272_000
        input_multiplier, output_multiplier = (2, 1.5) if elevated else (1, 1)
        self.estimated_cost_usd = (
            (input_rate * (self.ordinary_input_tokens * (1.25 if self.cache_details_missing_turns else 1) + self.cached_input_tokens * rates.cache_read_ratio + self.cache_write_input_tokens * rates.cache_write_ratio)) * input_multiplier
            + self.output_tokens * output_rate * output_multiplier
        ) / 1_000_000
        self.estimated_uncached_cost_usd = (
            self.input_tokens * input_rate * input_multiplier + self.output_tokens * output_rate * output_multiplier
        ) / 1_000_000
        self.elevated_price_turns = int(elevated)
        return self
