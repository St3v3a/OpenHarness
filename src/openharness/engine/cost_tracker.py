"""Simple usage aggregation."""

from __future__ import annotations

from openharness.api.usage import UsageSnapshot


class CostTracker:
    """Accumulate usage over the lifetime of a session."""

    def __init__(self) -> None:
        self._usage = UsageSnapshot()

    def add(self, usage: UsageSnapshot) -> None:
        """Add a usage snapshot to the running total."""
        self._usage = UsageSnapshot(**{
            name: getattr(self._usage, name) + getattr(usage, name)
            for name in UsageSnapshot.model_fields
        })

    def restore(self, usage: UsageSnapshot) -> None:
        self._usage = usage.model_copy(deep=True)

    @property
    def total(self) -> UsageSnapshot:
        """Return the aggregated usage."""
        return self._usage
