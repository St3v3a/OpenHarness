"""Tool-output context budget helpers."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# 40,000 / 8,000 since 2026-11-17 (were 16,000 / 3,000): over 30 days of
# investigations a third of knowledge-article reads (20K to 32K chars) reached
# the model as a 3,000-char preview. A result over the inline limit is still
# saved whole and previewed.
DEFAULT_TOOL_OUTPUT_INLINE_CHARS = 40_000
DEFAULT_TOOL_OUTPUT_PREVIEW_CHARS = 8_000
DEFAULT_MICROCOMPACT_TOOL_RESULT_CHARS = 4_000


def _read_positive_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        log.warning("Ignoring invalid %s=%r", name, raw)
        return default


def tool_output_inline_chars() -> int:
    return _read_positive_int_env(
        "OPENHARNESS_TOOL_OUTPUT_INLINE_CHARS",
        DEFAULT_TOOL_OUTPUT_INLINE_CHARS,
        minimum=256,
    )


def tool_output_preview_chars() -> int:
    return _read_positive_int_env(
        "OPENHARNESS_TOOL_OUTPUT_PREVIEW_CHARS",
        DEFAULT_TOOL_OUTPUT_PREVIEW_CHARS,
        minimum=128,
    )


def microcompact_tool_result_chars() -> int:
    return _read_positive_int_env(
        "OPENHARNESS_MICROCOMPACT_TOOL_RESULT_CHARS",
        DEFAULT_MICROCOMPACT_TOOL_RESULT_CHARS,
        minimum=256,
    )


def is_microcompactable_tool_result(tool_name: str, content: str) -> bool:
    """Return True when a tool result should be eligible for old-result clearing."""
    normalized = tool_name.strip()
    if normalized.startswith("mcp__"):
        return True
    return len(content) >= microcompact_tool_result_chars()
