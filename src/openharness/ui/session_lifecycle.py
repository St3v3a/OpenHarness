"""Shared local/gateway logical session transitions, preserving live resources."""
from __future__ import annotations

import logging
from uuid import uuid4
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, sanitize_conversation_messages
from openharness.services.session_storage import _persistable_tool_metadata
from openharness.tasks import get_task_manager

logger = logging.getLogger(__name__)
SESSION_KEYS = frozenset({
    "read_file_state", "invoked_skills", "async_agent_state", "async_agent_tasks",
    "recent_work_log", "recent_verified_work", "task_focus_state", "compact_checkpoints",
    "compact_last", "_suppress_next_user_goal", "memory_extract_last", "memory_extract_last_error",
    "context_estimated_tokens", "compaction_trigger", "cache_diagnostics_baseline",
})


async def apply_session_intent(bundle, result, *, save, rebuild_prompt, session_key=None):
    if not (result.start_new_session or result.resume_session_id):
        return
    engine = bundle.engine
    if getattr(engine, "_busy", False):
        raise RuntimeError("Session is busy; stop the current turn before /new or /resume")
    engine._busy = True
    try:
        snapshot = None
        restored_messages = None
        restored_usage = None
        if result.resume_session_id:
            snapshot = bundle.session_backend.load_by_id(bundle.cwd, result.resume_session_id)
            if not snapshot or not snapshot.get("session_id"):
                raise ValueError("Saved session not found or has no session identity")
            if session_key is not None and snapshot.get("session_key") != session_key:
                raise ValueError("Saved session does not belong to this channel")
            restored_messages = sanitize_conversation_messages([
                ConversationMessage.model_validate(item) for item in snapshot.get("messages", [])
            ])
            restored_usage = UsageSnapshot.model_validate(snapshot.get("usage") or {})
        active_tasks = engine.tool_metadata.get("async_agent_tasks") or []
        empty = not engine.messages and not engine.total_usage.total_tokens and not engine.total_usage.estimated_cost_usd and not active_tasks
        if empty and not result.force_new_session and snapshot is None:
            result.message = f"Session already empty: {bundle.session_id}"
            result.submit_prompt = result.new_session_prompt
            return
        old_id = bundle.session_id
        old_count = len(engine.messages)
        old_estimate = engine.tool_metadata.get("context_estimated_tokens", 0)
        # A failed old snapshot never changes the active session.
        await save()
        manager = get_task_manager()
        for entry in active_tasks:
            if isinstance(entry, dict) and entry.get("task_id"):
                task = manager.get_task(str(entry["task_id"]))
                if task is not None and task.status not in {"completed", "failed", "killed"}:
                    await manager.stop_task(task.id)
        engine.clear()
        for key in SESSION_KEYS:
            engine.tool_metadata.pop(key, None)
        engine.tool_metadata["session_generation"] = int(engine.tool_metadata.get("session_generation", 0)) + 1
        bundle.session_id = str(snapshot["session_id"]) if snapshot else uuid4().hex[:12]
        engine.tool_metadata["session_id"] = bundle.session_id
        if snapshot:
            metadata = _persistable_tool_metadata(snapshot.get("tool_metadata"))
            # Never revive worker tasks from a persisted session.
            engine.tool_metadata.update({k: val for k, val in metadata.items() if k in SESSION_KEYS and k not in {"async_agent_tasks", "async_agent_state"}})
            engine.load_messages(restored_messages)
            engine.restore_usage(restored_usage)
            result.replay_messages = restored_messages
            result.message = f"Restored session: {bundle.session_id}"
        else:
            result.message = f"Started a new session: {bundle.session_id}\nPrevious session saved as: {old_id}"
            result.submit_prompt = result.new_session_prompt
        engine.set_system_prompt(rebuild_prompt())
        logger.info("session_rotated", extra={"old_session_id": old_id, "new_session_id": bundle.session_id,
                    "old_message_count": old_count, "reason": "resume" if snapshot else result.session_reason})
        try:
            from aiworks_harness.telemetry.exporter import record_session_rotation
        except ImportError:
            pass
        else:
            record_session_rotation(old_id=old_id, new_id=bundle.session_id, old_count=old_count,
                                    old_estimate=old_estimate, reason="resume" if snapshot else result.session_reason)
        try:
            await save()
        except Exception:
            logger.exception("New session snapshot failed; in-memory identity retained")
            result.message += "\nSnapshot save failed; the active session will be saved on the next successful turn."
    finally:
        engine._busy = False
