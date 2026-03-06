"""Codex-compatible watch orchestration built on existing .claude skill scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from src.runtime.common import has_chinese, now_iso_with_tz, read_text
from src.store.database import Database


SUPPORTED_LENSES = {"deep_insight", "flash_brief", "dual_take", "timeline_trace"}

SENSOR_SCRIPT_PATHS = {
    "fetch-hacker-news": ".claude/skills/fetch-hacker-news/scripts/fetch.py",
    "fetch-github-trending": ".claude/skills/fetch-github-trending/scripts/fetch.py",
    "fetch-v2ex": ".claude/skills/fetch-v2ex/scripts/fetch.py",
    "fetch-tavily": ".claude/skills/fetch-tavily/scripts/search.py",
    "fetch-brave-search": ".claude/skills/fetch-brave-search/scripts/search.py",
    "fetch-exa": ".claude/skills/fetch-exa/scripts/search.py",
    "fetch-product-hunt": ".claude/skills/fetch-product-hunt/scripts/fetch.py",
    "fetch-request-hunt": ".claude/skills/fetch-request-hunt/scripts/search.py",
    "fetch-rss": ".claude/skills/fetch-rss/scripts/fetch.py",
    "fetch-reddit": ".claude/skills/fetch-reddit/scripts/fetch.py",
    "fetch-x": ".claude/skills/fetch-x/scripts/search.py",
    "fetch-news-api": ".claude/skills/fetch-news-api/scripts/search.py",
    "fetch-gnews": ".claude/skills/fetch-gnews/scripts/search.py",
    "fetch-arxiv": ".claude/skills/fetch-arxiv/scripts/search.py",
    "fetch-openalex": ".claude/skills/fetch-openalex/scripts/search.py",
}

DB_SAVE_ITEMS_SCRIPT = ".claude/skills/db-save-items/scripts/save.py"
DB_QUERY_SCRIPT = ".claude/skills/db-query-items/scripts/query.py"
DB_SAVE_ANALYSIS_SCRIPT = ".claude/skills/db-save-analysis/scripts/save.py"
PREPROCESS_CONTRACT_VERSION = 1
PREPROCESS_STATUS_VALID = "valid"
PREPROCESS_STATUS_IRRELEVANT = "irrelevant"
PREPROCESS_STATUS_INVALID = "invalid"
PREPROCESS_BATCH_SIZE = 8
DEFAULT_MAX_NEED_ANALYSIS_ITEMS = 24
MAX_NEED_ANALYSIS_ITEMS_CAP = 48
MAX_NEED_ANALYSIS_ITEMS_ENV = "SIGNEX_MAX_NEED_ANALYSIS_ITEMS"
MAX_NEED_CLUSTERS = 8
LEGACY_REPROCESS_LIMIT = 24
CLAUDE_TIMEOUT_SECONDS = 180
NEED_CLUSTER_SOURCE = "need_cluster"
CLAUDE_PERMISSION_MODE_BYPASS = "bypassPermissions"
INTENT_NOISE_TOKENS = (
    "discoverneeds-signex-preprocess-contract",
    "discoverneeds preprocess output contract",
    "return preprocess fields using this contract",
    "required preprocess keys",
    "output requirements",
    "example preprocess payload",
    "contract version",
    "return valid json",
    "preprocessstatus",
    "normalizedtitle",
    "normalizedcontent",
    "preprocesstags",
    "preprocessreason",
    "preprocesserror",
    "preprocessversion",
    "preprocessrunid",
    "preprocessedat",
)


@dataclass
class SensorRunResult:
    sensor: str
    success: bool
    items: list[dict[str, Any]]
    error: str = ""
    inserted: int = 0


@dataclass
class NeedInsight:
    id: int
    source: str
    title: str
    url: str
    need_summary: str
    pain_point: str
    target_user: str
    suggested_direction: str
    why_now: str
    confidence: int


@dataclass
class NeedCluster:
    cluster_id: str
    title: str
    summary: str
    pain_point: str
    target_user: str
    suggested_direction: str
    why_now: str
    confidence: int
    evidence_item_ids: list[int]
    evidence_urls: list[str]
    tags: list[str]
    item_id: int = 0
    source_id: str = ""


@dataclass
class PreprocessRunSummary:
    total_targets: int
    llm_batches: int
    success_count: int
    failure_count: int
    run_id: str = ""
    error: str = ""


@dataclass
class NeedCandidateSummary:
    total_items: int
    valid_count: int
    irrelevant_count: int
    invalid_count: int
    missing_count: int
    relevant_count: int
    selected_count: int
    selection_limit: int
    relevant_source_counts: dict[str, int]
    selected_source_counts: dict[str, int]


def _extract_json_payload(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fence_matches:
        block_text = block.strip()
        if not block_text:
            continue
        try:
            return json.loads(block_text)
        except json.JSONDecodeError:
            continue

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue

    return None


def _extract_json(raw: str) -> dict[str, Any]:
    payload = _extract_json_payload(raw)
    if isinstance(payload, dict):
        return payload
    return {}


def _run_claude_json(
    *,
    prompt: str,
    cwd: Path,
    timeout: int = CLAUDE_TIMEOUT_SECONDS,
    enable_tools: bool = False,
    permission_mode: str | None = None,
) -> tuple[dict[str, Any], str]:
    commands: list[list[str]] = []
    base_cmd = ["claude", "-p", prompt]

    if enable_tools:
        if permission_mode:
            commands.append([*base_cmd, "--permission-mode", permission_mode])
        commands.append(base_cmd)
    else:
        commands.append([*base_cmd, "--tools", ""])

    last_error = ""
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            last_error = stderr or f"claude returned non-zero exit code {proc.returncode}"
            continue

        payload = _extract_json_payload(proc.stdout)
        if not isinstance(payload, dict):
            last_error = "claude response is not valid JSON object"
            continue

        return payload, stderr

    raise RuntimeError(last_error or "claude invocation failed")


def _run_json_script(
    root_dir: Path,
    script_rel: str,
    payload: dict[str, Any] | None = None,
    args: list[str] | None = None,
    timeout: int = 180,
) -> tuple[dict[str, Any], str]:
    script_path = root_dir / script_rel
    if shutil.which("uv"):
        cmd = ["uv", "run", "python", str(script_path)] + (args or [])
    else:
        cmd = [sys.executable, str(script_path)] + (args or [])

    stdin_text = None
    if payload is not None:
        stdin_text = json.dumps(payload, ensure_ascii=False)

    proc = subprocess.run(
        cmd,
        cwd=str(root_dir),
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    data = _extract_json(proc.stdout)
    if not data:
        data = {
            "success": False,
            "items": [],
            "count": 0,
            "error": proc.stderr.strip() or "Script returned non-JSON output",
        }

    if proc.returncode != 0 and "success" not in data:
        data["success"] = False

    stderr = proc.stderr.strip()
    return data, stderr


def _require_script_success(data: dict[str, Any], stderr: str, script_rel: str) -> dict[str, Any]:
    if data.get("success") is True:
        return data

    err = data.get("error")
    if isinstance(err, str) and err.strip():
        message = err.strip()
    elif stderr:
        message = stderr
    else:
        message = "unknown script failure"

    raise RuntimeError(f"{script_rel} failed: {message}")


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _resolve_max_need_analysis_items() -> int:
    raw_value = os.getenv(MAX_NEED_ANALYSIS_ITEMS_ENV, "").strip()
    if not raw_value:
        return DEFAULT_MAX_NEED_ANALYSIS_ITEMS

    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_NEED_ANALYSIS_ITEMS

    return max(1, min(parsed, MAX_NEED_ANALYSIS_ITEMS_CAP))


def _format_source_counts(source_counts: dict[str, int]) -> str:
    return ", ".join(
        f"{key}:{value}" for key, value in sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))
    ) or "none"


def _count_items_by_source(items: list[dict[str, Any]]) -> dict[str, int]:
    source_counts: dict[str, int] = {}
    for item in items:
        source = _normalize_text(str(item.get("source") or "unknown")) or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
    return source_counts


def _summarize_need_candidates(
    items: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    *,
    selection_limit: int,
) -> NeedCandidateSummary:
    valid_count = 0
    irrelevant_count = 0
    invalid_count = 0
    missing_count = 0
    relevant_count = 0
    relevant_source_counts: dict[str, int] = {}

    for item in items:
        status = str(item.get("preprocessStatus") or "").strip().lower()
        source = _normalize_text(str(item.get("source") or "unknown")) or "unknown"
        if status == PREPROCESS_STATUS_VALID:
            valid_count += 1
            if _to_bool(item.get("isRelevant")):
                relevant_count += 1
                relevant_source_counts[source] = relevant_source_counts.get(source, 0) + 1
        elif status == PREPROCESS_STATUS_IRRELEVANT:
            irrelevant_count += 1
        elif status == PREPROCESS_STATUS_INVALID:
            invalid_count += 1
        else:
            missing_count += 1

    selected_source_counts = _count_items_by_source(selected_candidates)

    return NeedCandidateSummary(
        total_items=len(items),
        valid_count=valid_count,
        irrelevant_count=irrelevant_count,
        invalid_count=invalid_count,
        missing_count=missing_count,
        relevant_count=relevant_count,
        selected_count=len(selected_candidates),
        selection_limit=selection_limit,
        relevant_source_counts=relevant_source_counts,
        selected_source_counts=selected_source_counts,
    )


def _extract_feed_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)]+", text)
    return [u.rstrip(".,") for u in urls]


def _sanitize_watch_text(markdown_text: str) -> str:
    if not markdown_text:
        return ""

    kept_lines: list[str] = []
    in_code_block = False
    in_comment_block = False

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        lower = stripped.lower()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        if in_comment_block:
            if "-->" in lower:
                in_comment_block = False
            continue
        if "<!--" in lower:
            if "-->" not in lower:
                in_comment_block = True
            continue

        if any(token in lower for token in INTENT_NOISE_TOKENS):
            continue
        kept_lines.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    if cleaned:
        return cleaned

    for raw_line in markdown_text.splitlines():
        fallback = raw_line.strip()
        if fallback and not fallback.startswith("```") and not fallback.startswith("<!--"):
            return fallback
    return ""


def _safe_words_from_markdown(markdown_text: str) -> list[str]:
    markdown_text = _sanitize_watch_text(markdown_text)
    if not markdown_text:
        return []

    phrases: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        line = re.sub(r"^\*\*|\*\*$", "", line)
        line = re.sub(r"\([^)]*\)", "", line).strip()
        if not line:
            continue
        if line.lower().startswith(("role:", "domain:", "report language:", "focus:")):
            continue
        phrases.append(line)
    return phrases


def _truncate_text(value: str | None, limit: int = 1200) -> str:
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def _is_legacy_preprocess(item: dict[str, Any]) -> bool:
    run_id = str(item.get("preprocessRunId") or "").strip()
    reason = str(item.get("preprocessReason") or "").strip().lower()
    if run_id.startswith("watch-run-"):
        return True
    return reason.startswith("matched keywords") or "heuristic" in reason or reason.startswith("no watch-intent signal")


def _invalid_preprocess_payload(
    item: dict[str, Any],
    *,
    reason: str,
    error: str,
    preprocess_run_id: str,
    preprocessed_at: str,
) -> dict[str, Any]:
    source = _normalize_text(str(item.get("source") or ""))
    raw_title = _truncate_text(str(item.get("title") or ""), 220)
    tags = [source] if source else []
    return {
        "status": PREPROCESS_STATUS_INVALID,
        "isRelevant": False,
        "normalizedTitle": raw_title or None,
        "normalizedContent": None,
        "preprocessTags": tags or None,
        "preprocessReason": reason,
        "preprocessError": error,
        "preprocessRunId": preprocess_run_id,
        "preprocessedAt": preprocessed_at,
    }


def _build_preprocess_prompt(
    *,
    watch_name: str,
    intent_text: str,
    memory_text: str,
    language_code: str,
    batch: list[dict[str, Any]],
) -> str:
    compact_items: list[dict[str, Any]] = []
    for item in batch:
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        compact_items.append(
            {
                "id": item_id,
                "source": _normalize_text(str(item.get("source") or "")),
                "title": _truncate_text(str(item.get("title") or ""), 200),
                "content": _truncate_text(str(item.get("content") or ""), 420),
                "url": _truncate_text(str(item.get("url") or ""), 240),
                "published_at": str(item.get("published_at") or ""),
                "fetched_at": str(item.get("fetched_at") or ""),
            }
        )

    payload = {
        "watch_name": watch_name,
        "report_language": "Chinese" if language_code == "zh" else "English",
        "intent": _truncate_text(intent_text, 1200),
        "memory": _truncate_text(memory_text, 800),
        "items": compact_items,
    }

    return (
        "You are preprocessing raw watch data into structured product-need candidates.\n"
        "Return JSON only, with the exact shape: {\"items\": [...]}.\n"
        "Rules:\n"
        "1) Every input id must appear exactly once.\n"
        "2) preprocessStatus must be one of: valid, irrelevant, invalid.\n"
        "3) valid means the item has concrete user pain/feature demand aligned with watch intent.\n"
        "4) irrelevant means not useful for watch intent, isRelevant=false.\n"
        "5) invalid means malformed/unusable input, and preprocessError must explain why.\n"
        "6) normalizedTitle and normalizedContent must be rewritten concise summaries, not raw copy.\n"
        "7) normalizedContent should capture user scenario, pain point, and potential product direction in 2-4 sentences.\n"
        "8) preprocessTags is an array of short tags.\n"
        "9) Keep language consistent with report_language.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Output JSON only. No markdown, no explanation."
    )


def _normalize_preprocess_output(
    *,
    item: dict[str, Any],
    llm_entry: dict[str, Any] | None,
    preprocess_run_id: str,
    preprocessed_at: str,
) -> dict[str, Any]:
    if not isinstance(llm_entry, dict):
        return _invalid_preprocess_payload(
            item,
            reason="LLM output missing item",
            error="Missing preprocess payload for item id",
            preprocess_run_id=preprocess_run_id,
            preprocessed_at=preprocessed_at,
        )

    status = str(llm_entry.get("preprocessStatus") or llm_entry.get("status") or "").strip().lower()
    if status not in {PREPROCESS_STATUS_VALID, PREPROCESS_STATUS_IRRELEVANT, PREPROCESS_STATUS_INVALID}:
        status = PREPROCESS_STATUS_INVALID

    is_relevant = _to_bool(llm_entry.get("isRelevant"))
    if status == PREPROCESS_STATUS_VALID:
        is_relevant = True
    elif status in {PREPROCESS_STATUS_IRRELEVANT, PREPROCESS_STATUS_INVALID}:
        is_relevant = False

    normalized_title = _truncate_text(
        str(llm_entry.get("normalizedTitle") or llm_entry.get("title") or ""),
        280,
    )
    normalized_content = _truncate_text(
        str(llm_entry.get("normalizedContent") or llm_entry.get("summary") or ""),
        1200,
    )

    if status == PREPROCESS_STATUS_VALID and not normalized_title:
        normalized_title = _truncate_text(str(item.get("title") or ""), 280)
    if status == PREPROCESS_STATUS_VALID and not normalized_content:
        normalized_content = _truncate_text(str(item.get("content") or ""), 1200)

    raw_tags = llm_entry.get("preprocessTags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for raw in raw_tags:
            tag = _normalize_text(str(raw))
            if tag and tag not in tags:
                tags.append(tag)
    source = _normalize_text(str(item.get("source") or ""))
    if source and source not in tags:
        tags.insert(0, source)
    tags = tags[:6]

    reason = _truncate_text(str(llm_entry.get("preprocessReason") or ""), 300)
    if not reason:
        if status == PREPROCESS_STATUS_VALID:
            reason = "Aligned with watch intent and contains actionable need signal"
        elif status == PREPROCESS_STATUS_IRRELEVANT:
            reason = "Not aligned with watch intent"
        else:
            reason = "Preprocess normalization failed"

    error = _truncate_text(str(llm_entry.get("preprocessError") or ""), 320)
    if status == PREPROCESS_STATUS_INVALID and not error:
        error = "Invalid preprocess output"
    if status != PREPROCESS_STATUS_INVALID:
        error = None

    return {
        "status": status,
        "isRelevant": is_relevant,
        "normalizedTitle": normalized_title or None,
        "normalizedContent": normalized_content or None,
        "preprocessTags": tags or None,
        "preprocessReason": reason,
        "preprocessError": error,
        "preprocessRunId": preprocess_run_id,
        "preprocessedAt": preprocessed_at,
    }


def _persist_preprocess_payload(db: Database, item_id: int, payload: dict[str, Any]) -> None:
    db.update_preprocess_fields(
        item_id,
        preprocess_status=str(payload["status"]),
        is_relevant=bool(payload["isRelevant"]),
        normalized_title=payload["normalizedTitle"],
        normalized_content=payload["normalizedContent"],
        preprocess_tags=payload["preprocessTags"],
        preprocess_reason=payload["preprocessReason"],
        preprocess_error=payload["preprocessError"],
        preprocess_version=PREPROCESS_CONTRACT_VERSION,
        preprocess_run_id=str(payload["preprocessRunId"]),
        preprocessed_at=str(payload["preprocessedAt"]),
    )


def _apply_preprocess_to_items(
    root_dir: Path,
    *,
    watch_name: str,
    intent_text: str,
    memory_text: str,
    language_code: str,
    now: datetime,
) -> PreprocessRunSummary:
    db = Database(str(root_dir / "data/signex.db"))
    db.init()
    try:
        all_items = db.get_items()
        missing_items: list[dict[str, Any]] = []
        legacy_items: list[dict[str, Any]] = []
        for item in all_items:
            if _normalize_text(str(item.get("source") or "")) == NEED_CLUSTER_SOURCE:
                continue
            status = str(item.get("preprocessStatus") or "").strip().lower()
            if not status or status == "missing":
                missing_items.append(item)
                continue
            if _is_legacy_preprocess(item):
                legacy_items.append(item)

        target_items = missing_items + legacy_items[:LEGACY_REPROCESS_LIMIT]
        if not target_items:
            return PreprocessRunSummary(
                total_targets=0,
                llm_batches=0,
                success_count=0,
                failure_count=0,
                run_id="",
            )

        preprocess_run_id = f"watch-llm-{now_iso_with_tz(now)}"
        preprocessed_at = now_iso_with_tz(now)
        llm_batches = _chunks(target_items, PREPROCESS_BATCH_SIZE)
        success_count = 0
        failure_count = 0
        batch_errors: list[str] = []

        for batch in llm_batches:
            prompt = _build_preprocess_prompt(
                watch_name=watch_name,
                intent_text=intent_text,
                memory_text=memory_text,
                language_code=language_code,
                batch=batch,
            )
            llm_by_id: dict[int, dict[str, Any]] = {}
            try:
                try:
                    llm_payload, _ = _run_claude_json(prompt=prompt, cwd=root_dir, timeout=CLAUDE_TIMEOUT_SECONDS)
                except RuntimeError as exc:  # pragma: no cover - depends on external claude runtime
                    if "not valid JSON object" not in str(exc):
                        raise
                    retry_prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Reply with JSON object only. Do not add markdown code fences or prose."
                    )
                    llm_payload, _ = _run_claude_json(prompt=retry_prompt, cwd=root_dir, timeout=CLAUDE_TIMEOUT_SECONDS)

                llm_items = llm_payload.get("items")
                llm_entries = llm_items if isinstance(llm_items, list) else []
                for entry in llm_entries:
                    if not isinstance(entry, dict):
                        continue
                    entry_id = entry.get("id")
                    if isinstance(entry_id, int):
                        llm_by_id[entry_id] = entry
            except Exception as exc:  # pragma: no cover - depends on external claude runtime
                batch_errors.append(str(exc))
                llm_by_id = {}

            for item in batch:
                item_id = item.get("id")
                if not isinstance(item_id, int):
                    continue
                entry = llm_by_id.get(item_id)
                if entry is None and llm_by_id:
                    payload = _invalid_preprocess_payload(
                        item,
                        reason="LLM output missing item",
                        error="Missing preprocess payload for item id",
                        preprocess_run_id=preprocess_run_id,
                        preprocessed_at=preprocessed_at,
                    )
                elif entry is None:
                    payload = _invalid_preprocess_payload(
                        item,
                        reason="LLM preprocess failed",
                        error=batch_errors[-1] if batch_errors else "Unknown preprocess failure",
                        preprocess_run_id=preprocess_run_id,
                        preprocessed_at=preprocessed_at,
                    )
                else:
                    payload = _normalize_preprocess_output(
                        item=item,
                        llm_entry=entry,
                        preprocess_run_id=preprocess_run_id,
                        preprocessed_at=preprocessed_at,
                    )

                _persist_preprocess_payload(db, item_id, payload)
                if payload["status"] == PREPROCESS_STATUS_INVALID:
                    failure_count += 1
                else:
                    success_count += 1

        return PreprocessRunSummary(
            total_targets=len(target_items),
            llm_batches=len(llm_batches),
            success_count=success_count,
            failure_count=failure_count,
            run_id=preprocess_run_id,
            error="; ".join(batch_errors[:3]),
        )
    finally:
        db.close()


def _select_need_candidates(items: list[dict[str, Any]], max_items: int | None = None) -> list[dict[str, Any]]:
    resolved_max_items = _resolve_max_need_analysis_items() if max_items is None else max(1, min(max_items, MAX_NEED_ANALYSIS_ITEMS_CAP))
    candidates: list[dict[str, Any]] = []
    for item in items:
        source = _normalize_text(str(item.get("source") or ""))
        if source == NEED_CLUSTER_SOURCE:
            continue
        status = str(item.get("preprocessStatus") or "").strip().lower()
        if status != PREPROCESS_STATUS_VALID:
            continue
        if not _to_bool(item.get("isRelevant")):
            continue
        title = _normalize_text(str(item.get("normalizedTitle") or item.get("title") or ""))
        content = _normalize_text(str(item.get("normalizedContent") or item.get("content") or ""))
        if not title and not content:
            continue
        candidates.append(item)

    ordered = _sorted_items(candidates)
    if len(ordered) <= resolved_max_items:
        return ordered

    candidates_by_source: dict[str, list[dict[str, Any]]] = {}
    for item in ordered:
        source = _normalize_text(str(item.get("source") or "unknown")) or "unknown"
        candidates_by_source.setdefault(source, []).append(item)

    source_order = sorted(
        candidates_by_source,
        key=lambda source: (-len(candidates_by_source[source]), source),
    )

    selected: list[dict[str, Any]] = []
    while len(selected) < resolved_max_items:
        progressed = False
        for source in source_order:
            source_bucket = candidates_by_source[source]
            if not source_bucket:
                continue
            selected.append(source_bucket.pop(0))
            progressed = True
            if len(selected) >= resolved_max_items:
                break
        if not progressed:
            break

    return selected


def _load_candidate_fallback_pool(
    root_dir: Path,
    *,
    preprocess_run_id: str,
) -> list[dict[str, Any]]:
    db = Database(str(root_dir / "data/signex.db"))
    db.init()
    try:
        items = db.get_items()
    finally:
        db.close()

    if preprocess_run_id:
        current_run_items = [item for item in items if str(item.get("preprocessRunId") or "") == preprocess_run_id]
        run_candidates = _select_need_candidates(current_run_items)
        if run_candidates:
            return run_candidates

    return _select_need_candidates(items)


def _build_need_clustering_prompt(
    *,
    watch_name: str,
    intent_text: str,
    memory_text: str,
    language_code: str,
    candidates: list[dict[str, Any]],
    max_clusters: int = MAX_NEED_CLUSTERS,
) -> str:
    compact_candidates: list[dict[str, Any]] = []
    for item in candidates:
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        compact_candidates.append(
            {
                "id": item_id,
                "source": _normalize_text(str(item.get("source") or "")),
                "title": _truncate_text(str(item.get("normalizedTitle") or item.get("title") or ""), 220),
                "content": _truncate_text(str(item.get("normalizedContent") or item.get("content") or ""), 420),
                "url": _truncate_text(str(item.get("url") or ""), 240),
                "published_at": str(item.get("published_at") or ""),
                "fetched_at": str(item.get("fetched_at") or ""),
            }
        )

    payload = {
        "watch_name": watch_name,
        "report_language": "Chinese" if language_code == "zh" else "English",
        "intent": _truncate_text(intent_text, 1200),
        "memory": _truncate_text(memory_text, 900),
        "max_clusters": max(1, min(max_clusters, MAX_NEED_CLUSTERS)),
        "candidates": compact_candidates,
    }

    return (
        "You are clustering candidate user-need signals into actionable product need themes.\n"
        "Return JSON only, exact shape: {\"clusters\":[...]}\n"
        "Each cluster object must contain:\n"
        "{\"clusterId\":\"\",\"needTitle\":\"\",\"needSummary\":\"\",\"painPoint\":\"\",\"targetUser\":\"\","
        "\"suggestedDirection\":\"\",\"whyNow\":\"\",\"confidence\":0,\"evidenceItemIds\":[],\"evidenceUrls\":[],\"tags\":[]}\n"
        "Rules:\n"
        "1) Group candidates by the same underlying need; do not output one cluster per item unless evidence is truly isolated.\n"
        "2) evidenceItemIds must reference input ids and be unique.\n"
        "3) needSummary, painPoint, and suggestedDirection must be synthesized, not copied raw text.\n"
        "4) confidence is integer 0-100.\n"
        "5) Output at most max_clusters clusters.\n"
        "6) Keep language consistent with report_language.\n"
        "7) Prefer concrete product demand themes with clear implementation direction.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Output JSON only. No markdown, no explanation."
    )


def _normalize_cluster_id(raw_value: Any, index: int) -> str:
    text = _normalize_text(str(raw_value or ""))
    if not text:
        text = f"cluster-{index + 1}"
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
    return token or f"cluster-{index + 1}"


def _normalize_need_cluster(
    *,
    entry: dict[str, Any],
    index: int,
    candidate_by_id: dict[int, dict[str, Any]],
    language_code: str,
) -> NeedCluster | None:
    evidence_ids: list[int] = []
    raw_ids = entry.get("evidenceItemIds")
    if isinstance(raw_ids, list):
        for raw_id in raw_ids:
            if isinstance(raw_id, int) and raw_id in candidate_by_id and raw_id not in evidence_ids:
                evidence_ids.append(raw_id)
            elif isinstance(raw_id, str) and raw_id.isdigit():
                parsed_id = int(raw_id)
                if parsed_id in candidate_by_id and parsed_id not in evidence_ids:
                    evidence_ids.append(parsed_id)

    if not evidence_ids:
        raw_primary = entry.get("primaryItemId")
        if isinstance(raw_primary, int) and raw_primary in candidate_by_id:
            evidence_ids.append(raw_primary)

    if not evidence_ids:
        return None

    raw_urls = entry.get("evidenceUrls")
    evidence_urls: list[str] = []
    if isinstance(raw_urls, list):
        for raw in raw_urls:
            url = _truncate_text(str(raw), 260)
            if url and url not in evidence_urls:
                evidence_urls.append(url)
    for evidence_id in evidence_ids:
        candidate_url = _truncate_text(str(candidate_by_id[evidence_id].get("url") or ""), 260)
        if candidate_url and candidate_url not in evidence_urls:
            evidence_urls.append(candidate_url)

    title = _truncate_text(str(entry.get("needTitle") or entry.get("title") or ""), 180)
    summary = _truncate_text(str(entry.get("needSummary") or entry.get("summary") or ""), 420)
    pain_point = _truncate_text(str(entry.get("painPoint") or ""), 280)
    target_user = _truncate_text(str(entry.get("targetUser") or ""), 180)
    suggested_direction = _truncate_text(str(entry.get("suggestedDirection") or ""), 260)
    why_now = _truncate_text(str(entry.get("whyNow") or ""), 200)

    if not title:
        first_item = candidate_by_id[evidence_ids[0]]
        title = _truncate_text(str(first_item.get("normalizedTitle") or first_item.get("title") or ""), 180)
    if not summary:
        first_item = candidate_by_id[evidence_ids[0]]
        summary = _truncate_text(str(first_item.get("normalizedContent") or first_item.get("content") or ""), 420)
    if not pain_point:
        pain_point = "Pain point exists but needs refinement" if language_code != "zh" else "存在明确痛点，但需进一步细化"
    if not target_user:
        target_user = "Users reflected in evidence channels" if language_code != "zh" else "证据来源中反复出现的目标用户"
    if not suggested_direction:
        suggested_direction = (
            "Convert this need into a scoped product experiment"
            if language_code != "zh"
            else "将该需求转化为可验证的产品实验"
        )
    if not why_now:
        why_now = "Recent repeated signal in monitored channels" if language_code != "zh" else "该需求在近期监控渠道中反复出现"

    raw_confidence = entry.get("confidence")
    if isinstance(raw_confidence, float):
        confidence = int(round(raw_confidence))
    elif isinstance(raw_confidence, int):
        confidence = raw_confidence
    elif isinstance(raw_confidence, str) and raw_confidence.strip().isdigit():
        confidence = int(raw_confidence.strip())
    else:
        confidence = 55
    confidence = max(0, min(100, confidence))

    raw_tags = entry.get("tags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for raw in raw_tags:
            tag = _truncate_text(_normalize_text(str(raw)), 40)
            if tag and tag not in tags:
                tags.append(tag)
    tags = tags[:8]

    return NeedCluster(
        cluster_id=_normalize_cluster_id(entry.get("clusterId"), index),
        title=title or f"Need cluster {index + 1}",
        summary=summary or (title or f"Need cluster {index + 1}"),
        pain_point=pain_point,
        target_user=target_user,
        suggested_direction=suggested_direction,
        why_now=why_now,
        confidence=confidence,
        evidence_item_ids=evidence_ids,
        evidence_urls=evidence_urls[:8],
        tags=tags,
    )


def _fallback_need_clusters(candidates: list[dict[str, Any]], language_code: str) -> list[NeedCluster]:
    fallback_clusters: list[NeedCluster] = []
    for index, item in enumerate(candidates[: min(3, len(candidates))]):
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        title = _truncate_text(str(item.get("normalizedTitle") or item.get("title") or ""), 180) or f"Need cluster {index + 1}"
        summary = _truncate_text(str(item.get("normalizedContent") or item.get("content") or ""), 420) or title
        url = _truncate_text(str(item.get("url") or ""), 260)
        fallback_clusters.append(
            NeedCluster(
                cluster_id=f"fallback-{index + 1}",
                title=title,
                summary=summary,
                pain_point="Signal extracted from candidate evidence" if language_code != "zh" else "该候选证据中存在可执行痛点信号",
                target_user="Users represented by this item" if language_code != "zh" else "该条证据中反映的目标用户",
                suggested_direction=(
                    "Validate this need with a focused prototype and user interview"
                    if language_code != "zh"
                    else "用小范围原型与用户访谈验证该需求"
                ),
                why_now=(
                    "Detected in latest watch cycle"
                    if language_code != "zh"
                    else "在最新一轮监控中出现"
                ),
                confidence=45,
                evidence_item_ids=[item_id],
                evidence_urls=[url] if url else [],
                tags=[_normalize_text(str(item.get("source") or ""))] if item.get("source") else [],
            )
        )
    return fallback_clusters


def _cluster_need_candidates(
    *,
    root_dir: Path,
    watch_name: str,
    intent_text: str,
    memory_text: str,
    language_code: str,
    candidates: list[dict[str, Any]],
) -> list[NeedCluster]:
    if not candidates:
        return []

    candidate_by_id = {
        int(item["id"]): item
        for item in candidates
        if isinstance(item.get("id"), int)
    }
    if not candidate_by_id:
        return []

    prompt = _build_need_clustering_prompt(
        watch_name=watch_name,
        intent_text=intent_text,
        memory_text=memory_text,
        language_code=language_code,
        candidates=candidates,
    )
    payload: dict[str, Any] = {}
    try:
        payload, _ = _run_claude_json(
            prompt=prompt,
            cwd=root_dir,
            timeout=CLAUDE_TIMEOUT_SECONDS,
            enable_tools=True,
            permission_mode=CLAUDE_PERMISSION_MODE_BYPASS,
        )
    except Exception:  # pragma: no cover - depends on external claude runtime
        return _fallback_need_clusters(candidates, language_code)

    raw_clusters = payload.get("clusters")
    entries = raw_clusters if isinstance(raw_clusters, list) else []
    normalized: list[NeedCluster] = []
    used_cluster_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        cluster = _normalize_need_cluster(
            entry=entry,
            index=index,
            candidate_by_id=candidate_by_id,
            language_code=language_code,
        )
        if cluster is None:
            continue
        if cluster.cluster_id in used_cluster_ids:
            cluster.cluster_id = f"{cluster.cluster_id}-{index + 1}"
        used_cluster_ids.add(cluster.cluster_id)
        normalized.append(cluster)

    if not normalized:
        return _fallback_need_clusters(candidates, language_code)

    normalized.sort(key=lambda item: (item.confidence, len(item.evidence_item_ids)), reverse=True)
    return normalized[:MAX_NEED_CLUSTERS]


def _sanitize_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return token or fallback


def _build_cluster_content(cluster: NeedCluster, language_code: str) -> str:
    if language_code == "zh":
        lines = [
            f"需求摘要：{cluster.summary}",
            f"核心痛点：{cluster.pain_point}",
            f"目标用户：{cluster.target_user}",
            f"产品方向：{cluster.suggested_direction}",
            f"时机判断：{cluster.why_now}",
            f"证据条数：{len(cluster.evidence_item_ids)}",
            f"置信度：{cluster.confidence}",
        ]
    else:
        lines = [
            f"Need summary: {cluster.summary}",
            f"Pain point: {cluster.pain_point}",
            f"Target user: {cluster.target_user}",
            f"Suggested direction: {cluster.suggested_direction}",
            f"Why now: {cluster.why_now}",
            f"Evidence count: {len(cluster.evidence_item_ids)}",
            f"Confidence: {cluster.confidence}",
        ]
    return "\n".join(lines)


def _persist_need_clusters(
    *,
    root_dir: Path,
    watch_name: str,
    language_code: str,
    preprocess_run_id: str,
    preprocessed_at: str,
    clusters: list[NeedCluster],
    candidates: list[dict[str, Any]],
) -> list[NeedCluster]:
    if not clusters:
        return []

    candidate_by_id = {
        int(item["id"]): item
        for item in candidates
        if isinstance(item.get("id"), int)
    }
    watch_token = _sanitize_token(watch_name, "watch")
    run_token = _sanitize_token(preprocess_run_id, "run")

    db = Database(str(root_dir / "data/signex.db"))
    db.init()
    try:
        persisted: list[NeedCluster] = []
        for index, cluster in enumerate(clusters):
            cluster_token = _sanitize_token(cluster.cluster_id, f"cluster-{index + 1}")
            source_id = f"{watch_token}:{run_token}:{cluster_token}"

            evidence_items = [candidate_by_id[item_id] for item_id in cluster.evidence_item_ids if item_id in candidate_by_id]
            source_counts: dict[str, int] = {}
            published_at = ""
            for evidence in evidence_items:
                source_name = _normalize_text(str(evidence.get("source") or "unknown"))
                source_counts[source_name or "unknown"] = source_counts.get(source_name or "unknown", 0) + 1
                if not published_at:
                    published_at = str(evidence.get("published_at") or "")

            if not published_at:
                published_at = preprocessed_at

            metadata = {
                "kind": NEED_CLUSTER_SOURCE,
                "watchName": watch_name,
                "clusterId": cluster.cluster_id,
                "confidence": cluster.confidence,
                "evidenceItemIds": cluster.evidence_item_ids,
                "evidenceUrls": cluster.evidence_urls,
                "evidenceCount": len(cluster.evidence_item_ids),
                "sourceBreakdown": source_counts,
                "tags": cluster.tags,
            }

            item_id = db.upsert_item_with_preprocess(
                source=NEED_CLUSTER_SOURCE,
                source_id=source_id,
                title=cluster.title,
                url=cluster.evidence_urls[0] if cluster.evidence_urls else None,
                content=_build_cluster_content(cluster, language_code),
                metadata=metadata,
                fetched_at=preprocessed_at,
                published_at=published_at,
                preprocess_status=PREPROCESS_STATUS_VALID,
                is_relevant=True,
                normalized_title=cluster.title,
                normalized_content=cluster.summary,
                preprocess_tags=[*cluster.tags[:6], NEED_CLUSTER_SOURCE] if cluster.tags else [NEED_CLUSTER_SOURCE],
                preprocess_reason=(
                    "Generated from clustered valid candidates"
                    if language_code != "zh"
                    else "由有效候选信号聚类生成"
                ),
                preprocess_error=None,
                preprocess_version=PREPROCESS_CONTRACT_VERSION,
                preprocess_run_id=preprocess_run_id,
                preprocessed_at=preprocessed_at,
            )
            cluster.item_id = item_id
            cluster.source_id = source_id
            persisted.append(cluster)

        return persisted
    finally:
        db.close()


def _build_need_analysis_prompt(
    *,
    watch_name: str,
    intent_text: str,
    memory_text: str,
    language_code: str,
    item: dict[str, Any],
) -> str:
    payload = {
        "watch_name": watch_name,
        "report_language": "Chinese" if language_code == "zh" else "English",
        "intent": _truncate_text(intent_text, 1200),
        "memory": _truncate_text(memory_text, 800),
        "item": {
            "id": item.get("id"),
            "source": _normalize_text(str(item.get("source") or "")),
            "url": _truncate_text(str(item.get("url") or ""), 240),
            "title": _truncate_text(str(item.get("normalizedTitle") or item.get("title") or ""), 320),
            "content": _truncate_text(str(item.get("normalizedContent") or item.get("content") or ""), 700),
            "published_at": str(item.get("published_at") or ""),
            "fetched_at": str(item.get("fetched_at") or ""),
        },
    }

    return (
        "You are an analyst extracting one actionable product need from one candidate item.\n"
        "Return JSON only with keys:\n"
        "{\"needSummary\":\"\",\"painPoint\":\"\",\"targetUser\":\"\",\"suggestedDirection\":\"\",\"whyNow\":\"\",\"confidence\":0}\n"
        "Rules:\n"
        "1) Do not copy raw content verbatim; synthesize concise insight.\n"
        "2) confidence is integer 0-100.\n"
        "3) Keep language consistent with report_language.\n"
        "4) suggestedDirection must be concrete enough for product planning.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Output JSON only. No markdown, no explanation."
    )


def _fallback_need_insight(item: dict[str, Any]) -> NeedInsight:
    title = _normalize_text(str(item.get("normalizedTitle") or item.get("title") or "(untitled)"))
    source = _normalize_text(str(item.get("source") or "unknown"))
    url = _normalize_text(str(item.get("url") or ""))
    content = _truncate_text(str(item.get("normalizedContent") or item.get("content") or ""), 180)
    return NeedInsight(
        id=int(item.get("id") or 0),
        source=source or "unknown",
        title=title or "(untitled)",
        url=url,
        need_summary=title or "Need signal detected",
        pain_point=content or "Pain point signal exists but needs manual review",
        target_user="Users mentioned in source context",
        suggested_direction="Convert this signal into a scoped product experiment",
        why_now="Recent signal from monitored channels",
        confidence=45,
    )


def _analyze_need_item(
    *,
    root_dir: Path,
    watch_name: str,
    intent_text: str,
    memory_text: str,
    language_code: str,
    item: dict[str, Any],
) -> NeedInsight:
    fallback = _fallback_need_insight(item)
    try:
        payload, _ = _run_claude_json(
            prompt=_build_need_analysis_prompt(
                watch_name=watch_name,
                intent_text=intent_text,
                memory_text=memory_text,
                language_code=language_code,
                item=item,
            ),
            cwd=root_dir,
            timeout=CLAUDE_TIMEOUT_SECONDS,
        )
    except Exception:  # pragma: no cover - depends on external claude runtime
        return fallback

    summary = _truncate_text(str(payload.get("needSummary") or ""), 220)
    pain = _truncate_text(str(payload.get("painPoint") or ""), 280)
    target_user = _truncate_text(str(payload.get("targetUser") or ""), 160)
    direction = _truncate_text(str(payload.get("suggestedDirection") or ""), 220)
    why_now = _truncate_text(str(payload.get("whyNow") or ""), 180)
    confidence = payload.get("confidence")
    if isinstance(confidence, float):
        confidence = int(round(confidence))
    if not isinstance(confidence, int):
        confidence = fallback.confidence
    confidence = max(0, min(100, confidence))

    return NeedInsight(
        id=fallback.id,
        source=fallback.source,
        title=fallback.title,
        url=fallback.url,
        need_summary=summary or fallback.need_summary,
        pain_point=pain or fallback.pain_point,
        target_user=target_user or fallback.target_user,
        suggested_direction=direction or fallback.suggested_direction,
        why_now=why_now or fallback.why_now,
        confidence=confidence,
    )


def generate_search_queries(
    watch_name: str,
    intent_text: str,
    memory_text: str,
    now: datetime | None = None,
    max_queries: int = 6,
) -> list[str]:
    """Generate date-scoped queries for search-style sensors."""
    clock = now or datetime.now().astimezone()
    month_tag = clock.strftime("%Y-%m")

    candidate_phrases: list[str] = []
    candidate_phrases.extend(_safe_words_from_markdown(intent_text))

    # memory may contain strict preferences worth carrying to searches
    for phrase in _safe_words_from_markdown(memory_text):
        if any(token in phrase.lower() for token in ["focus", "关注", "偏好", "prefer", "track"]):
            candidate_phrases.append(phrase)

    candidate_phrases.append(watch_name.replace("-", " "))

    excludes: set[str] = set()
    for line in _safe_words_from_markdown(memory_text + "\n" + intent_text):
        if any(token in line.lower() for token in ["exclude", "排除", "不要", "ignore"]):
            for token in re.split(r"[,，/;；\s]+", line):
                if len(token) >= 2:
                    excludes.add(token.lower())

    dedup: list[str] = []
    for phrase in candidate_phrases:
        compact = re.sub(r"\s+", " ", phrase).strip(" :-")
        if not compact:
            continue
        lower = compact.lower()
        if any(x in lower for x in excludes):
            continue
        if lower in [q.lower() for q in dedup]:
            continue
        dedup.append(compact)

    queries: list[str] = []
    for phrase in dedup:
        queries.append(f"{phrase} {month_tag}")
        if has_chinese(phrase):
            queries.append(f"{phrase} 最新 {month_tag}")

    if not queries:
        queries = [f"{watch_name.replace('-', ' ')} {month_tag}"]

    return queries[:max_queries]


def select_sensors(intent_text: str, memory_text: str, max_sensors: int = 6) -> list[str]:
    """Choose the minimal relevant sensor set based on watch context."""
    text = f"{intent_text}\n{memory_text}".lower()
    sensors: list[str] = []

    def add(sensor: str) -> None:
        if sensor in SENSOR_SCRIPT_PATHS and sensor not in sensors:
            sensors.append(sensor)

    add("fetch-hacker-news")
    add("fetch-tavily")

    if any(k in text for k in ["github", "开源", "open source", "repo", "仓库"]):
        add("fetch-github-trending")

    if has_chinese(text) or any(k in text for k in ["中文", "国内", "v2ex"]):
        add("fetch-v2ex")

    if any(k in text for k in ["房产", "住房", "楼市", "租房", "买房", "housing", "real estate", "estate"]):
        add("fetch-news-api")
        add("fetch-gnews")
        add("fetch-reddit")

    if any(k in text for k in ["search", "趋势", "latest", "追踪", "watch", "monitor"]):
        add("fetch-tavily")

    if any(k in text for k in ["reddit", "社区", "讨论", "用户反馈", "pain point", "论坛"]):
        add("fetch-reddit")

    if any(k in text for k in ["新闻", "news", "industry", "行业", "媒体"]):
        add("fetch-news-api")
        add("fetch-gnews")

    if any(k in text for k in ["product", "launch", "新品", "发布", "app"]):
        add("fetch-product-hunt")

    if any(k in text for k in ["request", "需求", "痛点", "feature"]):
        add("fetch-request-hunt")

    if any(k in text for k in ["x", "twitter", "社交", "实时"]):
        add("fetch-x")

    if any(k in text for k in ["rss", "博客", "blog", "changelog"]):
        add("fetch-rss")

    if any(k in text for k in ["paper", "论文", "research", "学术", "preprint"]):
        add("fetch-arxiv")
        add("fetch-openalex")

    if len(sensors) < 3:
        add("fetch-github-trending")
        add("fetch-tavily")

    return sensors[:max_sensors]


def infer_lens(memory_text: str, override: str | None = None) -> str:
    if override in SUPPORTED_LENSES:
        return override

    memo = memory_text.lower()
    if any(k in memo for k in ["flash", "速览", "quick brief"]):
        return "flash_brief"
    if any(k in memo for k in ["dual", "正反", "利弊"]):
        return "dual_take"
    if any(k in memo for k in ["timeline", "时间线", "脉络"]):
        return "timeline_trace"
    return "deep_insight"


def _language_code(intent_text: str, identity_text: str) -> str:
    lower = identity_text.lower()
    if "report language" in lower and ("chinese" in lower or "中文" in lower):
        return "zh"
    if has_chinese(intent_text):
        return "zh"
    return "en"


def _academic_queries(queries: list[str]) -> list[str]:
    academic = []
    for q in queries:
        if any(k in q.lower() for k in ["llm", "agent", "model", "code", "paper", "research", "ai"]):
            academic.append(q)
    if not academic:
        academic = ["large language model agent", "code generation LLM"]
    return academic[:3]


def _default_subreddits(intent_text: str) -> list[str]:
    text = intent_text.lower()
    if any(k in text for k in ["startup", "创业", "saas"]):
        return ["startups", "SaaS", "entrepreneur"]
    if any(k in text for k in ["ai", "llm", "machine learning", "人工智能"]):
        return ["MachineLearning", "LocalLLaMA", "artificial"]
    return ["programming", "technology", "opensource"]


def _sensor_payload(
    sensor: str,
    queries: list[str],
    intent_text: str,
    identity_text: str,
    now: datetime,
) -> tuple[dict[str, Any] | None, list[str] | None]:
    lang = _language_code(intent_text, identity_text)

    if sensor in {"fetch-hacker-news", "fetch-github-trending", "fetch-v2ex"}:
        return None, None

    if sensor == "fetch-product-hunt":
        return None, ["--limit", "20", "--featured"]

    if sensor == "fetch-rss":
        feeds = _extract_feed_urls(intent_text)
        return {"feeds": feeds, "max_per_feed": 20}, None

    if sensor == "fetch-reddit":
        return {"subreddits": _default_subreddits(intent_text), "sort": "hot", "limit": 25}, None

    if sensor == "fetch-tavily":
        return {"queries": queries[:4], "days": 7}, None

    if sensor == "fetch-brave-search":
        return {"queries": queries[:2], "count": 10}, None

    if sensor == "fetch-exa":
        return {"queries": queries[:4], "num_results": 10, "days": 7}, None

    if sensor == "fetch-request-hunt":
        return {"queries": queries[:3], "limit": 20}, None

    if sensor == "fetch-news-api":
        return {"queries": queries[:3], "days": 7, "language": lang}, None

    if sensor == "fetch-gnews":
        return {"queries": queries[:3], "max_results": 10, "language": lang}, None

    if sensor == "fetch-x":
        compact_queries = [q[:60] for q in queries[:3]]
        return {"queries": compact_queries, "max_results": 10, "min_likes": 5}, None

    if sensor == "fetch-arxiv":
        return {
            "queries": _academic_queries(queries),
            "categories": ["cs.AI", "cs.CL", "cs.SE"],
            "max_results": 20,
        }, None

    if sensor == "fetch-openalex":
        return {
            "queries": _academic_queries(queries),
            "per_page": 20,
            "publication_year": f"{now.year - 1}-{now.year}",
        }, None

    return None, None


def _run_sensor(
    root_dir: Path,
    sensor: str,
    queries: list[str],
    intent_text: str,
    identity_text: str,
    now: datetime,
) -> SensorRunResult:
    script_rel = SENSOR_SCRIPT_PATHS[sensor]
    payload, args = _sensor_payload(sensor, queries, intent_text, identity_text, now)

    if sensor == "fetch-rss" and payload and not payload.get("feeds"):
        return SensorRunResult(sensor=sensor, success=True, items=[])

    if sensor not in {"fetch-hacker-news", "fetch-github-trending", "fetch-v2ex", "fetch-product-hunt", "fetch-reddit", "fetch-rss"}:
        if not payload or not payload.get("queries"):
            return SensorRunResult(sensor=sensor, success=True, items=[])

    data, stderr = _run_json_script(root_dir, script_rel, payload=payload, args=args)
    items = data.get("items") if isinstance(data.get("items"), list) else []
    success = bool(data.get("success", False))
    error = data.get("error", "")
    if stderr and not success:
        error = f"{error}; {stderr}" if error else stderr

    return SensorRunResult(sensor=sensor, success=success, items=items, error=error)


def _save_sensor_items(root_dir: Path, sensor_result: SensorRunResult) -> int:
    payload = {
        "success": sensor_result.success,
        "items": sensor_result.items,
    }
    data, stderr = _run_json_script(root_dir, DB_SAVE_ITEMS_SCRIPT, payload=payload)
    data = _require_script_success(data, stderr, DB_SAVE_ITEMS_SCRIPT)
    return int(data.get("inserted", 0) or 0)


def _query_unanalyzed(root_dir: Path, watch_name: str) -> list[dict[str, Any]]:
    data, stderr = _run_json_script(
        root_dir,
        DB_QUERY_SCRIPT,
        args=["--watch", watch_name, "--unanalyzed"],
    )
    data = _require_script_success(data, stderr, DB_QUERY_SCRIPT)
    items = data.get("items") if isinstance(data.get("items"), list) else []
    return [item for item in items if _normalize_text(str(item.get("source") or "")) != NEED_CLUSTER_SOURCE]


def _save_analysis(
    root_dir: Path,
    watch_name: str,
    item_ids: list[int],
    report_path: str,
    item_count: int,
    lens: str,
) -> None:
    payload = {
        "watch_name": watch_name,
        "item_ids": item_ids,
        "report_path": report_path,
        "item_count": item_count,
        "lens": lens,
    }
    data, stderr = _run_json_script(root_dir, DB_SAVE_ANALYSIS_SCRIPT, payload=payload)
    _require_script_success(data, stderr, DB_SAVE_ANALYSIS_SCRIPT)


def _to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> datetime:
        for item_key in ("published_at", "fetched_at"):
            dt = _to_datetime(item.get(item_key))
            if dt is not None:
                return dt
        return datetime.min

    return sorted(items, key=key, reverse=True)


def _item_markdown_line(item: dict[str, Any]) -> str:
    title = (item.get("normalizedTitle") or item.get("title") or "(untitled)").strip()
    url = item.get("url") or ""
    source = item.get("source") or "unknown"
    when = item.get("published_at") or item.get("fetched_at") or "unknown time"
    if url:
        return f"- [{title}]({url}) — `{source}` · {when}"
    return f"- {title} — `{source}` · {when}"


def _render_report(
    watch_name: str,
    lens: str,
    items: list[dict[str, Any]],
    need_clusters: list[NeedCluster],
    need_candidates: list[dict[str, Any]],
    preprocess_summary: PreprocessRunSummary,
    language_code: str,
    sensor_results: list[SensorRunResult],
    now: datetime,
) -> str:
    ordered = _sorted_items(items)
    recent = ordered[:12]
    time_str = now_iso_with_tz(now)

    source_counts = _count_items_by_source(items)

    source_summary = _format_source_counts(source_counts) or "No data"
    sensor_errors = [f"- `{r.sensor}`: {r.error or 'failed'}" for r in sensor_results if not r.success]
    candidate_summary = _summarize_need_candidates(
        items,
        need_candidates,
        selection_limit=_resolve_max_need_analysis_items(),
    )

    is_zh = language_code == "zh"

    if not items and not need_clusters:
        run_result_line = "本轮没有可分析的新数据。\n\n" if is_zh else "No new items were available for analysis in this run.\n\n"
        return (
            f"# {watch_name} insights\n\n"
            f"Generated at: {time_str}\n"
            f"Lens: `{lens}`\n\n"
            "---\n\n"
            "## Run Result\n"
            f"{run_result_line}"
            "---\n\n"
            "## Data Source Notes\n"
            + ("\n".join(sensor_errors) if sensor_errors else "- All selected sensors completed but returned no new data.")
            + "\n"
        )

    if lens == "flash_brief":
        body = "\n".join(_item_markdown_line(it) for it in recent[:5])
        return (
            f"# {watch_name} flash brief\n\n"
            f"Generated at: {time_str}\n"
            f"Lens: `{lens}`\n\n"
            "---\n\n"
            "## Top Signals\n"
            f"{body}\n\n"
            "---\n\n"
            "## Source Coverage\n"
            f"- {source_summary}\n"
        )

    if lens == "dual_take":
        positives = recent[: min(4, len(recent))]
        negatives = recent[min(4, len(recent)) : min(8, len(recent))]
        pos_text = "\n".join(_item_markdown_line(it) for it in positives) or "- No positive signals extracted."
        neg_text = "\n".join(_item_markdown_line(it) for it in negatives) or "- No obvious downside signals extracted."
        return (
            f"# {watch_name} dual take\n\n"
            f"Generated at: {time_str}\n"
            f"Lens: `{lens}`\n\n"
            "---\n\n"
            "## Bull Case\n"
            f"{pos_text}\n\n"
            "---\n\n"
            "## Bear Case\n"
            f"{neg_text}\n\n"
            "---\n\n"
            "## Source Coverage\n"
            f"- {source_summary}\n"
        )

    if lens == "timeline_trace":
        timeline = "\n".join(_item_markdown_line(it) for it in recent)
        return (
            f"# {watch_name} timeline trace\n\n"
            f"Generated at: {time_str}\n"
            f"Lens: `{lens}`\n\n"
            "---\n\n"
            "## Timeline\n"
            f"{timeline}\n\n"
            "---\n\n"
            "## Source Coverage\n"
            f"- {source_summary}\n"
        )

    candidate_funnel_lines = [
        f"- Relevant need candidates available: {candidate_summary.relevant_count}",
        f"- Candidates passed to clustering: {candidate_summary.selected_count} (limit {candidate_summary.selection_limit})",
        f"- Relevant candidate sources: {_format_source_counts(candidate_summary.relevant_source_counts)}",
        f"- Cluster input sources: {_format_source_counts(candidate_summary.selected_source_counts)}",
    ]
    candidate_funnel = "\n".join(candidate_funnel_lines)

    if not need_clusters:
        run_summary_lines = [
            f"- Total unanalyzed items: {candidate_summary.total_items}",
            f"- Preprocess status: valid={candidate_summary.valid_count}, irrelevant={candidate_summary.irrelevant_count}, invalid={candidate_summary.invalid_count}, missing={candidate_summary.missing_count}",
            candidate_funnel,
            f"- Preprocess refresh: targets={preprocess_summary.total_targets}, batches={preprocess_summary.llm_batches}, success={preprocess_summary.success_count}, failed={preprocess_summary.failure_count}",
        ]
        if preprocess_summary.error:
            run_summary_lines.append(f"- Preprocess warning: {preprocess_summary.error}")
        if sensor_errors:
            run_summary_lines.append(f"- Sensor failures: {len(sensor_errors)}")
        run_summary = "\n".join(run_summary_lines)

        if is_zh:
            return (
                f"# {watch_name} insights\n\n"
                f"Generated at: {time_str}\n"
                f"Lens: `{lens}`\n\n"
                "---\n\n"
                "## 运行摘要\n"
                f"{run_summary}\n\n"
                "---\n\n"
                "## 聚类需求卡\n"
                "本轮没有形成稳定的需求聚类，请检查传感器覆盖或放宽 watch 意图约束。\n\n"
                "---\n\n"
                "## Source Coverage\n"
                f"- {source_summary}\n"
            )
        return (
            f"# {watch_name} insights\n\n"
            f"Generated at: {time_str}\n"
            f"Lens: `{lens}`\n\n"
            "---\n\n"
            "## Run Summary\n"
            f"{run_summary}\n\n"
            "---\n\n"
            "## Cluster Need Cards\n"
            "No stable need clusters were produced in this run. Check source coverage or relax watch intent constraints.\n\n"
            "---\n\n"
            "## Source Coverage\n"
            f"- {source_summary}\n"
        )

    theme_lines: list[str] = []
    evidence_lines: list[str] = []
    detail_lines: list[str] = []
    for index, cluster in enumerate(need_clusters, start=1):
        theme_lines.append(
            f"{index}. {cluster.title} · evidence {len(cluster.evidence_item_ids)} · confidence {cluster.confidence}"
        )
        if cluster.evidence_urls:
            evidence_lines.append(f"{index}. " + " | ".join(cluster.evidence_urls[:4]))
        else:
            evidence_lines.append(f"{index}. (no evidence urls)")

        detail_lines.extend(
            [
                f"### {index}. {cluster.title}",
                f"- Need Summary: {cluster.summary}",
                f"- Pain Point: {cluster.pain_point}",
                f"- Target User: {cluster.target_user}",
                f"- Suggested Direction: {cluster.suggested_direction}",
                f"- Why Now: {cluster.why_now}",
                f"- Evidence Item IDs: {', '.join(str(item_id) for item_id in cluster.evidence_item_ids)}",
                "",
            ]
        )

    action_suggestions = [f"{index}. {cluster.suggested_direction}" for index, cluster in enumerate(need_clusters[:3], start=1)]
    if len(action_suggestions) < 3:
        action_suggestions.append(
            f"{len(action_suggestions) + 1}. Re-run with more demand-focused sources if signal volume is low."
        )
    action_block = "\n".join(action_suggestions[:3])

    preprocess_line = (
        f"- Preprocess refresh: targets={preprocess_summary.total_targets}, batches={preprocess_summary.llm_batches}, success={preprocess_summary.success_count}, failed={preprocess_summary.failure_count}"
    )
    if preprocess_summary.error:
        preprocess_line += f"\n- Preprocess warning: {preprocess_summary.error}"

    return (
        f"# {watch_name} insights\n\n"
        f"Generated at: {time_str}\n"
        f"Lens: `{lens}`\n\n"
        "---\n\n"
        "## Need Themes\n"
        + "\n".join(theme_lines)
        + "\n\n---\n\n"
        "## Evidence Links\n"
        + "\n".join(evidence_lines)
        + "\n\n---\n\n"
        "## Theme-by-Theme Analysis\n"
        + "\n".join(detail_lines).rstrip()
        + "\n\n---\n\n"
        "## Action Suggestions\n"
        f"{action_block}\n\n"
        "---\n\n"
        "## Run Diagnostics\n"
        f"- Total unanalyzed items: {candidate_summary.total_items}\n"
        f"- Preprocess status (for analyzed pool): valid={candidate_summary.valid_count}, irrelevant={candidate_summary.irrelevant_count}, invalid={candidate_summary.invalid_count}, missing={candidate_summary.missing_count}\n"
        f"{candidate_funnel}\n"
        f"- Need clusters generated: {len(need_clusters)}\n"
        f"{preprocess_line}\n"
        + (f"- Sensor failures: {len(sensor_errors)}\n" if sensor_errors else "- Sensor failures: 0\n")
        + "\n---\n\n"
        "## Source Coverage\n"
        f"- {source_summary}\n"
    )


def _signal_terms(intent_text: str) -> list[str]:
    tokens: list[str] = []
    for phrase in _safe_words_from_markdown(intent_text):
        for token in re.split(r"[,，/;；\s]+", phrase):
            normalized = token.strip("-:()[]{}")
            if len(normalized) >= 3:
                tokens.append(normalized.lower())
            elif has_chinese(normalized) and len(normalized) >= 2:
                tokens.append(normalized)
    dedup: list[str] = []
    for token in tokens:
        if token not in dedup:
            dedup.append(token)
    return dedup[:12]


def _detect_alerts(items: list[dict[str, Any]], intent_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    terms = _signal_terms(intent_text)

    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        score = 0
        text = f"{item.get('normalizedTitle', '')} {item.get('normalizedContent', '')} {item.get('title', '')} {item.get('content', '')}".lower()

        if any(term in text for term in terms):
            score += 2

        source = str(item.get("source") or "")
        if source in {"news_api", "gnews", "x_twitter", "product_hunt"}:
            score += 1

        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        numeric_signals = [
            metadata.get("votes_count", 0),
            metadata.get("like_count", 0),
            metadata.get("score", 0),
            metadata.get("stars_today", 0),
        ]
        if any(isinstance(v, (int, float)) and v >= 50 for v in numeric_signals):
            score += 1

        if score >= 2:
            ranked.append((score, item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    highs = [item for score, item in ranked if score >= 3][:3]
    mediums = [item for score, item in ranked if score == 2][:3]
    return highs, mediums


def _render_alert_markdown(
    watch_name: str,
    highs: list[dict[str, Any]],
    mediums: list[dict[str, Any]],
    now: datetime,
) -> str:
    lines = [
        f"# {watch_name} alert",
        "",
        f"Generated at: {now_iso_with_tz(now)}",
        "",
        "---",
        "",
    ]

    for level, level_items in (("High", highs), ("Medium", mediums)):
        for item in level_items:
            title = item.get("title") or "(untitled)"
            source = item.get("source") or "unknown"
            url = item.get("url") or ""
            lines.extend(
                [
                    f"## [{level}] {title}",
                    "",
                    f"- **Source**: {source}",
                    f"- **Link**: {url}" if url else "- **Link**: (none)",
                    "- **Reason**: Strong alignment with watch intent and elevated source signal.",
                    "",
                    "---",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def update_watch_state(state_path: Path, now: datetime | None = None) -> dict[str, Any]:
    """Update watch state.json last_run with timezone-aware ISO timestamp."""
    current = now or datetime.now().astimezone()
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    else:
        state = {}

    state.setdefault("check_interval", "1d")
    state.setdefault("status", "active")
    state["last_run"] = now_iso_with_tz(current)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def run_watch(
    root_dir: Path,
    watch_name: str,
    lens_override: str | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Execute one watch cycle using existing skill scripts as runtime primitives."""
    root = root_dir.resolve()
    watch_dir = root / "watches" / watch_name
    intent_path = watch_dir / "intent.md"
    memory_path = watch_dir / "memory.md"
    state_path = watch_dir / "state.json"

    if not watch_dir.exists() or not intent_path.exists():
        raise FileNotFoundError(f"Watch '{watch_name}' not found at {watch_dir}")

    intent_text = read_text(intent_path)
    memory_text = read_text(memory_path)
    identity_text = read_text(root / "profile/identity.md")
    runtime_intent = _sanitize_watch_text(intent_text) or intent_text
    runtime_memory = _sanitize_watch_text(memory_text) or memory_text

    selected_sensors = select_sensors(runtime_intent, runtime_memory)
    lens = infer_lens(runtime_memory, override=lens_override)
    language_code = _language_code(runtime_intent, identity_text)
    now = datetime.now().astimezone()
    queries = generate_search_queries(watch_name, runtime_intent, runtime_memory, now=now)

    sensor_results: list[SensorRunResult] = []
    inserted_items = 0

    for sensor in selected_sensors:
        result = _run_sensor(root, sensor, queries, runtime_intent, identity_text, now)
        result.inserted = _save_sensor_items(root, result)
        inserted_items += result.inserted
        sensor_results.append(result)

    preprocess_summary = _apply_preprocess_to_items(
        root,
        watch_name=watch_name,
        intent_text=runtime_intent,
        memory_text=runtime_memory,
        language_code=language_code,
        now=now,
    )

    items = _query_unanalyzed(root, watch_name)
    if since:
        threshold = _to_datetime(since)
        if threshold:
            filtered = []
            for item in items:
                fetched = _to_datetime(item.get("fetched_at"))
                if fetched and fetched >= threshold:
                    filtered.append(item)
            items = filtered

    need_clusters: list[NeedCluster] = []
    cluster_item_ids: list[int] = []
    need_candidates: list[dict[str, Any]] = []
    if lens == "deep_insight":
        need_candidates = _select_need_candidates(items)
        if not need_candidates:
            need_candidates = _load_candidate_fallback_pool(
                root,
                preprocess_run_id=preprocess_summary.run_id,
            )
        if need_candidates:
            clustered = _cluster_need_candidates(
                root_dir=root,
                watch_name=watch_name,
                intent_text=runtime_intent,
                memory_text=runtime_memory,
                language_code=language_code,
                candidates=need_candidates,
            )
            cluster_run_id = preprocess_summary.run_id or f"watch-cluster-{now_iso_with_tz(now)}"
            need_clusters = _persist_need_clusters(
                root_dir=root,
                watch_name=watch_name,
                language_code=language_code,
                preprocess_run_id=cluster_run_id,
                preprocessed_at=now_iso_with_tz(now),
                clusters=clustered,
                candidates=need_candidates,
            )
            cluster_item_ids = [cluster.item_id for cluster in need_clusters if cluster.item_id > 0]

    report_markdown = _render_report(
        watch_name,
        lens,
        items,
        need_clusters,
        need_candidates,
        preprocess_summary,
        language_code,
        sensor_results,
        now,
    )

    date_dir = now.strftime("%Y-%m-%d")
    report_dir = root / "reports" / date_dir / watch_name
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "insights.md"
    report_path.write_text(report_markdown, encoding="utf-8")

    raw_path = report_dir / "raw_intel.md"
    raw_lines = [
        f"# {watch_name} raw intel",
        "",
        f"Generated at: {now_iso_with_tz(now)}",
        "",
        "---",
        "",
    ]
    for item in _sorted_items(items)[:50]:
        raw_lines.append(_item_markdown_line(item))
    if len(raw_lines) <= 7:
        raw_lines.append("- No unanalyzed items in this cycle.")
    raw_path.write_text("\n".join(raw_lines).rstrip() + "\n", encoding="utf-8")

    highs, mediums = _detect_alerts(items, runtime_intent)
    alert_rel_path = ""
    if highs or mediums:
        alert_dir = root / "alerts" / date_dir
        alert_dir.mkdir(parents=True, exist_ok=True)
        alert_path = alert_dir / f"{watch_name}.md"
        alert_path.write_text(_render_alert_markdown(watch_name, highs, mediums, now), encoding="utf-8")
        alert_rel_path = str(alert_path.relative_to(root))

    item_ids = [int(item["id"]) for item in items if isinstance(item.get("id"), int)]
    for cluster_item_id in cluster_item_ids:
        if cluster_item_id not in item_ids:
            item_ids.append(cluster_item_id)
    report_rel_path = str(report_path.relative_to(root))
    _save_analysis(root, watch_name, item_ids, report_rel_path, len(item_ids), lens)
    update_watch_state(state_path, now=now)

    candidate_summary = _summarize_need_candidates(
        items,
        need_candidates,
        selection_limit=_resolve_max_need_analysis_items(),
    )

    return {
        "success": True,
        "watch": watch_name,
        "selected_sensors": selected_sensors,
        "inserted_items": inserted_items,
        "analyzed_items": len(item_ids),
        "need_candidates": len(need_candidates),
        "need_clusters": len(need_clusters),
        "need_funnel": {
            "total": candidate_summary.total_items,
            "valid": candidate_summary.valid_count,
            "irrelevant": candidate_summary.irrelevant_count,
            "invalid": candidate_summary.invalid_count,
            "missing": candidate_summary.missing_count,
            "relevant": candidate_summary.relevant_count,
            "selected": candidate_summary.selected_count,
            "selection_limit": candidate_summary.selection_limit,
            "relevant_sources": candidate_summary.relevant_source_counts,
            "selected_sources": candidate_summary.selected_source_counts,
        },
        "preprocess": {
            "targets": preprocess_summary.total_targets,
            "batches": preprocess_summary.llm_batches,
            "success": preprocess_summary.success_count,
            "failed": preprocess_summary.failure_count,
            "runId": preprocess_summary.run_id,
            "error": preprocess_summary.error,
        },
        "report_path": report_rel_path,
        "alert_path": alert_rel_path,
        "lens": lens,
        "sensor_errors": [
            {"sensor": r.sensor, "error": r.error} for r in sensor_results if not r.success and r.error
        ],
    }
