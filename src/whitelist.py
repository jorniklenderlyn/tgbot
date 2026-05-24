"""Whitelist loader with priority resolution and mtime-based cache.

Resolution order (explicit ID beats wildcard; suggest_only beats autonomous when equal):
  1. chat_id explicitly in suggest_only  ->  "suggest_only"
  2. chat_id explicitly in autonomous    ->  "autonomous"
  3. "*" in suggest_only                 ->  "suggest_only"
  4. "*" in autonomous                   ->  "autonomous"
  5. neither                             ->  None  (ignore)
"""

import json
import os
import sys

from src.config import WHITELIST_FILE

_cache: dict = {"mtime": 0.0, "autonomous_ids": set(), "suggest_ids": set(),
                "autonomous_wild": False, "suggest_wild": False}


def _reload_if_needed():
    try:
        mt = os.path.getmtime(WHITELIST_FILE)
    except OSError:
        return
    if mt == _cache["mtime"]:
        return

    try:
        with open(WHITELIST_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[whitelist] failed to load {WHITELIST_FILE}: {e}", file=sys.stderr)
        return

    auto_raw = data.get("autonomous", [])
    sug_raw = data.get("suggest_only", [])

    _cache["autonomous_ids"] = {x for x in auto_raw if isinstance(x, int)}
    _cache["suggest_ids"] = {x for x in sug_raw if isinstance(x, int)}
    _cache["autonomous_wild"] = "*" in auto_raw
    _cache["suggest_wild"] = "*" in sug_raw
    _cache["mtime"] = mt
    print(f"[whitelist] loaded: autonomous={_cache['autonomous_ids'] or ('*' if _cache['autonomous_wild'] else '{}')}, "
          f"suggest={_cache['suggest_ids'] or ('*' if _cache['suggest_wild'] else '{}')}", file=sys.stderr)


def resolve_mode(chat_id: int) -> str | None:
    """Returns 'autonomous', 'suggest_only', or None (ignore)."""
    _reload_if_needed()

    if chat_id in _cache["suggest_ids"]:
        return "suggest_only"
    if chat_id in _cache["autonomous_ids"]:
        return "autonomous"
    if _cache["suggest_wild"]:
        return "suggest_only"
    if _cache["autonomous_wild"]:
        return "autonomous"
    return None
