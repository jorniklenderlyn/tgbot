"""Backwards-compatible shim — re-exports from src.rag.

Scripts originally lived in this folder and imported `from _rag import ...`.
The canonical home is now `src/rag.py`; this shim keeps existing scripts
working without edits.
"""

import sys
from pathlib import Path

# Make the project root importable so `from src.rag import ...` works
# regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag import (  # noqa: E402,F401
    embed_one,
    embed_texts,
    ensure_collection,
    get_embedder,
    get_qdrant,
    message_payload_text,
    render_message,
)
