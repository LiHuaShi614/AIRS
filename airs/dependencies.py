from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RRTTA_ROOT = WORKSPACE_ROOT / "RRTTA-main"


def ensure_rrtta_importable() -> Path:
    if not (RRTTA_ROOT / "rrtta" / "__init__.py").is_file():
        raise RuntimeError(f"RRTTA dependency not found at {RRTTA_ROOT}")
    root = str(RRTTA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return RRTTA_ROOT
