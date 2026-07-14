"""测试环境配置。"""

from __future__ import annotations

import os
from pathlib import Path

_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".pytest-tmp"
_TEMP_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMP_ROOT))
