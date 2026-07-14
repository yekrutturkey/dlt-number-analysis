"""测试环境配置。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="dlt-number-analysis-pytest-"))
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMP_ROOT))
