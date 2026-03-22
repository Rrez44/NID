"""
Shared setup for training scripts.

Ensures project root is on sys.path so config can be imported.
Import this module first: `import common` (side-effect: adds project root to path).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
