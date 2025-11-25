#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_interpretability.py

Thin wrapper to run the interpretability pipeline from CLI:
- subcommands: 'shap' and 'saliency'
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- Igual que en la otra opinión: añadir src/ al sys.path si hace falta ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ---------------------------------------------------------------------------

from betavae_xai.interpretability.interpret_fold import main

if __name__ == "__main__":
    main()
