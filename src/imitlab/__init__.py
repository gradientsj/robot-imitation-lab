"""robot-imitation-lab: train and evaluate imitation-learning policies on open data.

This package keeps its top-level import light (no torch / lerobot) so the
pure-math modules (metrics, report) are importable and testable anywhere,
including CPU-only CI. Heavy dependencies are imported inside the train/eval
entry points.
"""

__version__ = "0.1.0"
