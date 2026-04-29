"""Architecture boundary for quantitative technical pattern features.

Pattern proxies are calculated inside compute_features() in features.py and are numeric,
not visual chart interpretation.
"""

from features import compute_features

__all__ = ["compute_features"]
