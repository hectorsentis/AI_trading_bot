import argparse
import json

import numpy as np

from config import LONG_THRESHOLD, SHORT_THRESHOLD, SIGNAL_MIN_CONFIDENCE, SIGNAL_MIN_MARGIN
from modeling_utils import CLASS_TO_NAME, CLASS_TO_POSITION


def generate_signal_from_probabilities(
    prob_short: float,
    prob_flat: float,
    prob_long: float,
    min_confidence: float = SIGNAL_MIN_CONFIDENCE,
    min_margin: float = SIGNAL_MIN_MARGIN,
    short_threshold: float = SHORT_THRESHOLD,
    long_threshold: float = LONG_THRESHOLD,
) -> dict:
    probs = np.array([prob_short, prob_flat, prob_long], dtype=float)
    order = np.argsort(probs)[::-1]
    best_idx = int(order[0])
    second_idx = int(order[1])

    best_prob = float(probs[best_idx])
    second_prob = float(probs[second_idx])
    margin = best_prob - second_prob

    reasons: list[str] = []
    final_position = CLASS_TO_POSITION[best_idx]

    if best_prob < float(min_confidence):
        reasons.append("confidence_below_min")
        final_position = 0

    if margin < float(min_margin):
        reasons.append("margin_below_min")
        final_position = 0

    if best_idx == 0 and prob_short < float(short_threshold):
        reasons.append("short_prob_below_threshold")
        final_position = 0

    if best_idx == 2 and prob_long < float(long_threshold):
        reasons.append("long_prob_below_threshold")
        final_position = 0

    if best_idx == 1:
        reasons.append("raw_class_flat")
        final_position = 0

    research_label = {1: "LONG", 0: "FLAT", -1: "SHORT"}[int(final_position)]
    if final_position > 0:
        spot_action_hint = "BUY_OR_HOLD"
    elif final_position < 0:
        spot_action_hint = "REDUCE_OR_FLAT"
    else:
        spot_action_hint = "CLOSE_OR_HOLD"

    return {
        "probabilities": {
            "short": float(prob_short),
            "flat": float(prob_flat),
            "long": float(prob_long),
        },
        "raw_class": CLASS_TO_NAME[best_idx],
        "raw_position": CLASS_TO_POSITION[best_idx],
        "confidence": best_prob,
        "margin_vs_second": margin,
        "second_class": CLASS_TO_NAME[second_idx],
        "final_signal_position": int(final_position),
        "final_signal_label": research_label,
        "research_signal_label": research_label,
        "spot_action_hint": spot_action_hint,
        "spot_policy": "Spot paper trading never opens short positions; SHORT is treated as bearish/risk-off.",
        "accepted": int(final_position) != 0,
        "reasons": reasons,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final signal from class probabilities.")
    parser.add_argument("--prob-short", type=float, required=True)
    parser.add_argument("--prob-flat", type=float, required=True)
    parser.add_argument("--prob-long", type=float, required=True)
    parser.add_argument("--min-confidence", type=float, default=SIGNAL_MIN_CONFIDENCE)
    parser.add_argument("--min-margin", type=float, default=SIGNAL_MIN_MARGIN)
    return parser.parse_args()


def main():
    args = parse_args()
    signal = generate_signal_from_probabilities(
        prob_short=args.prob_short,
        prob_flat=args.prob_flat,
        prob_long=args.prob_long,
        min_confidence=args.min_confidence,
        min_margin=args.min_margin,
    )
    print(json.dumps(signal, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
