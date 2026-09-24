"""Shared category definitions and classification logic for stacked breakdown plots."""
import json
from pathlib import Path

_COLLAPSE_TEXT = "== ... =="

# Bottom-to-top stack order; high_ppl label filled in dynamically.
# off_safe_llm requires llm_on_topic=True in predictions (written by llm_on_topic_scorer.py);
# when the field is absent it is treated as off_safe (backward-compatible).
LAYER_SPECS = [
    ("collapse",     "Collapse",                              "#8B0000"),
    ("sh_degen",     "Short degen",                          "#C0392B"),
    ("fmt_fail",     "Format fail",                          "#E74C3C"),
    ("high_ppl",     None,                                   "#FAB8A0"),
    ("off_safe",     "Off-topic, not violated",              "#7F8C8D"),
    ("on_violated",  "On-topic + violated",                  "#F39C12"),
    ("off_safe_llm", "Off-topic (WordNet), LLM-approved",   "#90EE90"),
    ("ic",           "IC (on-topic, not violated)",          "#27AE60"),
]

# 'red' outcomes: degenerate/junk generations, not a real measurement of constraint
# behavior -- exclude from anything that treats violated/not-violated as ground truth.
RED_OUTCOMES = ("collapse", "sh_degen", "fmt_fail", "high_ppl")


def make_layers(ppl_threshold: float, use_llm_violated: bool = False) -> list[tuple[str, str, str]]:
    return [
        (key,
         f"High PPL (>{int(ppl_threshold)})" if key == "high_ppl"
         else (f"{label} (LLM judge)" if key == "on_violated" and use_llm_violated else label),
         color)
        for key, label, color in LAYER_SPECS
    ]


def classify(r: dict, ppl_threshold: float, use_llm_violated: bool = False) -> str:
    gen_text = r.get("generated_text", "")
    gen_len  = r.get("gen_length", 99)
    fmt_adh  = r.get("fmt_adh", 1.0)
    ppl      = r.get("ppl", 0.0)
    on_topic = r.get("on_topic", False)

    if use_llm_violated:
        if "llm_violated" not in r:
            raise KeyError(
                f"record id={r.get('id')} is missing 'llm_violated' -- run "
                "scripts/llm_violated_scorer.py on this run's predictions.jsonl first"
            )
        violated = r["llm_violated"]
    else:
        violated = r.get("violated", False)

    if gen_text == _COLLAPSE_TEXT:
        return "collapse"
    if gen_len < 5:
        return "sh_degen"
    if fmt_adh == 0.0:
        return "fmt_fail"
    if ppl > ppl_threshold:
        return "high_ppl"
    if not use_llm_violated:
        # WordNet on_topic/violated share a word-list construction that makes this
        # combination impossible; not guaranteed once violated comes from the LLM judge.
        assert not (not on_topic and violated), \
            f"impossible category: off-topic + violated (id={r.get('id')})"
    if on_topic and violated:
        return "on_violated"
    if not on_topic:
        if r.get("llm_on_topic", False):
            return "off_safe_llm"
        return "off_safe"
    return "ic"


def compute_fractions(records: list[dict], ppl_threshold: float, use_llm_violated: bool = False) -> dict[str, float]:
    keys = [key for key, *_ in LAYER_SPECS]
    counts = {key: 0 for key in keys}
    for r in records:
        counts[classify(r, ppl_threshold, use_llm_violated)] += 1
    n = len(records)
    return {key: counts[key] / n for key in keys}


def load_predictions(path: Path) -> list[dict]:
    if not path.is_dir():
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # Pool records across all available seed dirs.
    records = []
    for seed_file in sorted(path.glob("seed_*/predictions.jsonl")):
        records.extend(json.loads(l) for l in seed_file.read_text().splitlines() if l.strip())
    return records
