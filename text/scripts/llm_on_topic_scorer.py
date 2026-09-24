"""
Post-hoc LLM on_topic scoring using Qwen3-8B.

For each prediction, prompts Qwen3-8B with a YES/NO question about whether the
generated text is on-topic for the datapoint's topic.  Adds `llm_on_topic` (bool)
and `llm_confidence` (float) fields to every line of predictions.jsonl in-place.

Files that already have `llm_on_topic` on every line are skipped.

Designed to mirror ppl_scorer.py so it can later be called from main.py with
    scorer = LLMOnTopicScorer()
    record = scorer.judge_record(record)   # adds llm_on_topic, llm_confidence

Usage:
    # Score all predictions.jsonl files under a run dir
    python scripts/llm_on_topic_scorer.py --run_dir runs/full_run/llada_instruct/pos_vs_neg_guidance

    # Score specific files
    python scripts/llm_on_topic_scorer.py runs/.../seed_0/predictions.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "Qwen/Qwen3-8B"

_SYSTEM_PROMPT = (
    "You are a precise evaluation assistant. "
    "You answer only YES or NO, with no explanation."
)


def _make_prompt(topic: str, generated_text: str) -> str:
    return (
        f'Topic category: "{topic}"\n'
        f'Generated text: {generated_text}\n\n'
        f'Is the main subject of the generated text an instance or member of the '
        f'category "{topic}"? Answer YES or NO.'
    )


class LLMOnTopicScorer:
    def __init__(self, model_name: str = JUDGE_MODEL):
        print(f"Loading {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).cuda()
        self.model.eval()

        # Pre-compute token ids for YES / NO (first subword token of each variant)
        self._yes_ids: set[int] = set()
        self._no_ids:  set[int] = set()
        for v in ("YES", " YES", "Yes", " Yes", "yes", " yes"):
            toks = self.tokenizer.encode(v, add_special_tokens=False)
            if toks:
                self._yes_ids.add(toks[0])
        for v in ("NO", " NO", "No", " No", "no", " no"):
            toks = self.tokenizer.encode(v, add_special_tokens=False)
            if toks:
                self._no_ids.add(toks[0])

    @torch.no_grad()
    def judge_record(self, record: dict) -> dict:
        """
        Add llm_on_topic (bool) and llm_confidence (float) to a prediction dict.
        Returns the same dict (mutated in-place and also returned for convenience).
        """
        topic     = record["datapoint"]["topic"]
        gen_text  = record.get("generated_text", "")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _make_prompt(topic, gen_text)},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs  = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out     = self.model(**inputs)
        logits  = out.logits[0, -1, :]   # next-token distribution

        yes_logit = max((logits[t].item() for t in self._yes_ids), default=-1e9)
        no_logit  = max((logits[t].item() for t in self._no_ids),  default=-1e9)

        exp_yes = torch.tensor(yes_logit).exp()
        exp_no  = torch.tensor(no_logit).exp()
        confidence = (exp_yes / (exp_yes + exp_no)).item()

        record["llm_on_topic"]   = confidence > 0.5
        record["llm_confidence"] = round(confidence, 4)
        return record

    def score_file(self, path: Path | str) -> None:
        path  = Path(path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            return

        # Skip if already fully scored
        first = json.loads(lines[0])
        if "llm_on_topic" in first:
            print(f"  skip (already scored): {path}")
            return

        updated: list[str] = []
        for i, line in enumerate(lines):
            record = json.loads(line)
            self.judge_record(record)
            updated.append(json.dumps(record))

        path.write_text("\n".join(updated) + "\n")
        n_yes = sum(1 for l in updated if json.loads(l).get("llm_on_topic"))
        print(f"  scored {len(updated)} predictions, llm_on_topic={n_yes/len(updated):.3f}  →  {path}")


def find_prediction_files(run_dir: str | Path) -> list[Path]:
    return sorted(Path(run_dir).rglob("predictions.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc LLM on_topic scorer (Qwen3-8B)")
    parser.add_argument("paths", nargs="*", metavar="FILE",
                        help="predictions.jsonl files to score")
    parser.add_argument("--run_dir", metavar="DIR",
                        help="Recursively find and score all predictions.jsonl under this dir")
    args = parser.parse_args()

    files: list[str] = list(args.paths)
    if args.run_dir:
        files += [str(p) for p in find_prediction_files(args.run_dir)]

    if not files:
        parser.error("Specify --run_dir or pass predictions.jsonl paths directly.")

    scorer = LLMOnTopicScorer()
    for f in files:
        scorer.score_file(Path(f))

    print("Done.")
