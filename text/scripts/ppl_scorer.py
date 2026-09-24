"""
Post-hoc PPL scoring using SmolLM3-3B as a fixed external LM.

Scores P(generated_text | context) where context = datapoint["context"]
(raw Wikipedia-style prompt, no chat template or suffix) for fair cross-model comparison.

Overwrites the `ppl` field in each predictions.jsonl in-place.
SmolLM3 bias: SmolLM3-generated outputs will have artificially low PPL.

Usage:
    # Score all runs under ./runs/
    python ppl_scorer.py --run_dir ./runs

    # Score specific files
    python ppl_scorer.py runs/full_run_llada8b_dlm/seed_0/predictions.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PPL_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"
MAX_TOKENS = 2048


class PPLScorer:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL)
        model = AutoModelForCausalLM.from_pretrained(PPL_MODEL, torch_dtype=torch.bfloat16).cuda()
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def score_text(self, prefix, text):
        if not text or not text.strip():
            return 0.0
        text_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        n_text = text_ids.shape[1]
        if n_text < 2:
            return 0.0
        prefix_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
        # keep the tail of the prefix to stay within budget, preserving all generated tokens
        n_prefix = min(prefix_ids.shape[1], MAX_TOKENS - n_text)
        if n_prefix > 0:
            prefix_ids = prefix_ids[:, -n_prefix:]
            input_ids = torch.cat([prefix_ids, text_ids], dim=1).cuda()
        else:
            input_ids = text_ids.cuda()
            n_prefix = 0
        labels = input_ids.clone()
        labels[0, :n_prefix] = -100
        outputs = self.model(input_ids, labels=labels)
        return torch.exp(outputs.loss).item()

    def score_file(self, path):
        path = Path(path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        updated = []
        ppls = []
        for line in lines:
            pred = json.loads(line)
            prefix = pred.get("datapoint", {}).get("context", "")
            ppl = self.score_text(prefix, pred.get("generated_text", ""))
            pred["ppl"] = ppl
            ppls.append(ppl)
            updated.append(json.dumps(pred))
        path.write_text("\n".join(updated) + "\n")
        avg = sum(ppls) / len(ppls) if ppls else 0.0
        print(f"{path}: n={len(ppls)}, avg_ppl={avg:.2f}")


def find_prediction_files(run_dir):
    return sorted(Path(run_dir).rglob("predictions.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="predictions.jsonl files to score")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Scan this directory recursively for all predictions.jsonl files")
    args = parser.parse_args()

    files = list(args.paths)
    if args.run_dir:
        files += [str(p) for p in find_prediction_files(args.run_dir)]

    if not files:
        parser.error("Specify --run_dir or pass predictions.jsonl paths directly.")

    print(f"Loading {PPL_MODEL} ...")
    scorer = PPLScorer()

    for f in files:
        scorer.score_file(f)

    print("Done.")
