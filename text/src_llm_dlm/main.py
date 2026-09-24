import sys
import json
import random
import argparse
import subprocess
import urllib.parse
from pathlib import Path

import torch

from rich import print
from rich.markup import escape as _escape
from alive_progress import alive_bar


from src_llm_dlm.prompt_cfg_sampler import PromptCFGSampler

from src_llm_dlm.utils import (
    get_data,
    get_lm,
    get_tokenizer,
    cleanup_generated_text,
)
from src_llm_dlm.generation import run_generation_ar, run_generation_dlm, run_generation_nemotron, run_generation_nemotron_cfg
from src_llm_dlm.prompts import DEFAULT_CONTINUATION_SUFFIX, DEFAULT_NO_COPY_SUFFIX, DEFAULT_CONTINUATION_PREFIX, VALID_PROMPT_TYPES, build_prompts
from src_llm_dlm.metrics import compute_prediction_metrics
from src_llm_dlm.experiment import ExperimentManager
from src_llm_dlm.metric_tracker import MetricTracker



def _spawn_diffusion_gemma_server(args) -> subprocess.Popen:
    """Start the DiffusionGemma generation server as a subprocess (uses venv_gemma)."""
    repo_root = Path(__file__).resolve().parent.parent
    venv_python = repo_root / "venv_gemma" / "bin" / "python"
    server_script = repo_root / "scripts" / "diffusion_gemma_server.py"
    port = urllib.parse.urlparse(args.diffusion_gemma_server).port or 8765
    cmd = [
        str(venv_python), str(server_script),
        "--model_id", args.model_name,
        "--quant", args.quant,
        "--port", str(port),
    ]
    print(f"Starting DiffusionGemma server: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def _pick_eval_version(args):
    version_split = args.eval_version_split or args.dataset_split
    if args.eval_version == -1 and version_split == 'train':
        return random.choice(range(0, 3))
    if args.eval_version == -1 and version_split == 'dev':
        return random.choice(range(3, 6))
    if args.eval_version == -1 and version_split == 'test':
        return random.choice(range(6, 35))
    return args.eval_version


def _guidance_label(args):
    if args.legacy_cfg and args.cfg_scale is not None:
        return f'masking_cfg_{args.cfg_scale}'
    if args.cfg_scale is not None and not args.legacy_cfg:
        return f'cfg_{args.cfg_scale}__{args.base_prompt}__{args.cond_pos_prompt}__{args.cond_neg_prompt}'
    return f'base__{args.base_prompt}'


def _write_trace_line(trace_file, datapoint, step_traces, tokenizer):
    """Write one JSONL line to generation_trace.jsonl for a single datapoint."""
    steps = []
    for st in step_traces:
        steps.append({
            "step": st.step,
            "block": st.block,
            "block_step": st.block_step,
            "log_p_c": st.log_p_c,
            "gen_token_ids": st.gen_token_ids,
            "gen_token_texts": [tokenizer.decode([tid]) for tid in st.gen_token_ids],
            "decoded_tokens": [
                {
                    "position": ev.position,
                    "token_id": ev.token_id,
                    "token_text": tokenizer.decode([ev.token_id]),
                    "norm_diff": ev.norm_diff,
                    "token_prob": ev.token_prob,
                    "token_ppl": ev.token_ppl,
                    "topk": (
                        {
                            role: [
                                {"token_id": tid, "token_text": tokenizer.decode([tid]), "prob": prob}
                                for tid, prob in candidates
                            ]
                            for role, candidates in ev.topk.items()
                        }
                        if ev.topk is not None else None
                    ),
                }
                for ev in st.decoded_events
            ],
        })
    record = {
        "id": datapoint["id"],
        "mask_token_id": tokenizer.mask_token_id,
        "steps": steps,
    }
    trace_file.write(json.dumps(record) + '\n')


def run(args, run_id, exp_manager=None):
    datasets, hierarchy, _ = get_data(args)

    _dg_proc = None

    if args.model_type == 'diffusion_gemma':
        from src_llm_dlm.diffusion_gemma_cfg.client import wait_for_server, generate as dg_generate
        # The server applies the chat template internally; skip it in decorate().
        args.apply_chat_template = False
        _dg_proc = _spawn_diffusion_gemma_server(args)
        wait_for_server(args.diffusion_gemma_server)
        model = None
        tokenizer = None
        _is_nemotron = False
        _nemotron_cfg_mode = False
        sampler = None
    else:
        model = get_lm(args)
        tokenizer = get_tokenizer(args)
        _is_nemotron = hasattr(model, 'ar_generate')
        _nemotron_cfg_mode = (
            _is_nemotron and args.model_type == 'dlm'
            and args.cfg_scale is not None
            and not args.legacy_cfg
        )
        sampler = PromptCFGSampler(model=model, tokenizer=tokenizer) \
            if args.model_type == 'dlm' and not _is_nemotron else None

    predictions = []
    prediction_metrics = []
    total = len(datasets[args.dataset_split])
    target = min(args.num_datapoints, total)

    trace_path = (
        exp_manager.prediction_paths[run_id].parent / 'generation_trace.jsonl'
        if exp_manager is not None and args.model_type == 'dlm' and not _is_nemotron
        else None
    )
    trace_file = open(trace_path, 'w') if trace_path is not None else None

    try:
        with alive_bar(target, enrich_print=False) as bar:
            for idx, datapoint in enumerate(datasets[args.dataset_split]):
                if idx < args.datapoint_offset:
                    continue
                if len(predictions) == args.num_datapoints:
                    break

                topic = datapoint['topic']
                constraint = datapoint['constraint']

                eval_version = _pick_eval_version(args)
                datapoint['version'] = eval_version

                if args.seed is not None:
                    torch.manual_seed(args.seed + idx)
                    torch.cuda.manual_seed_all(args.seed + idx)

                base_prompt, cond_pos_prompt, cond_neg_prompt = build_prompts(datapoint, args, tokenizer, eval_version)

                gen_tokens = None
                mask_entropy = None
                generated_text = None
                step_traces = None

                if args.model_type == 'diffusion_gemma':
                    # WARNING: cond_neg_prompt is NOT passed to DiffusionGemma — the server-side CFG
                    # still uses the old formula (cond_pos - base). cond_neg is silently ignored here.
                    generated_text = dg_generate(
                        base_prompt, cond_pos_prompt,
                        args.cfg_scale or 0.0,
                        args.max_gen_length,
                        args.diffusion_gemma_server,
                    )
                elif args.model_type == 'dlm' and not _is_nemotron:
                    gen_tokens, mask_entropy, step_traces = run_generation_dlm(
                        base_prompt, cond_pos_prompt, cond_neg_prompt, sampler, tokenizer, args,
                    )
                elif _is_nemotron and _nemotron_cfg_mode:
                    # WARNING: cond_neg_prompt is NOT passed to Nemotron CFG — nemotron_generate_cfg
                    # still uses the old formula (cond_pos - base). cond_neg is silently ignored here.
                    gen_tokens = run_generation_nemotron_cfg(
                        base_prompt, cond_pos_prompt, model, tokenizer, args,
                    )
                elif _is_nemotron:
                    gen_tokens = run_generation_nemotron(base_prompt, model, tokenizer, args)
                else:
                    gen_tokens = run_generation_ar(base_prompt, model, tokenizer, args)

                if gen_tokens is not None:
                    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                prefix_text = (args.continuation_prefix_text or DEFAULT_CONTINUATION_PREFIX) if args.continuation_prefix else None
                generated_text = cleanup_generated_text(generated_text, continuation_prefix=prefix_text)
                gen_ids_decoded = [tokenizer.decode([t]) for t in gen_tokens.tolist()] if gen_tokens is not None else []

                if step_traces is not None and trace_file is not None:
                    _write_trace_line(trace_file, datapoint, step_traces[0], tokenizer)
                    trace_file.flush()

                guidance_label = _guidance_label(args)

                prediction = dict(
                    datapoint=datapoint,
                    prompt=base_prompt,
                    cfg_prompt=cond_pos_prompt,
                    cond_neg_prompt=cond_neg_prompt,
                    guidance=guidance_label,
                    generated_text=generated_text,
                    refinements=[],
                    generated_tokens=gen_ids_decoded,
                    mask_entropy=mask_entropy,
                )
                predictions.append(prediction)

                prediction_metric_outputs = compute_prediction_metrics(
                    prediction, hierarchy, data_mode=args.data,
                )
                prediction_metric = prediction_metric_outputs['prediction_metric']
                prediction_metric['id'] = datapoint['id']
                extracted = prediction_metric_outputs['extracted']

                prediction_metrics.append(prediction_metric)

                exp_manager.take(
                    {**prediction, **prediction_metric},
                    log=True,
                    console=False,
                    run_id=run_id,
                    add_metric=True,
                )

                v = prediction_metric['violated']
                o = prediction_metric['on_topic']
                onv = prediction_metric['on_topic_not_violated']
                print(f'\n--- [{len(predictions)}/{target}] name: {args.name} ---')
                print(args)
                print(f'eval_version: {eval_version}')
                print(f'topic: {topic}')
                print(datapoint['context'])
                print(f'constraint: {constraint}')
                print(f'generated_text:')
                print(_escape(str(generated_text)))
                print(f'Extracted: {_escape(str(extracted))}')
                print(f'v: {v}')
                print(f'o: {o}')
                print(f'onv: {onv}')
                print(f'Accumulated violated: {exp_manager.compute_metric("violated", run_id=run_id):.4f}')
                print(f'Accumulated on_topic: {exp_manager.compute_metric("on_topic", run_id=run_id):.4f}')
                print(f'Accumulated on_topic_not_violated: {exp_manager.compute_metric("on_topic_not_violated", run_id=run_id):.4f}')

                bar()

        stats = exp_manager.aggregate_metrics()
        exp_manager.save_results(stats)
        print(stats)
        print(args)
        print('Run finished.')

    finally:
        if trace_file is not None:
            trace_file.close()
        if _dg_proc is not None:
            print("Shutting down DiffusionGemma server ...")
            _dg_proc.terminate()
            _dg_proc.wait()

    if model is not None:
        del model
    if sampler is not None:
        del sampler
    torch.cuda.empty_cache()

    if args.skip_scoring:
        print("Skipping PPL / LLM on_topic scoring (--skip_scoring).")
    else:
        _score_ppl(args)
        _score_llm_on_topic(args)


def _score_ppl(args):
    if not args.name:
        return
    seed = args.seed if args.seed is not None else 0
    pred_path = Path(args.run_dir) / args.name / f"seed_{seed}" / "predictions.jsonl"
    if not pred_path.exists():
        return
    scorer = Path(__file__).resolve().parent.parent / "scripts" / "ppl_scorer.py"
    print("Scoring PPL ...")
    import subprocess
    subprocess.run([sys.executable, str(scorer), str(pred_path)], check=True)


def _score_llm_on_topic(args):
    if not args.name:
        return
    seed = args.seed if args.seed is not None else 0
    pred_path = Path(args.run_dir) / args.name / f"seed_{seed}" / "predictions.jsonl"
    if not pred_path.exists():
        return
    scorer = Path(__file__).resolve().parent.parent / "scripts" / "llm_on_topic_scorer.py"
    print("Scoring LLM on_topic ...")
    import subprocess
    subprocess.run([sys.executable, str(scorer), str(pred_path)], check=True)


def get_argparse(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default="./runs")
    parser.add_argument("--seed", type=int, default=0,
        help="Per-datapoint torch seed base (actual seed = seed + dataset_idx). None = no seeding."
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_scoring", action="store_true", default=False,
        help="Skip the post-hoc PPL and LLM on_topic scoring subprocesses (both load their own "
             "models). Use for quick single-datapoint sanity checks where generation + the trace "
             "plot are enough; predictions.jsonl will be missing ppl/llm_on_topic/llm_confidence."
    )
    parser.add_argument("--override", type=str, default="manual",
        choices=['auto', 'manual', 'no'],
        help=(
            "`auto`: override all, "
            "`manual`: ask in terminal before override, "
            "`no`: no override."
        )
    )
    parser.add_argument("--data", type=str, default='wordnet',
        choices=['wordnet', 'wikidata'])

    # Data arguments
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--dev_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--hierarchy_path", type=str, default=None)
    parser.add_argument("--wiki_gold", type=str, default=None)
    parser.add_argument("--num_datapoints", type=int, default=500)
    parser.add_argument("--datapoint_offset", type=int, default=0,
        help="Skip the first N datapoints before starting generation.")
    parser.add_argument("--dataset_split", type=str, default='dev')
    parser.add_argument("--eval_version", type=int, default=0,
        help=(
            "-1: use dataset_split to decide which version to use.\n"
            "-2: use only the context."
        )
    )
    parser.add_argument("--eval_version_split", type=str, default=None,
        choices=['train', 'dev', 'test'],
        help=(
            "Override which split's instruction-version range --eval_version -1 samples from "
            "(train: 0-2, dev: 3-5, test: 6-34), independent of --dataset_split which still controls "
            "which data file/rows are loaded. Default: None, i.e. fall back to --dataset_split."
        )
    )
    parser.add_argument("--instructions_csv", type=str, default="instructions.csv",
        help="Filename (in prompts/instructions/) of the instruction template CSV to use, e.g. 'instructions_broad.csv'."
    )

    # Generation model arguments
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--model_type", type=str, default="ar",
        choices=["ar", "dlm", "diffusion_gemma"],
        help="ar: autoregressive LLM (default); dlm: diffusion language model; diffusion_gemma: DiffusionGemma via API server"
    )
    parser.add_argument("--quant", type=str, default="int4",
        choices=["int4", "none"],
        help="Quantization for DiffusionGemma (int4=NF4 4-bit, fits V100-32GB; none=bf16, needs >48GB)."
    )
    parser.add_argument("--diffusion_gemma_server", type=str, default="http://127.0.0.1:8765",
        help="URL of the DiffusionGemma generation server (auto-started as a subprocess when model_type=diffusion_gemma)."
    )
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_gen_length", type=int, default=60)
    parser.add_argument("--cfg_scale", type=float, default=None,
        help="Signed CFG scale: logits = base + scale*(cond_pos - cond_neg). Positive guides toward cond_pos, negative away. "
             "Use --legacy_cfg for token-masking CFG instead. None (default) means no CFG; 0.0 is a valid scale."
    )
    parser.add_argument("--legacy_cfg", action="store_true", default=False,
        help="Use token-masking CFG (original dllm behavior) instead of extra-prompt CFG. cfg_scale must be > 0."
    )
    _pt_choices = sorted(VALID_PROMPT_TYPES)
    parser.add_argument("--base_prompt", type=str, default=None, choices=_pt_choices,
        help="Content type for the base prompt. One of: " + ", ".join(_pt_choices) + "."
    )
    parser.add_argument("--cond_pos_prompt", type=str, default=None, choices=_pt_choices,
        help="Content type for the positive conditional prompt (cond_pos in CFG formula). "
             "None = no CFG (or legacy_cfg only)."
    )
    parser.add_argument("--cond_neg_prompt", type=str, default=None, choices=_pt_choices,
        help="Content type for the negative conditional prompt (cond_neg in CFG formula). "
             "None = use base logits as cond_neg, recovering the old formula base + scale*(cond_pos - base)."
    )
    parser.add_argument("--steps", type=int, default=128,
        help="Diffusion denoising steps for DLMs."
    )
    parser.add_argument("--remasking", type=str, default="low_confidence",
        choices=["low_confidence", "random"],
        help="Token selection strategy per diffusion step."
    )
    parser.add_argument("--block_size", type=int, default=128,
        help="Block size for blockwise diffusion scheduling (LLaDA) / block_length for Nemotron."
    )
    parser.add_argument("--threshold", type=float, default=None,
        help="Confidence threshold for Nemotron DLM unmasking (None = fixed top-k; 0.9 = model card default)."
    )
    parser.add_argument("--conditional_diff_threshold", type=float, default=None,
        help="Suppress tokens where log_softmax(cond) - log_softmax(base) > threshold. "
             "Requires --cfg_scale != 0. None = disabled."
    )
    parser.add_argument("--logical_guidance", action="store_true", default=False,
        help="Dynamic-scale NOT-gate guidance in probability space. Replaces the fixed cfg_scale formula "
             "with a scale driven by the running posterior log-prob (--log_p_prior). Requires --cfg_scale != 0."
    )
    parser.add_argument("--log_p_prior", type=float, default=0.0,
        help="Initial log-prob for the logical guidance posterior (default: 0.0, i.e. p=1.0). "
             "Only used when --logical_guidance is set."
    )
    parser.add_argument("--logical_guidance_tau", type=float, default=1.0,
        help="Scale factor applied to each log_p_c update (default: 1.0). Only used when --logical_guidance is set."
    )
    parser.add_argument("--logical_guidance_delta", type=float, default=0.0,
        help="Total cumulative shift added to log_p_c across all committed tokens (default: 0.0). "
             "Each token commit adds delta/gen_len. Only used when --logical_guidance is set."
    )
    parser.add_argument("--logical_guidance_pre_temp", type=float, default=1.0,
        help="Temperature dividing the base/cond_pos/cond_neg logits before the three softmaxes "
             "in the NOT-gate combination (sharpens each distribution if <1, flattens if >1). "
             "Default 1.0 is a no-op. Distinct from --temperature, which only affects Gumbel "
             "noise added after this combination, at token-selection time. Only used when "
             "--logical_guidance is set."
    )
    parser.add_argument("--trace_topk", type=int, default=20,
        help="Number of top candidate tokens (with probabilities) recorded per prompt-role "
             "('base'/'cond_pos'/'cond_neg') at each decoded position in generation_trace.jsonl, "
             "taken from each role's own raw softmax distribution before CFG combination. No "
             "dedup: if cond_neg_prompt_ids is None, 'cond_neg' duplicates 'base'. <=0 disables "
             "and omits topk from the trace."
    )
    parser.add_argument("--apply_chat_template", action="store_true", default=False,
        help="Wrap the context in the tokenizer's chat template (for instruction-tuned models)."
    )
    parser.add_argument("--enable_thinking", action="store_true", default=False,
        help="Allow reasoning models to emit <think> traces (default: off — cleaner benchmark output)."
    )
    parser.add_argument("--continuation_suffix", action="store_true", default=False,
        help="Append a suffix before the chat template to steer instruct models toward direct continuation."
    )
    parser.add_argument("--continuation_suffix_text", type=str, default=None,
        help=f"Override the default suffix text (default: '{DEFAULT_CONTINUATION_SUFFIX}')."
    )
    parser.add_argument("--no_copy_suffix", action="store_true", default=False,
        help="Append a no-copy instruction to the suffix."
    )
    parser.add_argument("--no_copy_suffix_text", type=str, default=None,
        help=f"Override the no-copy suffix text (default: '{DEFAULT_NO_COPY_SUFFIX}')."
    )
    parser.add_argument("--continuation_prefix", action="store_true", default=False,
        help="Append a format-priming prefix (e.g. '==') to the prompt so the model starts mid-block."
    )
    parser.add_argument("--continuation_prefix_text", type=str, default=None,
        help=f"Override the prefix text (default: '{DEFAULT_CONTINUATION_PREFIX}')."
    )

    if config is not None and isinstance(config, dict):
        parser.set_defaults(**config)
    args = parser.parse_args()
    return args


def main():
    """
    python -m src_llm_dlm.main --name RUN_NAME [--model_type ar|dlm]
    """
    args = get_argparse()

    seed = args.seed if args.seed is not None else 0
    # Seed all RNGs from --seed (added 2026-07-16). Before this date only torch was seeded
    # per-datapoint; random was hardcoded to 0. Per-datapoint torch seeding still happens
    # inside run() (seed + idx), overriding these globals for each datapoint.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    exp_manager = ExperimentManager(
        name=args.name,
        run_dir=args.run_dir,
        override=True,
        num_runs=1,
        seed_offset=seed,
        metric_tracker=MetricTracker(),
    )
    exp_manager.save_command()
    run(args, run_id=0, exp_manager=exp_manager)


if __name__ == '__main__':
    main()

