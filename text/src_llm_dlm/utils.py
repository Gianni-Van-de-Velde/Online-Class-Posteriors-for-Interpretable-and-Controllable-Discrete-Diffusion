import re
import json
import random
from collections import defaultdict

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)


def remove_parenthesis(s):
    m = re.search(r'(.*) \(.*\)', s)
    if m is None:
        return s
    else:
        return m.group(1)


def cleanup_generated_text(text, continuation_prefix=None):
    # Strip <think>...</think> blocks emitted by reasoning models before taking first line.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    sents = text.strip().split('\n')
    text = sents[0]
    # We add this prefix (as if the model generated it) when we primed the model with == to make sure it generates something sensible. 
    # Mostly for prettier prints or results.
    if continuation_prefix:
        text = continuation_prefix + text
    return text


def get_hierarchy(path):
    with open(path) as f:
        hierarchy_ = json.load(f)
    hierarchy = dict()
    for topic, leafs in hierarchy_.items():
        new_topic = topic.replace('_', ' ')
        new_leafs = [l.replace('_', ' ') for l in leafs]
        hierarchy[new_topic] = new_leafs
    return hierarchy


def get_data(args, randomize=False, hierarchy_only=False):
    hierarchy = get_hierarchy(args.hierarchy_path)

    if hierarchy_only:
        return hierarchy

    datasets = defaultdict(list)
    data_paths = {
        'train': args.train_path,
        'dev': args.dev_path,
        'test': args.test_path,
    }
    for dataset_split, dataset_path in data_paths.items():
        if dataset_path is not None:
            with open(dataset_path) as f:
                for line in f:
                    datapoint = json.loads(line.strip())

                    topic = datapoint['topic']
                    constraint = datapoint['constraint']
                    if (
                        constraint not in hierarchy[topic] or
                        constraint == topic or
                        constraint not in hierarchy
                    ):
                        continue

                    datasets[dataset_split].append(datapoint)
        if randomize:
            random.shuffle(datasets[dataset_split])

        num_datapoints = getattr(args, dataset_split + '_num_datapoints', 100000000)
        datasets[dataset_split] = datasets[dataset_split][:num_datapoints]
    return datasets, hierarchy, None


def get_lm(args):
    model_name = args.model_name
    model_type = getattr(args, 'model_type', 'ar')

    if model_type == 'dlm':
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )
        except Exception:
            # Custom model not registered under AutoModelForCausalLM (e.g. Nemotron);
            # fall back to AutoModel with trust_remote_code.
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )

    model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f'{"DLM" if model_type == "dlm" else "Model"} "{model_name}" loaded.')
    return model


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # DLMs need a mask_token; LLaDA uses <|mdm_mask|> but doesn't register it by default.
    if getattr(args, 'model_type', 'ar') == 'dlm' and tokenizer.mask_token_id is None:
        vocab = tokenizer.get_vocab()
        for candidate in ('<|mdm_mask|>', '<|mask|>', '[MASK]'):
            if candidate in vocab:
                tokenizer.add_special_tokens({'mask_token': candidate})
                break
    return tokenizer


