"""
Topic/constraint descriptions as the instruction.
"""
import csv
from pathlib import Path


def _make_entry(type_, topic, neg, pos, same_baseline):
    if type_ == 'begin+end':
        return {
            'negative':      topic + "\n{context}\n" + neg,
            'positive':      topic + "\n{context}\n" + pos,
            'no_constraint': topic + "\n{context}",
            'same_baseline': topic + "\n{context}\n" + same_baseline,
        }
    combined_neg = f"{topic} {neg}"
    combined_pos = f"{topic} {pos}"
    combined_sb  = f"{topic} {same_baseline}"
    if type_ == 'begin':
        return {
            'negative':      combined_neg + "\n{context}",
            'positive':      combined_pos + "\n{context}",
            'no_constraint': topic + "\n{context}",
            'same_baseline': combined_sb  + "\n{context}",
        }
    else:  # end
        return {
            'negative':      "{context}\n" + combined_neg,
            'positive':      "{context}\n" + combined_pos,
            'no_constraint': "{context}\n" + topic,
            'same_baseline': "{context}\n" + combined_sb,
        }


_MAPPING_CACHE = {}


def _load_mapping(instructions_csv):
    if instructions_csv not in _MAPPING_CACHE:
        mapping = {}
        with open(Path(__file__).parent.parent / 'prompts' / 'instructions' / instructions_csv, 'r') as f:
            for row in csv.DictReader(f):
                mapping[int(row['index'])] = _make_entry(
                    row['type'], row['topic'],
                    row['negative_constraint'], row['positive_constraint'], row['no_constraint']
                )
        _MAPPING_CACHE[instructions_csv] = mapping
    return _MAPPING_CACHE[instructions_csv]


STANDALONE_NEGATIVE_TEMPLATE = "Do not mention {}."
STANDALONE_POSITIVE_TEMPLATE = "Mention {}."


def build_context(datapoint, version, variant, instructions_csv='instructions.csv'):
    # variant: 'negative', 'positive', 'no_constraint', 'same_baseline', 'context' (raw datapoint context),
    # or 'pure_guidance_negative'/'pure_guidance_positive' (bare standalone constraint phrase, no topic/article context)
    if variant == 'context':
        return datapoint['context']
    if variant == 'pure_guidance_negative':
        return build_standalone_constraint(datapoint, STANDALONE_NEGATIVE_TEMPLATE)
    if variant == 'pure_guidance_positive':
        return build_standalone_constraint(datapoint, STANDALONE_POSITIVE_TEMPLATE)
    topic, constraint = datapoint['topic'], datapoint['constraint']
    mapping = _load_mapping(instructions_csv)
    return mapping[version][variant].format(context=datapoint['context'], topic=topic, constraint=constraint).strip()


def build_standalone_constraint(datapoint, template):
    return template.format(datapoint['constraint'])
