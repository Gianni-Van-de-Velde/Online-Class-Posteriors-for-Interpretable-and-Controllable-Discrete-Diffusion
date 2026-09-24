from src_llm_dlm.diverse_instructions import build_context

DEFAULT_CONTINUATION_SUFFIX = 'Only output the continuation in == ... == format, no other text.'
DEFAULT_NO_COPY_SUFFIX = 'Your continuation must be entirely original.'
DEFAULT_CONTINUATION_PREFIX = '=='

VALID_PROMPT_TYPES = {
    'negative', 'positive', 'no_constraint', 'same_baseline', 'context',
    'pure_guidance_negative', 'pure_guidance_positive',
}


def _apply_chat_template(content, tokenizer, enable_thinking=True):
    msgs = [{'role': 'user', 'content': content}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if not enable_thinking:
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(msgs, **kwargs)


def build_prompt_triple(datapoint, args, eval_version):
    """Select raw content strings for base, cond_pos, and cond_neg.

    Returns (base_content, cond_pos_content | None, cond_neg_content | None).
    None means the slot is unused.
    """
    instructions_csv = getattr(args, 'instructions_csv', None) or 'instructions.csv'
    base = build_context(datapoint, eval_version, args.base_prompt, instructions_csv) if args.base_prompt else None
    cond_pos = build_context(datapoint, eval_version, args.cond_pos_prompt, instructions_csv) if args.cond_pos_prompt else None
    cond_neg = build_context(datapoint, eval_version, args.cond_neg_prompt, instructions_csv) if args.cond_neg_prompt else None
    return base, cond_pos, cond_neg


def decorate(content, args, tokenizer, *, with_suffix=True, with_prefix=True):
    """Apply the decoration pipeline to a raw content string.

    with_suffix=False skips continuation_suffix and no_copy_suffix.
    with_prefix=False skips continuation_prefix.
    Chat template is always gated by args.apply_chat_template.
    """
    if with_suffix and args.continuation_suffix:
        content = content + '\n' + (args.continuation_suffix_text or DEFAULT_CONTINUATION_SUFFIX)
    if with_suffix and args.no_copy_suffix:
        content = content + '\n' + (args.no_copy_suffix_text or DEFAULT_NO_COPY_SUFFIX)
    if args.apply_chat_template:
        content = _apply_chat_template(content, tokenizer, enable_thinking=args.enable_thinking)
    # This is after apply chat template, which may be confusing, but it will never happen at the same time.
    # Though a confusing name, continuation_prefix will prime a base model with == before letting it generate so that it doesn't just end the sentence directly
    # So this is called with base models, while apply chat template is only for instruct models.
    if with_prefix and args.continuation_prefix:
        content = content + '\n' + (args.continuation_prefix_text or DEFAULT_CONTINUATION_PREFIX)
    return content


def build_prompts(datapoint, args, tokenizer, eval_version):
    """Build and decorate all three prompts.

    Returns (base_prompt, cond_pos_prompt | None, cond_neg_prompt | None) — fully decorated, ready to tokenize.
    """
    base_content, cond_pos_content, cond_neg_content = build_prompt_triple(datapoint, args, eval_version)
    base = decorate(base_content, args, tokenizer) if base_content is not None else None
    cond_pos = decorate(cond_pos_content, args, tokenizer) if cond_pos_content is not None else None
    cond_neg = decorate(cond_neg_content, args, tokenizer) if cond_neg_content is not None else None
    return base, cond_pos, cond_neg
