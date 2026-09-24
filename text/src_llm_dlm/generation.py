import torch


def run_generation_ar(base_prompt, model, tokenizer, args):
    input_ids = tokenizer.encode(base_prompt, return_tensors='pt').cuda()
    do_sample = args.top_p is not None
    gen_kwargs = dict(max_new_tokens=args.max_gen_length, pad_token_id=tokenizer.pad_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, top_p=args.top_p, temperature=args.temperature)
    output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs)
    return output[0, input_ids.shape[1]:]


def run_generation_dlm(base_prompt, cond_pos_prompt, cond_neg_prompt, sampler, tokenizer, args):
    """DLM generation via PromptCFGSampler with optional extra-prompt or token-masking CFG."""
    prompt_ids = tokenizer.encode(base_prompt, return_tensors='pt').cuda()[0]  # 1D
    cond_pos_ids = (
        tokenizer.encode(cond_pos_prompt, return_tensors='pt').cuda()[0]
        if cond_pos_prompt is not None else None
    )
    cond_neg_ids = (
        tokenizer.encode(cond_neg_prompt, return_tensors='pt').cuda()[0]
        if cond_neg_prompt is not None else None
    )
    output = sampler.sample(
        inputs=[prompt_ids],
        max_new_tokens=args.max_gen_length,
        steps=args.steps,
        temperature=args.temperature,
        remasking=args.remasking,
        block_size=args.block_size,
        cfg_scale=args.cfg_scale,
        cond_pos_prompt_ids=cond_pos_ids,
        cond_neg_prompt_ids=cond_neg_ids,
        conditional_diff_threshold=args.conditional_diff_threshold,
        logical_guidance=args.logical_guidance,
        log_p_prior=args.log_p_prior,
        logical_guidance_tau=args.logical_guidance_tau,
        logical_guidance_delta=args.logical_guidance_delta,
        logical_guidance_pre_temp=args.logical_guidance_pre_temp,
        trace_topk=args.trace_topk,
    )
    return output[0, prompt_ids.shape[0]:], sampler.last_mask_entropy, sampler.last_step_traces


def run_generation_nemotron(base_prompt, model, tokenizer, args):
    """Generation for Nemotron-Labs-Diffusion using its native generate() / ar_generate() APIs."""
    prompt_ids = tokenizer.encode(base_prompt, return_tensors='pt').cuda()
    if args.model_type == 'dlm':
        block_length = args.block_size
        # Nemotron requires max_new_tokens % block_length == 0; round up.
        max_new_tokens = ((args.max_gen_length + block_length - 1) // block_length) * block_length
        out_ids, _ = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            block_length=block_length,
            threshold=getattr(args, 'threshold', None),
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        out_ids, _ = model.ar_generate(
            prompt_ids,
            max_new_tokens=args.max_gen_length,
            eos_token_id=tokenizer.eos_token_id,
        )
    return out_ids[0, prompt_ids.shape[1]:]


def run_generation_nemotron_cfg(base_prompt, cond_pos_prompt, model, tokenizer, args):
    """Nemotron DLM generation with CFG via separate KV caches."""
    from src_llm_dlm.nemotron_cfg.generate import nemotron_generate_cfg
    prompt_ids = tokenizer.encode(base_prompt, return_tensors='pt').cuda()  # [1, P]
    cond_pos_ids = (
        tokenizer.encode(cond_pos_prompt, return_tensors='pt').cuda()[0]  # 1D — nemotron_generate_cfg expects [P_c]
        if cond_pos_prompt is not None else None
    )
    block_length = args.block_size
    max_new_tokens = ((args.max_gen_length + block_length - 1) // block_length) * block_length
    out_ids, _ = nemotron_generate_cfg(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        cfg_scale=args.cfg_scale,
        cond_pos_prompt_ids=cond_pos_ids,
        threshold=getattr(args, 'threshold', None),
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    return out_ids[0, prompt_ids.shape[1]:]
