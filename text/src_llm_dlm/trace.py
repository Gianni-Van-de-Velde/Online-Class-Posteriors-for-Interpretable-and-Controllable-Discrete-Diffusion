from dataclasses import dataclass


@dataclass
class TokenDecodeEvent:
    """A single token committed (unmasked) during one diffusion step."""
    position: int          # 0-indexed within the gen span
    token_id: int
    norm_diff: float | None   # log_softmax(cond_pos) - log_softmax(cond_neg) at (pos, token); None if no CFG
    token_prob: float      # softmax(logits)[token_id] at (pos, token) under the final combined
                            # (post-CFG, pre-gumbel-noise) logits used to pick this token — the
                            # model's own confidence at the moment of decoding. Independent of
                            # `remasking` strategy (unlike the x0_p used for unmask-order selection,
                            # which is random noise when remasking="random").
    token_ppl: float       # perplexity of the full predicted distribution at (pos), i.e.
                            # 2**H where H = Shannon entropy (bits) of softmax(logits) at that
                            # position = Π_v p_v^(-p_v). Independent of which token was actually
                            # chosen there (unlike token_prob) — measures how spread-out the
                            # model's guess was at that position, not how surprised it was by
                            # the realized token.
    topk: dict[str, list[tuple[int, float]]] | None  # per-prompt-role top-k (token_id, prob) at
                            # (pos), from each role's raw softmax distribution BEFORE CFG
                            # combination. Keys are whichever of "base"/"cond_pos"/"cond_neg" had
                            # logits computed this step (no dedup — if cond_neg falls back to base
                            # logits because cond_neg_prompt_ids is None, "cond_neg" duplicates
                            # "base"). None when trace_topk <= 0 disabled capture.


@dataclass
class StepTrace:
    """Snapshot of the diffusion state entering one denoising step."""
    step: int              # global step index across all blocks
    block: int
    block_step: int
    log_p_c: float         # per-item posterior log-prob used as guidance input this step
    gen_token_ids: list[int]                # gen-span token IDs *before* this step's transfers
    decoded_events: list[TokenDecodeEvent]  # tokens committed during this step
