"""
Extra-prompt CFG sampler for diffusion language models.

Extends MDLMSampler with classifier-free guidance driven by a separate
guidance prompt, rather than by masking the base prompt tokens.

Guidance modes (selected automatically based on which args are provided):
  - Extra-prompt CFG: unconditional = model(base_prompt | masks),
    conditional = model(guidance_prompt | masks) — base prompt absent from
    the conditional pass. Signed cfg_scale drives direction: positive guides
    toward the guidance prompt, negative guides away from it.
  - Token-masking CFG (legacy): unconditional replaces original prompt tokens
    with mask tokens. Activated when cfg_scale > 0 and no cond_pos_prompt_ids.
  - No CFG: plain forward pass.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src_llm_dlm.dllm.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from src_llm_dlm.dllm.utils import add_gumbel_noise, get_num_transfer_tokens
from src_llm_dlm.trace import StepTrace, TokenDecodeEvent


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float | None = None
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False


@dataclass
class PromptCFGSamplerConfig(MDLMSamplerConfig):
    cond_pos_prompt_ids: torch.Tensor | None = None  # 1-D cond_pos; sign of cfg_scale sets direction
    cond_neg_prompt_ids: torch.Tensor | None = None  # 1-D cond_neg; None → use base logits as cond_neg (old formula)
    conditional_diff_threshold: float | None = None  # suppress tokens where log_softmax(cond_pos) - log_softmax(cond_neg) > threshold
    logical_guidance: bool = False  # dynamic-scale NOT-gate in probability space (replaces fixed cfg_scale)
    log_p_prior: float = 0.0  # initial log-prob for the logical guidance posterior
    logical_guidance_tau: float = 1.0  # scale factor applied to each log_p_c update
    logical_guidance_delta: float = 0.0  # total cumulative shift added to log_p_c over all committed tokens
    logical_guidance_pre_temp: float = 1.0  # temperature dividing base/cond_pos/cond_neg logits before the NOT-gate softmaxes (1.0 = no-op)
    trace_topk: int | None = 20  # top-k candidates (per prompt-role) recorded at each decoded position in the trace; <=0 or None disables


@dataclass
class CFGLogitsOutput:
    logits: torch.Tensor
    norm_diff_all: torch.Tensor | None = None
    base_logits: torch.Tensor | None = None      # raw base-prompt logits, pre-CFG-combination
    cond_pos_logits: torch.Tensor | None = None  # raw cond_pos logits, pre-CFG-combination (None if cond_pos_prompt_ids is None)
    cond_neg_logits: torch.Tensor | None = None  # raw cond_neg logits, pre-CFG-combination (None only when cond_pos_prompt_ids is also None, i.e. no CFG at all; falls back to the *same tensor object* as base_logits, not a separate forward pass, when cond_neg_prompt_ids specifically is None but cond_pos_prompt_ids is set)


@dataclass
class LogPCState:
    value: torch.Tensor          # [B] float32, on model device
    tau: float = 1.0
    delta_per_token: float | torch.Tensor = 0.0  # scalar or [B] (infill uses per-item counts)

    def update(self, norm_diff_at_x0: torch.Tensor, transfer_index: torch.Tensor) -> None:
        # norm_diff_at_x0: [B, T]  transfer_index: [B, T] bool
        delta = (norm_diff_at_x0 * transfer_index).sum(dim=-1)          # [B]
        n_committed = transfer_index.float().sum(dim=-1)                 # [B]
        self.value = self.value + self.tau * delta + self.delta_per_token * n_committed


def _topk_by_role(
    base_logits: torch.Tensor,
    cond_pos_logits: torch.Tensor | None,
    cond_neg_logits: torch.Tensor | None,
    j: int,
    pos: int,
    k: int,
) -> dict[str, list[tuple[int, float]]]:
    """Top-k (token_id, prob) at (j, pos) for each role whose logits were actually computed.

    No dedup: if cond_neg falls back to base logits (cond_neg_prompt_ids is None), it's the
    same tensor object as base_logits and will report identical values under its own "cond_neg"
    key — accepted redundancy in exchange for not tracking prompt-content equality.
    """
    result: dict[str, list[tuple[int, float]]] = {}
    for role, t in (("base", base_logits), ("cond_pos", cond_pos_logits), ("cond_neg", cond_neg_logits)):
        if t is None:
            continue
        probs = F.softmax(t[j, pos].float(), dim=-1)
        vals, idx = probs.topk(min(k, probs.shape[-1]))
        result[role] = list(zip(idx.tolist(), vals.tolist()))
    return result


@dataclass
class PromptCFGSampler(BaseSampler):

    def _compute_logits(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        B: int,
        prompt_lens: list[int],
        cfg_scale: float | None,
        cond_pos_prompt_ids: torch.Tensor | None,
        cond_neg_prompt_ids: torch.Tensor | None,
        unmasked_index: torch.Tensor,
        logical_guidance: bool = False,
        log_p_c: LogPCState | None = None,
        conditional_diff_threshold: float | None = None,
        logical_guidance_pre_temp: float = 1.0,
    ) -> CFGLogitsOutput:
        logits = self.model(x, attention_mask=attention_mask).logits
        # Cloned so it stays valid regardless of what happens to `logits` downstream
        base_logits = logits.clone()

        cond_pos, cond_neg = None, None
        if cond_pos_prompt_ids is not None:
            cond_pos = self._run_cond_forward(x, [cond_pos_prompt_ids] * B, attention_mask, prompt_lens)
            if cond_neg_prompt_ids is not None:
                cond_neg = self._run_cond_forward(x, [cond_neg_prompt_ids] * B, attention_mask, prompt_lens)
            else:
                cond_neg = logits  # fallback: cond_neg == base → recovers old formula

        norm_diff_all = None
        if cond_pos is not None and cond_neg is not None:
            norm_diff_all = (
                F.log_softmax(cond_pos.float(), dim=-1) -
                F.log_softmax(cond_neg.float(), dim=-1)
            )

        do_cfg = cfg_scale is not None and cfg_scale != 0.0
        # ── Extra-prompt CFG ─────────────────────────────────────────
        if cond_pos_prompt_ids is not None and do_cfg:
            if logical_guidance:
                orig_dtype = logits.dtype
                p_c = torch.sigmoid(log_p_c.value).view(B, 1, 1)
                scale = p_c / (1.0 - p_c + 1e-8)
                inv_temp = 1.0 / logical_guidance_pre_temp
                p_base = F.softmax(logits.float() * inv_temp, dim=-1)
                p_cond_pos = F.softmax(cond_pos.float() * inv_temp, dim=-1)
                p_cond_neg = F.softmax(cond_neg.float() * inv_temp, dim=-1)
                p_combined = (p_base - scale * (p_cond_pos - p_cond_neg)).clamp(min=0)
                logits = torch.log(p_combined + 1e-8).to(orig_dtype)
            else:
                logits = logits + cfg_scale * (cond_pos - cond_neg)

            if conditional_diff_threshold is not None:
                logits = logits.masked_fill(norm_diff_all > conditional_diff_threshold, float('-inf'))

        # ── Token-masking CFG (legacy) ───────────────────────────────
        elif do_cfg:
            mask_id = self.tokenizer.mask_token_id
            un_x = x.clone()
            un_x[unmasked_index] = mask_id
            un_logits = self.model(un_x, attention_mask=attention_mask).logits
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        return CFGLogitsOutput(
            logits=logits, norm_diff_all=norm_diff_all,
            base_logits=base_logits, cond_pos_logits=cond_pos, cond_neg_logits=cond_neg,
        )

    @staticmethod
    def _compute_x0_confidence(logits: torch.Tensor, x0: torch.Tensor, remasking: str) -> torch.Tensor:
        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            return torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            return torch.rand(x0.shape, device=x0.device)
        else:
            raise NotImplementedError(remasking)

    def _run_cond_forward(
        self,
        x: torch.Tensor,                        # [B, T]
        replacement_prompts: list[torch.Tensor], # per-item 1-D tensors on model device
        attention_mask: torch.Tensor,            # [B, T]
        prompt_lens: list[int],
    ) -> torch.Tensor:                           # [B, T, vocab_size]
        """Conditional forward pass: item i sees [replacement_prompts[i] | gen_tokens_i].

        The base prompt is excluded; replacement_prompts[i] provides the full
        context. For extra-prefix behaviour (guidance prepended to base prompt),
        pass [guidance | base_prompt_i] as the replacement for item i.

        Returns logits in x's coordinate space; prompt positions are zero
        (they are never masked and are not used by the token-selection logic).
        """
        B, T = x.shape
        V = self.model.config.vocab_size
        out = torch.zeros(B, T, V, device=x.device, dtype=torch.float32)
        for i in range(B):
            rp = replacement_prompts[i].unsqueeze(0)                        # [1, P_i]
            P_i = rp.shape[1]
            pl = prompt_lens[i]
            gen_end = int(attention_mask[i].sum().item())
            gen_tokens = x[i, pl:gen_end].unsqueeze(0)                     # [1, gen_len]
            x_repl = torch.cat([rp, gen_tokens], dim=1)                    # [1, P_i+gen_len]
            attn_repl = torch.ones(
                1, P_i + gen_end - pl, dtype=attention_mask.dtype, device=x.device
            )
            out[i, pl:gen_end] = self.model(
                x_repl, attention_mask=attn_repl
            ).logits[0, P_i:]
        return out

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: PromptCFGSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = PromptCFGSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        cond_pos_prompt_ids = kwargs.get("cond_pos_prompt_ids", config.cond_pos_prompt_ids)
        cond_neg_prompt_ids = kwargs.get("cond_neg_prompt_ids", config.cond_neg_prompt_ids)
        conditional_diff_threshold = kwargs.get("conditional_diff_threshold", config.conditional_diff_threshold)
        logical_guidance = kwargs.get("logical_guidance", config.logical_guidance)
        log_p_prior = kwargs.get("log_p_prior", config.log_p_prior)
        logical_guidance_tau = kwargs.get("logical_guidance_tau", config.logical_guidance_tau)
        logical_guidance_delta = kwargs.get("logical_guidance_delta", config.logical_guidance_delta)
        logical_guidance_pre_temp = kwargs.get("logical_guidance_pre_temp", config.logical_guidance_pre_temp)
        trace_topk = kwargs.get("trace_topk", config.trace_topk)

        if cond_pos_prompt_ids is not None:
            cond_pos_prompt_ids = cond_pos_prompt_ids.to(self.model.device)
        if cond_neg_prompt_ids is not None:
            cond_neg_prompt_ids = cond_neg_prompt_ids.to(self.model.device)

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (used only for token-masking CFG fallback)
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        step_mask_entropies = []
        
        delta_per_token = logical_guidance_delta / max(max_new_tokens, 1)
        log_p_c = LogPCState(
            value=torch.full((B,), log_p_prior, device=self.model.device, dtype=torch.float32),
            tau=logical_guidance_tau,
            delta_per_token=delta_per_token,
        )
        global_step = 0
        _step_traces: list[list] = [[] for _ in range(B)]

        for b in range(num_blocks):
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                mask_index = x == mask_id

                log_p_c_snap = log_p_c.value.tolist()
                gen_ids_snap = [
                    x[j, prompt_lens[j]:prompt_lens[j] + max_new_tokens].tolist()
                    for j in range(B)
                ]

                cfg_out = self._compute_logits(
                    x, attention_mask, B, prompt_lens,
                    cfg_scale, cond_pos_prompt_ids, cond_neg_prompt_ids, unmasked_index,
                    logical_guidance, log_p_c, conditional_diff_threshold,
                    logical_guidance_pre_temp,
                )
                logits = cfg_out.logits
                norm_diff_all = cfg_out.norm_diff_all

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # ent = Shannon entropy (bits) of the full predicted distribution at every
                # position; dist_ppl_all = 2**ent = Π_v p_v^(-p_v), the perplexity of that
                # distribution itself (independent of which token ends up chosen there).
                p_ent = F.softmax(logits, dim=-1)
                ent = -(p_ent * torch.log2(p_ent + 1e-10)).sum(dim=-1)
                dist_ppl_all = 2.0 ** ent
                if mask_index.any():
                    step_mask_entropies.append(ent[mask_index].mean().item())

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # x0_p is the confidence score used to choose which tokens to unmask next.
                # low_confidence: re-mask tokens where the model is least sure (highest mask prob).
                # random: re-mask uniformly at random.
                x0_p = self._compute_x0_confidence(logits, x0, remasking)

                # token_prob_all is the model's true softmax confidence in the chosen token,
                # independent of `remasking` strategy (x0_p above is random noise when
                # remasking="random"). Logged per decoded token for trace analysis.
                token_prob_all = p_ent.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

                for j in range(B):
                    _step_traces[j].append(StepTrace(
                        step=global_step,
                        block=b,
                        block_step=i,
                        log_p_c=log_p_c_snap[j],
                        gen_token_ids=gen_ids_snap[j],
                        decoded_events=[
                            TokenDecodeEvent(
                                position=pos - prompt_lens[j],
                                token_id=x0[j, pos].item(),
                                norm_diff=(
                                    norm_diff_all[j, pos, x0[j, pos].item()].item()
                                    if norm_diff_all is not None else None
                                ),
                                token_prob=token_prob_all[j, pos].item(),
                                token_ppl=dist_ppl_all[j, pos].item(),
                                topk=(
                                    _topk_by_role(
                                        cfg_out.base_logits, cfg_out.cond_pos_logits, cfg_out.cond_neg_logits,
                                        j, pos, trace_topk,
                                    )
                                    if trace_topk and trace_topk > 0 else None
                                ),
                            )
                            for pos in transfer_index[j].nonzero(as_tuple=True)[0].tolist()
                        ],
                    ))
                global_step += 1

                if norm_diff_all is not None:
                    norm_diff_at_x0 = norm_diff_all.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
                    log_p_c.update(norm_diff_at_x0, transfer_index)

                if histories is not None:
                    histories.append(x.clone())

        self.last_mask_entropy = float(np.mean(step_mask_entropies)) if step_mask_entropies else None
        self.last_step_traces = _step_traces

        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        cond_pos_prompt_ids = kwargs.get("cond_pos_prompt_ids", getattr(config, "cond_pos_prompt_ids", None))
        cond_neg_prompt_ids = kwargs.get("cond_neg_prompt_ids", getattr(config, "cond_neg_prompt_ids", None))
        conditional_diff_threshold = kwargs.get("conditional_diff_threshold", getattr(config, "conditional_diff_threshold", None))
        logical_guidance = kwargs.get("logical_guidance", getattr(config, "logical_guidance", False))
        log_p_prior = kwargs.get("log_p_prior", getattr(config, "log_p_prior", 0.0))
        logical_guidance_tau = kwargs.get("logical_guidance_tau", getattr(config, "logical_guidance_tau", 1.0))
        logical_guidance_delta = kwargs.get("logical_guidance_delta", getattr(config, "logical_guidance_delta", 0.0))
        logical_guidance_pre_temp = kwargs.get("logical_guidance_pre_temp", getattr(config, "logical_guidance_pre_temp", 1.0))

        if cond_pos_prompt_ids is not None:
            cond_pos_prompt_ids = cond_pos_prompt_ids.to(self.model.device)
        if cond_neg_prompt_ids is not None:
            cond_neg_prompt_ids = cond_neg_prompt_ids.to(self.model.device)

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        mask_counts = torch.tensor(
            [(t == mask_id).sum().item() for t in inputs], dtype=torch.float32, device=self.model.device
        ).clamp(min=1)
        log_p_c = LogPCState(
            value=torch.full((B,), log_p_prior, device=self.model.device, dtype=torch.float32),
            tau=logical_guidance_tau,
            delta_per_token=logical_guidance_delta / mask_counts,  # [B]
        )
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                cfg_out = self._compute_logits(
                    x, attention_mask, B, [0] * B,
                    cfg_scale, cond_pos_prompt_ids, cond_neg_prompt_ids, unmasked_index,
                    logical_guidance, log_p_c, conditional_diff_threshold,
                    logical_guidance_pre_temp,
                )
                logits = cfg_out.logits
                norm_diff_all = cfg_out.norm_diff_all

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                x0_p = self._compute_x0_confidence(logits, x0, remasking)

                for j in range(B):
                    end_j = start + widths[j]
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                x[transfer_index] = x0[transfer_index]

                if norm_diff_all is not None:
                    norm_diff_at_x0 = norm_diff_all.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
                    log_p_c.update(norm_diff_at_x0, transfer_index)

                if histories is not None:
                    histories.append(x.clone())

        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)
