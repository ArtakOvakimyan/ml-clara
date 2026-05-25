# Patches for apple/ml-clara

These patches adapt [apple/ml-clara](https://github.com/apple/ml-clara) to run with Qwen2 and the Russian SberQUAD dataset.

## Patch list

### 01-ring_attn_utils.patch

**File:** `openrlhf/models/ring_attn_utils.py`

**What:** Wraps all `flash_attn` imports in `try/except ImportError`. Adds stub implementations for `index_first_axis`, `pad_input`, `unpad_input`, `all_gather`, plus `get_ring_attn_group` / `set_ring_attn_group` (imported by `deepspeed.py`).

**Why:** T4 (Turing, sm_75) does not support Flash Attention 2, which requires Ampere (sm_80+). Flash Attention is only used in multi-node ring attention paths, which are never invoked in single-GPU training. Stubs raise `NotImplementedError` if actually called, so correctness is not affected.

---

### 02-modeling_clara.patch

**File:** `openrlhf/models/modeling_clara.py`

Contains four independent changes:

**(a) Attention implementation:** `flash_attention_2` → `sdpa` in the default `CLaRaConfig` and in all `AutoModelForCausalLM.from_pretrained` calls. SDPA is built into PyTorch 2.4 and works on T4.

**(b) Compressor MLP dim:** `compr_mlp_hidden_dim=8096` → `1792`. The original value is tuned for Mistral-7B (hidden_size=4096, ~2× hidden). Qwen2-0.5B has hidden_size=896, so 896×2=1792.

**(c) Removes `enable_thinking=False`:** This parameter is Qwen3/QwQ-specific. Qwen2 chat templates do not accept it and raise `TypeError`. Removed from all `apply_chat_template` calls.

**(d) List-to-string coercion in `_blend_standard_prompt`:** Adds type checks at the start of the function so that `docs`, `query`, `answer` are converted to strings if passed as lists. Needed because our SberQUAD data groups multiple QA pairs per context, which the upstream function doesn't expect.

**Revert for A100:** partially. (a) revert to flash_attention_2. (b) adjust per model (7168 for T-Lite). (c) still needed for any Qwen2-based model. (d) still needed for SberQUAD-formatted data.

---

### 03-train_sft.patch

**File:** `openrlhf/cli/train_sft.py`

**What:** Two changes:
1. Hardcoded `attn_implementation='flash_attention_2'` → `'sdpa'`
2. Removes `scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1}` from the `get_scheduler` call

**Why:** (1) Same as above — T4 doesn't support flash attention. (2) The `min_lr` parameter is not supported by `transformers.get_cosine_schedule_with_warmup` in public transformers ≥4.46. Upstream CLaRa was developed against a pinned internal transformers version that supports it. Removing the kwarg makes the scheduler decay to 0 cosine-style, which is a reasonable default.

---

### 04-actor.patch

**File:** `openrlhf/models/actor.py`

**What:** `attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"` → replaces `"flash_attention_2"` with `"sdpa"`.

**Why:** Same as 01, 03 — T4 compatibility.

---

### 05-deepspeed.patch

**File:** `openrlhf/utils/deepspeed/deepspeed.py`

**What:** Wraps `import transformers.modeling_flash_attention_utils` and the `deterministic_g = True` call in `try/except`.

**Why:** The module path `transformers.modeling_flash_attention_utils` fails to import when flash_attn is not installed. The `deterministic_g` flag is only meaningful when flash_attn is actually being used, so it's safe to silently skip.

---


## Post-training config fix

After Stage 2 training finishes, the checkpoint's `config.json` has `compr_base_model_name` set to the original (Mistral) path. Fix it:

```bash
python -c "
import json
cfg = json.load(open('checkpoints/stage2/config.json'))
cfg['compr_base_model_name'] = 'Qwen/Qwen2-0.5B-Instruct'
json.dump(cfg, open('checkpoints/stage2/config.json', 'w'), indent=2)
"
cp ml-clara/openrlhf/models/modeling_clara.py checkpoints/stage2/
```

This is done automatically by `scripts/fix_config.py`.