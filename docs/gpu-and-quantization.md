# GPU & Quantization (LoRA vs QLoRA)

LLM Studio detects your GPU and recommends a fine-tuning method and batch plan
that fits your VRAM. You can always override it on the **Configure** tab.

## LoRA vs QLoRA

| | **LoRA** | **QLoRA** |
|--|----------|-----------|
| Base weights | 16-bit (bf16/fp16) | **4-bit** (NF4) |
| VRAM | Higher | **Much lower** |
| Speed | Fastest | Slightly slower |
| Quality | Highest | Very close to LoRA |
| Use when | You have VRAM headroom | VRAM is tight / large models |

Both train small **adapter** matrices and leave the base weights frozen.

## How the recommendation works

1. **Detect** free VRAM on the primary GPU (`pynvml`, falling back to `torch`).
2. **Estimate** peak training memory (conservative heuristic):
   ```
   weights      ≈ params_b × (2.0 GB bf16 | 0.55 GB 4-bit)
   activations  ≈ params_b × seq_len × batch × 1.5e-4 GB   (grad-checkpointed)
   adapter+opt  ≈ ~0.3–0.7 GB (scales with LoRA rank)
   overhead     ≈ ~1.5 GB
   ```
3. **Choose** LoRA if there's ≥ `qlora_threshold_gb` (default 24 GB) free *and*
   LoRA fits the safety budget (`vram_safety_factor`, default 85%); otherwise QLoRA.
4. **Fit** the largest batch size that stays under budget, shortening
   `max_seq_length` if nothing fits, then set gradient accumulation to reach a
   sensible effective batch size.

> The estimate is intentionally cautious — it keeps beginners out of OOM
> territory. Real usage is often a bit lower.

## Rough VRAM guidance (QLoRA, seq 2048)

| Model | Approx. free VRAM to train |
|-------|---------------------------|
| 1–3B | 6–8 GB |
| 7–8B | 10–14 GB |
| 9B (Gemma 2) | 14–18 GB |
| 14B | 18–24 GB |

LoRA (16-bit) needs roughly **3–4×** the weight memory of QLoRA.

## Tuning knobs that affect memory
- **`max_seq_length`** — biggest lever on activation memory. Lower it if you OOM.
- **`per_device_train_batch_size`** — lower to fit; raise `gradient_accumulation_steps` to keep the effective batch.
- **`gradient_checkpointing`** — `"unsloth"` (default) trades compute for big memory savings.
- **`optim="adamw_8bit"`** or `paged_adamw_8bit` — 8-bit optimizer states save memory.
- **LoRA rank (`lora_r`)** — higher rank = more capacity and a little more memory.

## Multi-GPU
Unsloth fine-tuning currently uses a **single GPU**; extra GPUs sit idle. The
recommender warns when it sees more than one. (Multi-GPU is on the roadmap.)

## Policy settings
Adjust in `config/default.yaml` (or via env):
```yaml
gpu:
  vram_safety_factor: 0.85   # fraction of free VRAM used in the fit check
  qlora_threshold_gb: 24.0   # below this free VRAM → recommend QLoRA
```
