# `core.assistant` — in-app LLM guidance

A local instruct model (Qwen2.5 by default) that helps non-technical users.

```
llm.py                 AssistantLLM — load/unload (resource-managed), VRAM-aware fallback
hyperparam_advisor.py  HyperparameterAdvisor — LLM or heuristic config recommendations
data_assistant.py      DataAssistant — infer field mapping, advise cleaning, synthesize QA
prompts.py             prompt builders + robust JSON extraction
```

## Resource management
- Picks `fallback_model_id` automatically when free VRAM is below
  `assistant.min_vram_gb_for_primary`.
- Unloads after each call unless `keep_resident` is set.
- **Always** unloaded right before training (`JobManager.on_before_train`).

## Graceful degradation
Every advisor has a deterministic **heuristic fallback**, so hyperparameter and
mapping suggestions work even when the assistant is disabled or the training
stack isn't installed.

```python
advice = HyperparameterAdvisor(assistant).advise(ctx)   # → field updates + rationale
suggestion = DataAssistant(assistant).suggest_mapping(cols, rows)
```
