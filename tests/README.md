# Tests

Unit tests for the **pure-logic** layers that don't need a GPU or the training
stack: data schema/normalization/validation, the VRAM recommender, training
config validation, and the event bus.

```bash
pip install -e ".[dev]"
pytest
```

Tests that would require `torch`/`unsloth` (engine, inference, real training) are
intentionally out of scope here — exercise those on the GPU machine.
