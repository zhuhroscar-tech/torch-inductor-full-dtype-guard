[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-inductor-full-dtype-guard

A call-site workaround and diagnostic for a real `torch.compile(backend="inductor")` correctness bug: `torch.full(size, fill_value, dtype=X)` silently drops the `dtype` cast when `fill_value` is a *symbolic* (traced) scalar -- a closure int across a recompile, or any value from `.item()`. Eager saturates a `bool` fill of `3` to `True` before a downstream `.sum()`; Inductor keeps the raw uncast value, giving a silently wrong result with no error, warning, or non-finite marker. Upstream reference: [pytorch/pytorch#194062](https://github.com/pytorch/pytorch/issues/194062) (open as of this writing).

```python
import torch

def f(x):
    return torch.full((2,), x.item(), dtype=torch.bool).sum()

print(f(torch.tensor(3)))                                # eager:    tensor(2)
print(torch.compile(f, fullgraph=True)(torch.tensor(3)))  # inductor: tensor(6)
```

A second symptom of the same missing cast is a silently-skipped safety check: `torch.full(size, 300, dtype=torch.int8)` raises `RuntimeError: value cannot be converted to type int8_t without overflow` in eager mode for a symbolic fill, but under `torch.compile(backend="inductor")` the same call silently succeeds and returns a meaningless wrapped-around int8 value instead.

## Install and check

Requires Python 3.9+ and a compatible PyTorch installation (`torch>=2.0` in the optional extra).

```bash
git clone https://github.com/zhuhroscar-tech/torch-inductor-full-dtype-guard.git
cd torch-inductor-full-dtype-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-inductor-full-dtype-guard
torch-inductor-full-dtype-guard --json
```

The CLI reruns the divergence checks across several bool-fill values (0, 1, 2, 3, 7) and int8-overflow fill values (300, -200, 1000) against the currently installed torch build, and verifies the `safe_full` guard matches eager in every case. Its JSON includes the installed torch version, per-case results, `any_bool_fill_divergence`, `any_int8_silent_overflow`, and `guard_fully_correct`.

Exit codes describe the **guard check**, not just native bug detection: `0` means every guard case matched eager, `1` means a guard check failed, and `2` means torch could not be imported.

## Use in Python

```python
from torch_inductor_full_dtype_guard import safe_full

mask = safe_full((2,), x.item(), dtype=torch.bool)  # matches torch.full(...) exactly, including under torch.compile
```

`safe_full` is a drop-in replacement for `torch.full` at call sites where the fill value may be symbolic under compilation. It forces the actual fill call to run in eager mode via `torch.compiler.disable` -- a deliberate, narrow Dynamo graph break at this single, already-cheap allocation op -- so the real dtype cast and overflow check always run, matching eager exactly whether or not the caller itself is under `torch.compile`.

## Scope and limitations

- This tool does **not** globally monkey-patch `torch.full`. Call `safe_full` explicitly at your own call sites.
- The guard mechanism (`torch.compiler.disable`) introduces a graph break at the `full()` call site, which forgoes fusion opportunities for that op. This is a deliberate correctness-over-fusion tradeoff; benchmark before using it on a hot path with many symbolic fills.
- CPU Inductor is the reproduced and guarded path on this host. CUDA/Triton codegen was not tested (no CUDA available on this host) -- the issue's author notes the MPS backend source "looks the same" but did not run in their build either.
- The constant-fill case (a literal Python int/bool known at trace time) was never affected by this bug; `safe_full` still matches eager there, it is simply unnecessary overhead in that case.
- Diagnostic samples do not prove correctness for every possible input; behavior depends on the installed torch version. If a future torch release fixes pytorch/pytorch#194062 upstream, `any_bool_fill_divergence` and `any_int8_silent_overflow` should both report `False` on that version -- `safe_full` remains a safe no-op-equivalent guard in that case.

## Development

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_inductor_full_dtype_guard
```

See the [implementation](src/torch_inductor_full_dtype_guard/core.py), [tests](tests/test_core.py), and [CI config](.github/workflows/ci.yml). Licensed under [MIT](LICENSE).
