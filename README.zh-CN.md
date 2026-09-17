[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-inductor-full-dtype-guard

针对 `torch.compile(backend="inductor")` 的一个真实正确性 bug 的调用点绕过方案与诊断工具：当 `fill_value` 是**符号化（被追踪）标量**时（例如跨越重新编译的闭包整数，或来自 `.item()` 的值），`torch.full(size, fill_value, dtype=X)` 会悄悄丢弃 `dtype` 类型转换。Eager 模式会在下游 `.sum()` 之前将值为 `3` 的 `bool` 填充值饱和为 `True`；而 Inductor 保留了未转换的原始值，从而在没有任何报错、警告或非有限值标记的情况下产生一个悄然错误的结果。上游参考：[pytorch/pytorch#194062](https://github.com/pytorch/pytorch/issues/194062)（截至撰写时仍处于 open 状态）。

```python
import torch

def f(x):
    return torch.full((2,), x.item(), dtype=torch.bool).sum()

print(f(torch.tensor(3)))                                # eager:    tensor(2)
print(torch.compile(f, fullgraph=True)(torch.tensor(3)))  # inductor: tensor(6)
```

同一个缺失的类型转换还带来第二个症状——一个被悄悄跳过的安全检查：`torch.full(size, 300, dtype=torch.int8)` 在 eager 模式下对于符号化填充值会抛出 `RuntimeError: value cannot be converted to type int8_t without overflow`，但在 `torch.compile(backend="inductor")` 下同样的调用会悄悄成功，返回一个毫无意义的、发生环绕（wrap-around）的 int8 值。

## 安装与检查

需要 Python 3.9+ 以及兼容的 PyTorch 安装（可选依赖项中的 `torch>=2.0`）。

```bash
git clone https://github.com/zhuhroscar-tech/torch-inductor-full-dtype-guard.git
cd torch-inductor-full-dtype-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-inductor-full-dtype-guard
torch-inductor-full-dtype-guard --json
```

CLI 会针对当前安装的 torch 版本，在多个 bool 填充值（0、1、2、3、7）以及 int8 溢出填充值（300、-200、1000）上重新执行差异检测，并验证 `safe_full` 防护函数在每个用例中是否与 eager 结果一致。JSON 输出包含已安装的 torch 版本、各用例结果、`any_bool_fill_divergence`、`any_int8_silent_overflow` 以及 `guard_fully_correct`。

退出码描述的是**防护检查**的结果，而不仅仅是原生 bug 的检测结果：`0` 表示所有防护用例均与 eager 一致，`1` 表示某个防护检查失败，`2` 表示无法导入 torch。

## 在 Python 中使用

```python
from torch_inductor_full_dtype_guard import safe_full

mask = safe_full((2,), x.item(), dtype=torch.bool)  # 与 torch.full(...) 完全一致，即使在 torch.compile 之下
```

`safe_full` 是 `torch.full` 的一个直接替代品，适用于填充值在编译时可能是符号化的调用点。它通过 `torch.compiler.disable` 强制实际的填充调用在 eager 模式下运行——这是在这个唯一的、本身开销很小的分配操作调用点上有意引入的、范围很窄的 Dynamo 计算图中断——从而保证真正的类型转换与溢出检查始终会执行，无论调用方本身是否处于 `torch.compile` 之下，结果都与 eager 完全一致。

## 适用范围与局限性

- 本工具**不会**全局猴子补丁 `torch.full`。请在你自己的调用点显式调用 `safe_full`。
- 该防护机制（`torch.compiler.disable`）会在 `full()` 调用点引入一次计算图中断，从而放弃该操作的融合优化机会。这是一个刻意为之的、以正确性换取融合优化的权衡；如果要在包含大量符号化填充调用的热路径上使用，请先自行做基准测试。
- 本仓库复现并防护的路径是 CPU Inductor；由于本机没有 CUDA，未测试 CUDA/Triton 代码生成路径——原始 issue 的作者也提到 MPS 后端源码"看起来一样"，但在其构建环境中未能运行。
- 常量填充值（追踪时已知的字面 Python int/bool）从未受此 bug 影响；`safe_full` 在该情况下依然与 eager 一致，只是多了一层不必要的开销。
- 诊断样本并不能证明对所有可能的输入都正确；实际行为取决于所安装的 torch 版本。如果未来的 torch 版本在上游修复了 pytorch/pytorch#194062，`any_bool_fill_divergence` 与 `any_int8_silent_overflow` 在该版本上都应报告为 `False`——此时 `safe_full` 仍然是一个安全的、等效于空操作的防护方案。

## 开发

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_inductor_full_dtype_guard
```

参见[实现代码](src/torch_inductor_full_dtype_guard/core.py)、[测试](tests/test_core.py)与 [CI 配置](.github/workflows/ci.yml)。采用 [MIT](LICENSE) 许可证。
