# Speed Viterbi - Rust 实现的 Viterbi 解码

这是一个使用 Rust 和 PyO3 实现的高性能 Viterbi 解码库，用于语音识别中的对齐任务。

## 构建

### 前置要求

- Rust (最新稳定版)
- Python 3.8+
- maturin (用于构建 Python 扩展)

### 安装 maturin

```bash
pip install maturin
```

或者使用 cargo:

```bash
cargo install maturin
```

### 构建和安装

在 `speed-viterbi` 目录下运行:

```bash
# 开发模式安装（推荐用于开发）
maturin develop

# 或者发布模式安装
maturin develop --release
```

## 使用方法

```python
import numpy as np
import speed_viterbi

# 准备数据
B = 2      # 批次大小
T_max = 20 # 最大时间步长
V = 10     # 词汇表大小
U_max = 15 # 最大序列长度

log_probs_batch = np.random.randn(B, T_max, V).astype(np.float32)
y_batch = np.random.randint(0, V, (B, U_max)).astype(np.int64)
T_batch = np.random.randint(5, T_max + 1, (B,)).astype(np.int64)
U_batch = np.random.randint(3, U_max + 1, (B,)).astype(np.int64)

# 执行 Viterbi 解码
alignments = speed_viterbi.viterbi_decoding(
    log_probs_batch,
    y_batch,
    T_batch,
    U_batch
)

# 或者获取性能统计
result = speed_viterbi.viterbi_decoding(
    log_probs_batch,
    y_batch,
    T_batch,
    U_batch,
    return_timings=True
)
alignments = result["alignments"]
timings = result["timings"]
```

## API

### `viterbi_decoding`

执行 Viterbi 解码算法。

**参数:**
- `log_probs_batch`: numpy array, shape `(B, T_max, V)`, 对数概率矩阵
- `y_batch`: numpy array, shape `(B, U_max)`, token IDs（包括 blank）
- `t_batch`: numpy array, shape `(B,)`, 每个样本的实际时间步长
- `u_batch`: numpy array, shape `(B,)`, 每个样本的实际序列长度
- `padding_value`: float, 可选, 默认 `-3.4e38`, padding 值
- `return_timings`: bool, 可选, 默认 `False`, 是否返回性能统计

**返回:**
- 如果 `return_timings=False`: 返回对齐结果列表 `List[List[int]]`
- 如果 `return_timings=True`: 返回字典，包含 `alignments` 和 `timings`

## 测试

运行测试脚本:

```bash
python test_viterbi.py
```

## 性能

Rust 实现相比纯 Python/PyTorch 实现应该具有更好的性能，特别是在大规模数据处理时。

## 实现细节

- 使用 `ndarray` 进行高效的数组操作
- 使用 `PyO3` 提供 Python 绑定
- 支持 NumPy 数组作为输入，无需数据转换开销

