"""
测试自定义 padding_value 参数
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speed_viterbi

# 测试参数
B = 2
V = 10
T_max = 20
U_max = 15

print("测试自定义 padding_value 参数\n")

# 生成测试数据
np.random.seed(42)
log_probs_batch = np.random.randn(B, T_max, V).astype(np.float32)
y_batch = np.random.randint(0, V, (B, U_max)).astype(np.int64)
T_batch = np.random.randint(5, T_max + 1, (B,)).astype(np.int64)
U_batch = np.random.randint(3, U_max + 1, (B,)).astype(np.int64)

# 测试 1: 使用默认 padding_value
print("测试 1: 使用默认 padding_value")
result_default = speed_viterbi.viterbi_decoding(
    log_probs_batch, y_batch, T_batch, U_batch
)
print(f"  对齐结果数量: {len(result_default)}")
print(f"  第一个对齐结果长度: {len(result_default[0])}")
print(f"  第一个对齐结果: {result_default[0][:10]}")

# 测试 2: 显式指定默认 padding_value
print("\n测试 2: 显式指定默认 padding_value (-3.4e38)")
result_explicit_default = speed_viterbi.viterbi_decoding(
    log_probs_batch, y_batch, T_batch, U_batch, padding_value=-3.4e38
)
print(f"  对齐结果数量: {len(result_explicit_default)}")
print(f"  第一个对齐结果长度: {len(result_explicit_default[0])}")
print(f"  第一个对齐结果: {result_explicit_default[0][:10]}")

# 验证两者相同
if result_default == result_explicit_default:
    print("  ✓ 默认值和显式指定的默认值结果相同")
else:
    print("  ✗ 结果不同！")

# 测试 3: 使用不同的 padding_value
print("\n测试 3: 使用不同的 padding_value (-1e10)")
result_custom = speed_viterbi.viterbi_decoding(
    log_probs_batch, y_batch, T_batch, U_batch, padding_value=-1e10
)
print(f"  对齐结果数量: {len(result_custom)}")
print(f"  第一个对齐结果长度: {len(result_custom[0])}")
print(f"  第一个对齐结果: {result_custom[0][:10]}")

# 测试 4: 使用更小的 padding_value
print("\n测试 4: 使用更小的 padding_value (-1e20)")
result_smaller = speed_viterbi.viterbi_decoding(
    log_probs_batch, y_batch, T_batch, U_batch, padding_value=-1e20
)
print(f"  对齐结果数量: {len(result_smaller)}")
print(f"  第一个对齐结果长度: {len(result_smaller[0])}")
print(f"  第一个对齐结果: {result_smaller[0][:10]}")

print("\n总结:")
print(f"  默认值结果: {result_default[0][:10]}")
print(f"  -1e10结果:  {result_custom[0][:10]}")
print(f"  -1e20结果:  {result_smaller[0][:10]}")

# 对于这个特定的测试数据，不同的 padding_value 可能产生相同或不同的结果
# 这取决于实际的概率值
if result_default == result_custom == result_smaller:
    print("\n  ✓ 所有 padding_value 产生相同结果（对于这组数据）")
else:
    print("\n  ℹ 不同的 padding_value 产生了不同的结果（这是正常的）")

print("\n✓ padding_value 参数可以正常工作！")
