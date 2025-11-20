"""
比较并行版本和串行版本的 Viterbi 解码
"""
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speed_viterbi

print("=" * 60)
print("Viterbi 解码：并行 vs 串行性能测试")
print("=" * 60)

# 测试参数
B = 32
V = 5000
T_max = 1000
U_max = 200

print(f"\n测试参数: B={B}, V={V}, T_max={T_max}, U_max={U_max}")
print(f"张量大小: log_probs_batch={B}x{T_max}x{V}, y_batch={B}x{U_max}\n")

# 生成测试数据
np.random.seed(42)
log_probs_batch = np.random.randn(B, T_max, V).astype(np.float32)
y_batch = np.random.randint(0, V, (B, U_max)).astype(np.int64)
T_batch = np.random.randint(T_max // 2, T_max + 1, (B,)).astype(np.int64)
U_batch = np.random.randint(U_max // 2, U_max + 1, (B,)).astype(np.int64)

# 测试并行版本
print("=" * 60)
print("测试 1: 并行版本 (viterbi_decoding)")
print("=" * 60)
start = time.time()
alignments_parallel = speed_viterbi.viterbi_decoding(
    log_probs_batch, y_batch, T_batch, U_batch
)
time_parallel = (time.time() - start) * 1000
print(f"总时间: {time_parallel:.4f} ms")
print(f"对齐结果数量: {len(alignments_parallel)}")
print(f"第一个对齐结果长度: {len(alignments_parallel[0])}")
print(f"第一个对齐结果示例: {alignments_parallel[0][:20].tolist()}")

# 测试串行版本
print("\n" + "=" * 60)
print("测试 2: 串行版本 (viterbi_decoding_serial)")
print("=" * 60)
start = time.time()
alignments_serial = speed_viterbi.viterbi_decoding_serial(
    log_probs_batch, y_batch, T_batch, U_batch
)
time_serial = (time.time() - start) * 1000
print(f"总时间: {time_serial:.4f} ms")
print(f"对齐结果数量: {len(alignments_serial)}")
print(f"第一个对齐结果长度: {len(alignments_serial[0])}")
print(f"第一个对齐结果示例: {alignments_serial[0][:20].tolist()}")

# 验证结果一致性
print("\n" + "=" * 60)
print("结果验证")
print("=" * 60)

all_match = True
for i in range(len(alignments_parallel)):
    # 转换为 list 进行比较
    if not np.array_equal(alignments_parallel[i], alignments_serial[i]):
        all_match = False
        print(f"✗ Batch {i} 不匹配！")
        print(f"  并行: {alignments_parallel[i][:10].tolist()}")
        print(f"  串行: {alignments_serial[i][:10].tolist()}")
        break

if all_match:
    print("✓ 并行版本和串行版本产生完全相同的结果")
else:
    print("✗ 结果不一致！")

# 性能对比
print("\n" + "=" * 60)
print("性能对比")
print("=" * 60)
print(f"并行版本: {time_parallel:.4f} ms")
print(f"串行版本: {time_serial:.4f} ms")
print(f"加速比: {time_serial / time_parallel:.2f}x")

if time_parallel < time_serial:
    print(f"✓ 并行版本快 {time_serial / time_parallel:.2f}x")
else:
    print(f"! 串行版本反而更快（可能是数据规模太小）")

# 小规模测试
print("\n" + "=" * 60)
print("小规模测试 (B=2)")
print("=" * 60)

B_small = 2
log_probs_small = np.random.randn(B_small, T_max, V).astype(np.float32)
y_small = np.random.randint(0, V, (B_small, U_max)).astype(np.int64)
T_small = np.random.randint(T_max // 2, T_max + 1, (B_small,)).astype(np.int64)
U_small = np.random.randint(U_max // 2, U_max + 1, (B_small,)).astype(np.int64)

start = time.time()
result_parallel_small = speed_viterbi.viterbi_decoding(
    log_probs_small, y_small, T_small, U_small
)
time_parallel_small = (time.time() - start) * 1000

start = time.time()
result_serial_small = speed_viterbi.viterbi_decoding_serial(
    log_probs_small, y_small, T_small, U_small
)
time_serial_small = (time.time() - start) * 1000

print(f"并行版本: {time_parallel_small:.4f} ms")
print(f"串行版本: {time_serial_small:.4f} ms")
if time_serial_small < time_parallel_small:
    print(f"✓ 小规模数据串行版本更快 ({time_parallel_small / time_serial_small:.2f}x)")
else:
    print(f"  并行版本仍然更快 ({time_serial_small / time_parallel_small:.2f}x)")

# 建议
print("\n" + "=" * 60)
print("使用建议")
print("=" * 60)
print("• 大批次 (B >= 16): 使用 viterbi_decoding (并行版本)")
print("• 小批次 (B < 16):  使用 viterbi_decoding_serial (串行版本)")
print("• 嵌套并行场景:     使用 viterbi_decoding_serial (避免线程争用)")
print("=" * 60)
