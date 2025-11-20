"""
测试 Rust 实现的 Viterbi 解码
"""
import time
import numpy as np
import sys
import os

# 添加项目路径以便导入 torch_test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import speed_viterbi
    RUST_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 speed_viterbi，请先编译 Rust 模块")
    print("运行: maturin develop")
    RUST_AVAILABLE = False

from torch_test import viterbi_decoding as torch_viterbi_decoding
import torch


def test_viterbi_rust():
    """测试 Rust 版本的 Viterbi 解码"""
    if not RUST_AVAILABLE:
        return

    # 测试参数 - 扩大规模以测试真实场景
    B = 32  # 批次大小
    V = 5000  # 词汇表大小（接近实际 ASR 系统）
    T_max = 1000  # 最大时间步（对应约10-20秒音频）
    U_max = 200  # 最大序列长度

    print(f"测试参数: B={B}, V={V}, T_max={T_max}, U_max={U_max}")
    print(f"张量大小: log_probs_batch={B}x{T_max}x{V}, y_batch={B}x{U_max}")

    # 生成测试数据
    np.random.seed(42)  # 固定随机种子以便复现
    log_probs_batch = np.random.randn(B, T_max, V).astype(np.float32)
    y_batch = np.random.randint(0, V, (B, U_max)).astype(np.int64)
    # 确保有不同长度的序列
    T_batch = np.random.randint(T_max // 2, T_max + 1, (B,)).astype(np.int64)
    U_batch = np.random.randint(U_max // 2, U_max + 1, (B,)).astype(np.int64)
    
    # 测试 Rust 版本
    print("\n=== Rust 版本 ===")
    alignments_rust = speed_viterbi.viterbi_decoding(
        log_probs_batch, y_batch, T_batch, U_batch
    )
    
    print(f"对齐结果数量: {len(alignments_rust)}")
    if len(alignments_rust) > 0:
        print(f"第一个对齐结果长度: {len(alignments_rust[0])}")
        print(f"第一个对齐结果示例（前20个）: {alignments_rust[0][:20]}")

    # 测试 PyTorch 版本进行对比
    print("\n=== PyTorch 版本 ===")
    log_probs_torch = torch.from_numpy(log_probs_batch)
    y_torch = torch.from_numpy(y_batch)
    T_torch = torch.from_numpy(T_batch)
    U_torch = torch.from_numpy(U_batch)

    alignments_torch = torch_viterbi_decoding(
        log_probs_torch, y_torch, T_torch, U_torch
    )

    print(f"对齐结果数量: {len(alignments_torch)}")
    if len(alignments_torch) > 0:
        print(f"第一个对齐结果长度: {len(alignments_torch[0])}")
        print(f"第一个对齐结果示例（前20个）: {alignments_torch[0][:20]}")
    
    # 比较结果
    print("\n=== 详细结果比较 ===")
    if len(alignments_rust) != len(alignments_torch):
        print(f"✗ 批次数量不匹配 (Rust: {len(alignments_rust)}, Torch: {len(alignments_torch)})")
        return

    print(f"✓ 批次数量匹配: {len(alignments_rust)}")

    # 统计信息
    all_match = True
    length_mismatches = 0
    value_mismatches = 0
    total_aligned_frames = 0

    for i in range(len(alignments_rust)):
        rust_len = len(alignments_rust[i])
        torch_len = len(alignments_torch[i])

        if rust_len != torch_len:
            length_mismatches += 1
            all_match = False
            print(f"✗ Batch {i}: 长度不匹配 (Rust: {rust_len}, Torch: {torch_len})")
            continue

        # 验证完整对齐序列
        rust_vals = alignments_rust[i]
        torch_vals = [int(x) for x in alignments_torch[i]]

        if rust_vals != torch_vals:
            value_mismatches += 1
            all_match = False
            # 找出第一个不匹配的位置
            first_mismatch = -1
            for j in range(rust_len):
                if rust_vals[j] != torch_vals[j]:
                    first_mismatch = j
                    break

            print(f"✗ Batch {i}: 值不匹配（长度: {rust_len}，首个差异位置: {first_mismatch}）")
            if first_mismatch >= 0:
                start = max(0, first_mismatch - 5)
                end = min(rust_len, first_mismatch + 6)
                print(f"  位置 {start}-{end-1}:")
                print(f"    Rust:  {rust_vals[start:end]}")
                print(f"    Torch: {torch_vals[start:end]}")
        else:
            total_aligned_frames += rust_len

    # 输出统计结果
    print(f"\n=== 统计摘要 ===")
    print(f"总批次数: {len(alignments_rust)}")
    print(f"长度不匹配: {length_mismatches}")
    print(f"值不匹配: {value_mismatches}")
    print(f"完全匹配批次: {len(alignments_rust) - length_mismatches - value_mismatches}")

    if all_match:
        print(f"\n✓✓✓ 所有 {len(alignments_rust)} 个批次的对齐结果完全匹配！")
        print(f"总计验证了 {total_aligned_frames} 个对齐帧")
    else:
        print(f"\n✗ 发现 {length_mismatches + value_mismatches} 个批次存在差异")

    return all_match



def test_viterbi_production_data():
    """测试 Rust 版本的 Viterbi 解码"""
    if not RUST_AVAILABLE:
        return

   
    
    # 生成测试数据
    with open("data/log_probs_batch.npy", "rb") as f:
        log_probs_batch = np.load(f)
    with open("data/y_batch.npy", "rb") as f:
        y_batch = np.load(f)
    with open("data/T_batch.npy", "rb") as f:
        T_batch = np.load(f)
    with open("data/U_batch.npy", "rb") as f:
        U_batch = np.load(f)
    
    print(f"测试参数: B={log_probs_batch.shape[0]}, V={log_probs_batch.shape[2]}, T_max={log_probs_batch.shape[1]}, U_max={y_batch.shape[1]}")
    print(f"张量大小: log_probs_batch={log_probs_batch.shape[0]}x{log_probs_batch.shape[1]}x{log_probs_batch.shape[2]}, y_batch={log_probs_batch.shape[0]}x{y_batch.shape[1]}")

    # 输出每个举证的数据类型
    print(f"log_probs_batch 数据类型: {log_probs_batch.dtype}")
    print(f"y_batch 数据类型: {y_batch.dtype}")
    print(f"T_batch 数据类型: {T_batch.dtype}")
    print(f"U_batch 数据类型: {U_batch.dtype}")
    
    # 测试 Rust 版本
    print("\n=== Rust 版本 ===")
    main_start = time.time()
    alignments_rust = speed_viterbi.viterbi_decoding(
        log_probs_batch, y_batch, T_batch, U_batch
    )
    main_elapsed = time.time() - main_start
    print(f"Rust 版本总耗时: {main_elapsed*1000:.4f} ms")
    print(f"对齐结果数量: {len(alignments_rust)}")
    if len(alignments_rust) > 0:
        print(f"第一个对齐结果长度: {len(alignments_rust[0])}")
        print(f"第一个对齐结果示例（前20个）: {alignments_rust[0][:20]}")

    # 测试 PyTorch 版本进行对比
    print("\n=== PyTorch 版本 ===")
    log_probs_torch = torch.from_numpy(log_probs_batch)
    y_torch = torch.from_numpy(y_batch)
    T_torch = torch.from_numpy(T_batch)
    U_torch = torch.from_numpy(U_batch)

    alignments_torch = torch_viterbi_decoding(
        log_probs_torch, y_torch, T_torch, U_torch
    )

    print(f"对齐结果数量: {len(alignments_torch)}")
    if len(alignments_torch) > 0:
        print(f"第一个对齐结果长度: {len(alignments_torch[0])}")
        print(f"第一个对齐结果示例（前20个）: {alignments_torch[0][:20]}")
    
    # 比较结果
    print("\n=== 详细结果比较 ===")
    if len(alignments_rust) != len(alignments_torch):
        print(f"✗ 批次数量不匹配 (Rust: {len(alignments_rust)}, Torch: {len(alignments_torch)})")
        return

    print(f"✓ 批次数量匹配: {len(alignments_rust)}")

    # 统计信息
    all_match = True
    length_mismatches = 0
    value_mismatches = 0
    total_aligned_frames = 0

    for i in range(len(alignments_rust)):
        rust_len = len(alignments_rust[i])
        torch_len = len(alignments_torch[i])

        if rust_len != torch_len:
            length_mismatches += 1
            all_match = False
            print(f"✗ Batch {i}: 长度不匹配 (Rust: {rust_len}, Torch: {torch_len})")
            continue

        # 验证完整对齐序列
        rust_vals = alignments_rust[i]
        torch_vals = [int(x) for x in alignments_torch[i]]

        if rust_vals != torch_vals:
            value_mismatches += 1
            all_match = False
            # 找出第一个不匹配的位置
            first_mismatch = -1
            for j in range(rust_len):
                if rust_vals[j] != torch_vals[j]:
                    first_mismatch = j
                    break

            print(f"✗ Batch {i}: 值不匹配（长度: {rust_len}，首个差异位置: {first_mismatch}）")
            if first_mismatch >= 0:
                start = max(0, first_mismatch - 5)
                end = min(rust_len, first_mismatch + 6)
                print(f"  位置 {start}-{end-1}:")
                print(f"    Rust:  {rust_vals[start:end]}")
                print(f"    Torch: {torch_vals[start:end]}")
        else:
            total_aligned_frames += rust_len

    # 输出统计结果
    print(f"\n=== 统计摘要 ===")
    print(f"总批次数: {len(alignments_rust)}")
    print(f"长度不匹配: {length_mismatches}")
    print(f"值不匹配: {value_mismatches}")
    print(f"完全匹配批次: {len(alignments_rust) - length_mismatches - value_mismatches}")

    if all_match:
        print(f"\n✓✓✓ 所有 {len(alignments_rust)} 个批次的对齐结果完全匹配！")
        print(f"总计验证了 {total_aligned_frames} 个对齐帧")
    else:
        print(f"\n✗ 发现 {length_mismatches + value_mismatches} 个批次存在差异")

    return all_match

if __name__ == "__main__":
    test_viterbi_production_data()
    test_viterbi_rust()
