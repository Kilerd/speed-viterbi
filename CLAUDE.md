# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance Viterbi decoding library for speech recognition alignment tasks, implemented in Rust with Python bindings via PyO3. The implementation provides both parallel and serial versions of the Viterbi algorithm for batch processing.

## Build and Development Commands

### Building the Project

```bash
# Development build (faster compilation, slower runtime)
maturin develop

# Release build (optimized for performance)
maturin develop --release
```

### Running Tests

```bash
# Run main test suite (compares Rust vs PyTorch implementations)
python test_viterbi.py

# Compare parallel vs serial performance
python test_serial_vs_parallel.py

# Test custom padding values
python test_custom_padding.py
```

## Architecture

### Core Implementation (src/lib.rs)

- **`viterbi_decode_single_batch`**: Core single-batch Viterbi algorithm
  - Input: log probabilities (T_max × V), token sequence (U_max), actual lengths
  - Uses dynamic programming with backpointer tracking
  - Handles letter repetition masking to prevent invalid transitions
  - Returns alignment as vector of frame-to-token indices

- **`viterbi_decoding_impl`**: Parallel batch processing using Rayon
  - Processes multiple batches concurrently with `par_iter()`
  - Best for batch sizes B ≥ 16

- **`viterbi_decoding_impl_serial`**: Serial batch processing
  - Sequential iteration over batches
  - Recommended for small batches (B < 16) or nested parallel contexts

### Python Bindings

Two exported functions via PyO3:
- **`viterbi_decoding`**: Parallel version (default)
- **`viterbi_decoding_serial`**: Single-threaded version

Both accept NumPy arrays as input with zero-copy views.

### Key Constants

- `PADDING_VALUE`: -3.4e38 (default padding for invalid positions)
- `BACKPOINTER_PADDING`: -99 (placeholder for backpointer array)

### Data Flow

1. Python NumPy arrays → PyO3 readonly views → ndarray ArrayView
2. Batch processing (parallel or serial) → per-batch Viterbi decode
3. Forward pass with backpointer recording → backtracking for alignment
4. Vec<Vec<usize>> → Python List[List[int]]

## Implementation Notes

### Memory Management
- Uses readonly array views to avoid copying large tensors
- Temporary arrays (v_prev, backpointers, etc.) allocated per batch
- Efficient memory reuse within time step loop

### Algorithm Details
- Supports variable-length sequences via `t_batch` and `u_batch` parameters
- Handles padding via masking at positions ≥ actual length
- Letter repetition detection prevents invalid blank-token-blank patterns
- Three-way comparison at each step: stay (0), advance 1 (+1), skip blank (+2)

### Testing Strategy
Test files expect a `torch_test.py` module (not in repo) that provides a PyTorch reference implementation for correctness validation. Tests compare:
- Result lengths across batches
- Exact alignment values at each time step
- Performance metrics between implementations

## Dependencies

- **pyo3**: Python bindings (v0.21)
- **numpy**: NumPy integration (v0.21)
- **ndarray**: N-dimensional arrays (v0.15)
- **rayon**: Data parallelism (v1.10)

## Python Module Name

The Python module is named `speed_viterbi`. Import it as:
```python
import speed_viterbi
```
