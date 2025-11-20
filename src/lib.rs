use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray3, PyArray2, PyArray1, PyArrayMethods};
use numpy::ndarray::{Array2, Array1, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;

const PADDING_VALUE: f32 = -3.4e38;
const BACKPOINTER_PADDING: i8 = -99;

/// 单个 batch 的 Viterbi 解码
fn viterbi_decode_single_batch(
    log_probs: ArrayView2<f32>,  // (T_max, V)
    y_seq: ArrayView1<i64>,      // (U_max,)
    t_len: usize,
    u_len: usize,
    padding_value: f32,
) -> Vec<usize> {
    let (t_max, v) = log_probs.dim();
    let u_max = y_seq.len();

    // 初始化 v_prev: (U_max,)
    let mut v_prev = Array1::<f32>::from_elem(u_max, padding_value);
    for u in 0..2.min(u_max) {
        let y_val = y_seq[u] as usize;
        if y_val < v {
            v_prev[u] = log_probs[[0, y_val]];
        } else if y_val == v {
            v_prev[u] = padding_value;
        }
    }

    // 初始化 backpointers
    let mut backpointers_rel = Array2::<i8>::from_elem((t_max, u_max), BACKPOINTER_PADDING);

    // letter_repetition_mask
    let mut letter_repetition_mask = Array1::<bool>::from_elem(u_max, false);
    for u in 2..u_max {
        if y_seq[u] == y_seq[u - 2] {
            letter_repetition_mask[u] = true;
        }
    }

    // 临时数组
    let mut e_current = Array1::<f32>::from_elem(u_max, padding_value);
    let mut v_prev_shifted = Array1::<f32>::from_elem(u_max, padding_value);
    let mut v_prev_shifted2 = Array1::<f32>::from_elem(u_max, padding_value);

    // 前向传播
    for t in 1..t_max {
        // 获取当前时间步的 log probs
        e_current.fill(padding_value);
        for u in 0..u_max {
            let y_val = y_seq[u] as usize;
            if y_val < v {
                e_current[u] = log_probs[[t, y_val]];
            }
        }

        // 应用 mask
        if t >= t_len {
            if u_len < u_max {
                e_current[u_len] = 0.0;
            }
            if u_len > 0 && u_len - 1 < u_max {
                e_current[u_len - 1] = 0.0;
            }
        }

        // shifted 版本
        v_prev_shifted[0] = padding_value;
        for u in 1..u_max {
            v_prev_shifted[u] = v_prev[u - 1];
        }

        v_prev_shifted2[0] = padding_value;
        if u_max > 1 {
            v_prev_shifted2[1] = padding_value;
        }
        for u in 2..u_max {
            if letter_repetition_mask[u] {
                v_prev_shifted2[u] = padding_value;
            } else {
                v_prev_shifted2[u] = v_prev[u - 2];
            }
        }

        // 计算最大值
        for u in 0..u_max {
            let e = e_current[u];
            let c0 = v_prev[u] + e;
            let c1 = v_prev_shifted[u] + e;
            let c2 = v_prev_shifted2[u] + e;

            let (max_val, max_idx) = if c1 > c0 {
                if c2 > c1 { (c2, 2) } else { (c1, 1) }
            } else {
                if c2 > c0 { (c2, 2) } else { (c0, 0) }
            };

            v_prev[u] = max_val;
            backpointers_rel[[t, u]] = max_idx;
        }
    }

    // 回溯
    let mut current_u = if u_len == 1 {
        0
    } else {
        let val_at_ub_minus_2 = v_prev[u_len - 2];
        let val_at_ub_minus_1 = v_prev[u_len - 1];
        if val_at_ub_minus_1 > val_at_ub_minus_2 {
            u_len - 1
        } else {
            u_len - 2
        }
    };

    let mut alignment = Vec::with_capacity(t_max);
    alignment.push(current_u);
    for t in (1..t_max).rev() {
        let bp = backpointers_rel[[t, current_u]] as i32;
        current_u = (current_u as i32 - bp) as usize;
        alignment.push(current_u);
    }

    alignment.reverse();
    alignment.truncate(t_len);
    alignment
}

/// Viterbi 解码的核心实现（并行版本）
fn viterbi_decoding_impl(
    log_probs_batch: &ArrayView3<f32>,  // (B, T_max, V)
    y_batch: &ArrayView2<i64>,          // (B, U_max)
    t_batch: &ArrayView1<i64>,          // (B,)
    u_batch: &ArrayView1<i64>,          // (B,)
    padding_value: f32,
) -> Vec<Vec<usize>> {
    let b = log_probs_batch.shape()[0];

    // 并行处理每个 batch
    (0..b)
        .into_par_iter()
        .map(|i| {
            let log_probs = log_probs_batch.index_axis(Axis(0), i);
            let y_seq = y_batch.index_axis(Axis(0), i);
            let t_len = t_batch[i] as usize;
            let u_len = u_batch[i] as usize;

            viterbi_decode_single_batch(log_probs, y_seq, t_len, u_len, padding_value)
        })
        .collect()
}

/// Viterbi 解码的串行实现（单线程版本）
fn viterbi_decoding_impl_serial(
    log_probs_batch: &ArrayView3<f32>,  // (B, T_max, V)
    y_batch: &ArrayView2<i64>,          // (B, U_max)
    t_batch: &ArrayView1<i64>,          // (B,)
    u_batch: &ArrayView1<i64>,          // (B,)
    padding_value: f32,
) -> Vec<Vec<usize>> {
    let (b, _t_max, _v) = log_probs_batch.dim();

    // 串行处理每个 batch
    (0..b)
        .map(|i| {
            let log_probs = log_probs_batch.index_axis(Axis(0), i);
            let y_seq = y_batch.index_axis(Axis(0), i);
            let t_len = t_batch[i] as usize;
            let u_len = u_batch[i] as usize;

            viterbi_decode_single_batch(log_probs, y_seq, t_len, u_len, padding_value)
        })
        .collect()
}


/// Python 绑定的 Viterbi 解码函数（并行版本，默认）
#[pyfunction]
#[pyo3(signature = (log_probs_batch, y_batch, t_batch, u_batch, padding_value=None))]
fn viterbi_decoding(
    log_probs_batch: &Bound<'_, PyArray3<f32>>,
    y_batch: &Bound<'_, PyArray2<i64>>,
    t_batch: &Bound<'_, PyArray1<i64>>,
    u_batch: &Bound<'_, PyArray1<i64>>,
    padding_value: Option<f32>,
) -> PyResult<PyObject> {
    let padding = padding_value.unwrap_or(PADDING_VALUE);

    // 转换为 ndarray（使用视图避免拷贝，无需 unsafe）
    // 需要保持 readonly 的绑定以延长生命周期
    let log_probs_ro = log_probs_batch.readonly();
    let y_ro = y_batch.readonly();
    let t_ro = t_batch.readonly();
    let u_ro = u_batch.readonly();

    let log_probs = log_probs_ro.as_array();
    let y = y_ro.as_array();
    let t = t_ro.as_array();
    let u = u_ro.as_array();

    // 执行 Viterbi 解码（并行版本）
    let alignments = viterbi_decoding_impl(&log_probs, &y, &t, &u, padding);

    // 转换为 Python 列表
    Python::with_gil(|py| {
        let result = PyList::empty_bound(py);
        for alignment in alignments {
            let py_alignment = PyList::new_bound(py, alignment);
            result.append(py_alignment)?;
        }
        Ok(result.to_object(py))
    })
}

/// Python 绑定的 Viterbi 解码函数（串行/单线程版本）
#[pyfunction]
#[pyo3(signature = (log_probs_batch, y_batch, t_batch, u_batch, padding_value=None))]
fn viterbi_decoding_serial(
    log_probs_batch: &Bound<'_, PyArray3<f32>>,
    y_batch: &Bound<'_, PyArray2<i64>>,
    t_batch: &Bound<'_, PyArray1<i64>>,
    u_batch: &Bound<'_, PyArray1<i64>>,
    padding_value: Option<f32>,
) -> PyResult<PyObject> {
    let padding = padding_value.unwrap_or(PADDING_VALUE);

    // 转换为 ndarray（使用视图避免拷贝，无需 unsafe）
    // 需要保持 readonly 的绑定以延长生命周期
    let log_probs_ro = log_probs_batch.readonly();
    let y_ro = y_batch.readonly();
    let t_ro = t_batch.readonly();
    let u_ro = u_batch.readonly();

    let log_probs = log_probs_ro.as_array();
    let y = y_ro.as_array();
    let t = t_ro.as_array();
    let u = u_ro.as_array();

    // 执行 Viterbi 解码（串行版本）
    let alignments = viterbi_decoding_impl_serial(&log_probs, &y, &t, &u, padding);

    // 转换为 Python 列表
    Python::with_gil(|py| {
        let result = PyList::empty_bound(py);
        for alignment in alignments {
            let py_alignment = PyList::new_bound(py, alignment);
            result.append(py_alignment)?;
        }
        Ok(result.to_object(py))
    })
}

/// Python 模块定义
#[pymodule]
fn speed_viterbi(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(viterbi_decoding, m)?)?;
    m.add_function(wrap_pyfunction!(viterbi_decoding_serial, m)?)?;
    Ok(())
}
