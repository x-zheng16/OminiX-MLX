//! Regression guard for `QwenMLP::forward` + `mlx_rs_core::fused_swiglu`.
//!
//! Pins the contract `fused_swiglu(up, gate) == silu(gate) * up` so a miswired
//! swap in `qwen3_asr_mlx::qwen::QwenMLP::forward` (e.g., flipped argument
//! order) or a kernel regression breaks here instead of silently corrupting
//! decode output.

use std::sync::Once;

use mlx_rs::nn;
use mlx_rs::Array;

use mlx_rs_core::fused_swiglu;

// `mlx-sys` build.rs copies `mlx.metallib` to `target/<profile>/`, but integration-
// test binaries live at `target/<profile>/deps/`. MLX loads the metallib from the
// binary's directory, so without this the default metallib isn't discoverable
// from test runs. Mirror it next to the test binary once per process.
static METALLIB_INIT: Once = Once::new();

fn ensure_metallib_beside_test_binary() {
    METALLIB_INIT.call_once(|| {
        let Ok(exe) = std::env::current_exe() else { return };
        let Some(deps_dir) = exe.parent() else { return };
        let dest = deps_dir.join("mlx.metallib");
        if dest.exists() {
            return;
        }
        let Some(profile_dir) = deps_dir.parent() else { return };
        let src = profile_dir.join("mlx.metallib");
        if !src.exists() {
            return;
        }
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(&src, &dest);
        }
        #[cfg(not(unix))]
        {
            let _ = std::fs::copy(&src, &dest);
        }
    });
}

#[test]
fn fused_swiglu_matches_naive_silu_multiply() {
    ensure_metallib_beside_test_binary();

    // Distinctive magnitudes so swapped arguments would fail atol=1e-3.
    let up_data: [f32; 8] = [0.1, 2.0, -0.5, 3.0, 0.9, -1.1, 1.3, -1.5];
    let gate_data: [f32; 8] = [-2.0, 0.1, 3.0, -0.5, -1.0, 1.2, -1.4, 1.6];

    let up = Array::from_slice(&up_data, &[1, 2, 4]);
    let gate = Array::from_slice(&gate_data, &[1, 2, 4]);

    let naive = nn::silu(&gate).unwrap().multiply(&up).unwrap();
    let fused = fused_swiglu(&up, &gate).unwrap();

    let close = fused
        .all_close(&naive, 1e-3, 1e-3, None)
        .expect("all_close must succeed");
    let close_slice: &[bool] = close.as_slice();
    assert_eq!(
        close_slice,
        &[true],
        "fused_swiglu(up, gate) must equal silu(gate).multiply(up); \
         miswire in QwenMLP or kernel regression if this fires. \
         fused={:?} naive={:?}",
        fused.as_slice::<f32>(),
        naive.as_slice::<f32>(),
    );
}
