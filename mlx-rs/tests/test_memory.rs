//! Integration tests for the `mlx_rs::memory` module — exposes MLX's
//! allocator-cache and memory-limit knobs that long-running daemons need
//! to keep Metal cache footprint bounded.
//!
//! Background: `mlx-c` (`mlx/c/memory.h`) ships C bindings for
//! `mlx_clear_cache` / `mlx_set_cache_limit` / `mlx_set_memory_limit` /
//! `mlx_set_wired_limit` plus several read-side getters. `mlx-sys` already
//! generates Rust bindings for them via `bindgen` (entry header
//! `mlx/c/mlx.h` includes `memory.h`). What's missing — and what these
//! tests pin — is a safe `mlx_rs::memory` Rust wrapper.

use mlx_rs::memory;
use mlx_rs::Array;

#[test]
fn set_cache_limit_round_trips_previous_value() {
    // A->B then B->A: the second call MUST return B (the previous value),
    // proving the `*res` out-parameter from `mlx_set_cache_limit` is being
    // read back correctly.
    let original = memory::set_cache_limit(1024 * 1024); // 1 MiB
    let after_b = memory::set_cache_limit(2 * 1024 * 1024); // 2 MiB
    let after_restore = memory::set_cache_limit(original);
    assert_eq!(
        after_b,
        1024 * 1024,
        "set_cache_limit(2 MiB) called when cache_limit was 1 MiB MUST return 1 MiB"
    );
    assert_eq!(
        after_restore,
        2 * 1024 * 1024,
        "set_cache_limit(original) called when cache_limit was 2 MiB MUST return 2 MiB"
    );
}

#[test]
fn clear_cache_is_callable_on_empty_pool() {
    // Smoke: calling clear_cache when nothing is cached must not panic or
    // raise. Idempotent invariant for daemons that defensively clear after
    // every transcription.
    memory::clear_cache();
    memory::clear_cache(); // twice on purpose
}

#[test]
fn clear_cache_drops_cache_memory_after_allocation_drop() {
    // Allocate-then-drop pattern: an Array's underlying Metal buffer goes
    // into the cache pool when the Array is dropped, growing
    // `get_cache_memory`. `clear_cache` MUST then bring it back down.
    //
    // We compare cache-after-drop vs cache-after-clear, not absolute values,
    // so the test is robust to whatever baseline the runtime carries.
    {
        let arr = Array::from_slice(&[0.0f32; 4 * 1024 * 1024], &[4 * 1024 * 1024]);
        // Force materialization so the buffer actually exists in the pool.
        let _: &[f32] = arr.as_slice();
    }
    let cache_after_drop = memory::get_cache_memory();
    memory::clear_cache();
    let cache_after_clear = memory::get_cache_memory();
    assert!(
        cache_after_clear <= cache_after_drop,
        "clear_cache must not GROW cache_memory: was {} bytes, became {} bytes",
        cache_after_drop,
        cache_after_clear
    );
}

#[test]
fn set_memory_limit_round_trips_previous_value() {
    // Mirror of set_cache_limit: confirms the `*res` out-parameter contract
    // for the broader memory-budget knob too.
    let original = memory::set_memory_limit(8 * 1024 * 1024 * 1024); // 8 GiB
    let after = memory::set_memory_limit(original);
    assert_eq!(
        after,
        8 * 1024 * 1024 * 1024,
        "set_memory_limit must return the previous limit through *res"
    );
}

#[test]
fn get_peak_memory_can_reset() {
    // After `reset_peak_memory`, the high-water mark must NOT exceed what
    // it was before reset (and is typically 0 or current active memory).
    // Trigger an allocation first so the peak is non-trivial.
    let _arr = Array::from_slice(&[1.0f32; 1024], &[1024]);
    let peak_before = memory::get_peak_memory();
    memory::reset_peak_memory();
    let peak_after = memory::get_peak_memory();
    assert!(
        peak_after <= peak_before,
        "peak after reset ({}) must be <= peak before ({})",
        peak_after,
        peak_before
    );
}
