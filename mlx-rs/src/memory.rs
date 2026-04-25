//! MLX memory and allocator-cache management.
//!
//! `mlx-c` exposes a memory-control surface (`mlx/c/memory.h`) — allocator
//! cache cap, per-process memory budget, wired-memory limit, plus several
//! getters — that has no counterpart in the higher-level `mlx-rs` crate
//! until this module. Long-running daemons (e.g., the `xiaoyu` STT
//! daemon) need these knobs to keep Metal cache footprint bounded:
//! without them, MLX's allocator pool retains every working-set sized
//! allocation it has ever seen, and a single long transcription can
//! permanently push the process footprint multiple gigabytes higher.
//!
//! ## Typical daemon usage
//!
//! ```rust,ignore
//! use mlx_rs::memory;
//!
//! // Cap the Metal allocator cache at startup; trades a small amount of
//! // recycle speed for predictable footprint.
//! let _previous = memory::set_cache_limit(2 * 1024 * 1024 * 1024); // 2 GiB
//!
//! // ... run inference ...
//!
//! // After a long operation, return any retained buffers to the OS.
//! memory::clear_cache();
//! ```
//!
//! ## Naming
//!
//! Names mirror the C bindings without the `mlx_` prefix. The compile-time
//! `clear_cache` exposed by [`crate::transforms::compile`] is unrelated —
//! that one drops compiled-graph IR; this one drops Metal allocator
//! buffers.

/// Free any cached Metal buffers held by MLX's allocator pool.
///
/// Live `Array` data is unaffected — only the recycle pool is drained.
/// Idempotent and infallible in practice; safe to call after every
/// inference pass to bound footprint.
pub fn clear_cache() {
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
}

/// Set the Metal allocator cache size limit in bytes; returns the previous
/// limit.
///
/// Once the cache crosses `limit`, MLX returns buffers to the OS instead of
/// pooling them. Setting `0` disables caching entirely (every allocation
/// hits the OS) — strongest footprint guarantee, slowest recycle path.
///
/// Reasonable starting points on Apple Silicon: 2 GiB for a single-model
/// daemon, 4-8 GiB for multi-model processes.
pub fn set_cache_limit(limit: usize) -> usize {
    let mut previous: usize = 0;
    unsafe {
        mlx_sys::mlx_set_cache_limit(&mut previous as *mut _, limit);
    }
    previous
}

/// Set the per-process Metal allocation budget in bytes; returns the
/// previous limit.
///
/// Unlike [`set_cache_limit`], which only governs the recycle pool, this
/// is the hard ceiling on total Metal allocations. Allocations beyond it
/// fail.
pub fn set_memory_limit(limit: usize) -> usize {
    let mut previous: usize = 0;
    unsafe {
        mlx_sys::mlx_set_memory_limit(&mut previous as *mut _, limit);
    }
    previous
}

/// Set the wired-memory limit in bytes; returns the previous limit.
///
/// On Apple Silicon, "wired" memory is locked into physical RAM and not
/// eligible for compression / swap. Tuning this is rarely useful from
/// userland; default behaviour is correct for inference workloads.
pub fn set_wired_limit(limit: usize) -> usize {
    let mut previous: usize = 0;
    unsafe {
        mlx_sys::mlx_set_wired_limit(&mut previous as *mut _, limit);
    }
    previous
}

/// Bytes currently held by live `Array`s.
///
/// Excludes the recycle pool; see [`get_cache_memory`] for that.
pub fn get_active_memory() -> usize {
    let mut res: usize = 0;
    unsafe {
        mlx_sys::mlx_get_active_memory(&mut res as *mut _);
    }
    res
}

/// Bytes held by MLX's allocator cache pool (recyclable, not live).
pub fn get_cache_memory() -> usize {
    let mut res: usize = 0;
    unsafe {
        mlx_sys::mlx_get_cache_memory(&mut res as *mut _);
    }
    res
}

/// Current per-process Metal memory budget; matches the value returned
/// previously by [`set_memory_limit`].
pub fn get_memory_limit() -> usize {
    let mut res: usize = 0;
    unsafe {
        mlx_sys::mlx_get_memory_limit(&mut res as *mut _);
    }
    res
}

/// Peak bytes ever held by live `Array`s during this process, since the
/// last [`reset_peak_memory`] call (or process start).
pub fn get_peak_memory() -> usize {
    let mut res: usize = 0;
    unsafe {
        mlx_sys::mlx_get_peak_memory(&mut res as *mut _);
    }
    res
}

/// Reset the high-water mark returned by [`get_peak_memory`].
pub fn reset_peak_memory() {
    unsafe {
        mlx_sys::mlx_reset_peak_memory();
    }
}
