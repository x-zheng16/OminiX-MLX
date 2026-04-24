//! Regression guard for perf-phase Item 1 in OminiX-MLX (moona3k
//! `3db67fe perf: cache mel filter asset and Hann window in audio frontend`
//! equivalent, reconstructed from context — see CAVEAT below).
//!
//! In moona3k Python, the upstream commit added `lru_cache` to avoid rebuilding
//! the mel filterbank and Hann window on every `extract_fbank_features` call.
//! In the Rust fork the same invariant is held structurally: `MelFrontend`
//! owns the `mel_filters` / `window` / `fft` fields, constructed once in
//! `::new` and reused via `&self` on every `compute_mel_spectrogram` call.
//!
//! This test pins that invariant behaviorally. A regression that moves
//! filterbank construction inside `compute_mel_spectrogram` (rebuilding a
//! local `mel_filters` per call) would make `compute_mel_spectrogram`
//! insensitive to mutations of `self.mel_filters`, and the behavioral test
//! fires. The pointer-stability test additionally pins that the owned field
//! is not replaced during a call (e.g., `self.mel_filters = ...` inside the
//! hot path).
//!
//! CAVEAT ON PROVENANCE: the moona3k Python perf-phase items (1/2/4/5) were
//! enumerated privately by Xiang; the mapping to moona3k commits
//! (3db67fe / 3fb16d4 / 7bc22fb / 68fb1cc) was reconstructed from context
//! during the recovery dispatch on 2026-04-25 (meta-orch task
//! `recover-and-proceed-a-plus-b`). Item 1 in that mapping is 3db67fe. The
//! reconstruction is not confirmed against an authoritative backlog; if the
//! mapping turns out to be wrong, this test is still a valid structural
//! regression guard for `MelFrontend` — it does not depend on the mapping
//! being correct.

use qwen3_asr_mlx::audio::{testing, AudioConfig, MelFrontend};

use std::sync::Once;

// `mlx-sys` build.rs copies `mlx.metallib` to `target/<profile>/`, but
// integration-test binaries live at `target/<profile>/deps/`. MLX loads the
// metallib from the binary's directory, so without this the default metallib
// isn't discoverable from test runs. Mirror it next to the test binary once
// per process. Matches the boilerplate in `tests/test_fused_swiglu.rs`.
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

/// Deterministic synthetic PCM at 16 kHz — sine + cosine mix, enough samples
/// to give `compute_mel_spectrogram` multiple frames (`n_fft=400`, `hop=160`).
fn synthetic_samples(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 / 16000.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 880.0 / 16000.0 * t).cos()
        })
        .collect()
}

#[test]
fn mel_frontend_uses_owned_filter_bank_on_every_call() {
    // Behavioral invariant: `compute_mel_spectrogram` reads `&self.mel_filters`
    // — NOT a local `mel_filters = create_whisper_mel_filterbank(...)` rebuilt
    // per call. Mutating `self.mel_filters` between calls MUST change the
    // output; if it doesn't, the hot path has regressed to per-call rebuild.
    ensure_metallib_beside_test_binary();

    let mut frontend = MelFrontend::new(AudioConfig::default());
    let samples = synthetic_samples(1600);

    let out1 = frontend
        .compute_mel_spectrogram(&samples)
        .expect("baseline mel spectrogram");
    let out1_data: Vec<f32> = out1.as_slice::<f32>().to_vec();

    // Multiplying every filter by 2.0 doubles each mel energy before the
    // `log10 + Whisper normalize` stages — output MUST shift.
    testing::scale_mel_filters(&mut frontend, 2.0);

    let out2 = frontend
        .compute_mel_spectrogram(&samples)
        .expect("scaled mel spectrogram");
    let out2_data: Vec<f32> = out2.as_slice::<f32>().to_vec();

    assert_eq!(
        out1_data.len(),
        out2_data.len(),
        "output shape changed unexpectedly"
    );

    assert_ne!(
        out1_data, out2_data,
        "MelFrontend regression: scaling `self.mel_filters` did not change \
         `compute_mel_spectrogram` output. This means the hot path rebuilt \
         `mel_filters` locally instead of reading the owned struct field — \
         the construction-time cache invariant is broken."
    );
}

#[test]
fn mel_frontend_owned_buffers_are_pointer_stable_across_calls() {
    // Structural invariant: `self.mel_filters` / `self.window` are owned
    // `Vec<f32>` fields on `MelFrontend`, never reallocated by
    // `compute_mel_spectrogram`. A regression that assigns
    // `self.mel_filters = ...` inside the hot path would reallocate the
    // backing buffer and shift the pointer.
    ensure_metallib_beside_test_binary();

    let frontend = MelFrontend::new(AudioConfig::default());
    let mel_ptr0 = testing::mel_filters_slice(&frontend).as_ptr();
    let win_ptr0 = testing::window_slice(&frontend).as_ptr();

    let samples = synthetic_samples(1600);
    let _ = frontend
        .compute_mel_spectrogram(&samples)
        .expect("first spectrogram");
    let mel_ptr1 = testing::mel_filters_slice(&frontend).as_ptr();
    let win_ptr1 = testing::window_slice(&frontend).as_ptr();

    let _ = frontend
        .compute_mel_spectrogram(&samples)
        .expect("second spectrogram");
    let mel_ptr2 = testing::mel_filters_slice(&frontend).as_ptr();
    let win_ptr2 = testing::window_slice(&frontend).as_ptr();

    assert_eq!(mel_ptr0, mel_ptr1, "mel_filters relocated after first call");
    assert_eq!(mel_ptr1, mel_ptr2, "mel_filters relocated after second call");
    assert_eq!(win_ptr0, win_ptr1, "window relocated after first call");
    assert_eq!(win_ptr1, win_ptr2, "window relocated after second call");
}
