//! Tests that `load()` merges `quantization_config.json` into `config.json`
//! when the main config lacks a `quantization` block.

use std::fs;
use tempfile::tempdir;

use qwen3_asr_mlx::testing::merge_quantization_config;

#[test]
fn merges_sibling_quantization_file_when_main_config_has_none() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["Qwen3ASRForConditionalGeneration"]}"#,
    )
    .unwrap();
    fs::write(
        dir.path().join("quantization_config.json"),
        r#"{"bits":4,"group_size":64}"#,
    )
    .unwrap();

    let mut cfg: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("config.json")).unwrap())
            .unwrap();

    merge_quantization_config(dir.path(), &mut cfg);

    assert_eq!(cfg["quantization"]["bits"], 4);
    assert_eq!(cfg["quantization"]["group_size"], 64);
}

#[test]
fn leaves_config_untouched_when_quantization_already_present() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"quantization":{"bits":8,"group_size":32}}"#,
    )
    .unwrap();
    fs::write(
        dir.path().join("quantization_config.json"),
        r#"{"bits":4,"group_size":64}"#,
    )
    .unwrap();

    let mut cfg: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("config.json")).unwrap())
            .unwrap();

    merge_quantization_config(dir.path(), &mut cfg);

    assert_eq!(cfg["quantization"]["bits"], 8);
    assert_eq!(cfg["quantization"]["group_size"], 32);
}

#[test]
fn noop_when_sibling_file_absent() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["Qwen3ASRForConditionalGeneration"]}"#,
    )
    .unwrap();

    let mut cfg: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("config.json")).unwrap())
            .unwrap();

    merge_quantization_config(dir.path(), &mut cfg);

    assert!(cfg.get("quantization").is_none());
}
