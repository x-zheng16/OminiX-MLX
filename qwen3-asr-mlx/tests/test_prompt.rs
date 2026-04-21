//! Tests for the chat-template prompt assembled by `build_prompt`.
//!
//! Guards the `set_context`-driven system-block injection against silent
//! regressions — both the empty-context format (must stay byte-identical to
//! the pre-set_context format) and the non-empty-context path (must land
//! the context inside the system block).

use qwen3_asr_mlx::testing::format_transcribe_prompt;

#[test]
fn empty_context_preserves_pre_set_context_format() {
    let expected = "<|im_start|>system\n\
                    <|im_end|>\n\
                    <|im_start|>user\n\
                    <|audio_start|><|audio_pad|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n\
                    <|im_start|>assistant\n\
                    language Chinese<asr_text>";
    assert_eq!(format_transcribe_prompt("", 3, "Chinese"), expected);
}

#[test]
fn non_empty_context_lands_inside_system_block() {
    let prompt = format_transcribe_prompt("VOCABULARY: lingyun", 2, "English");
    assert!(
        prompt.contains("<|im_start|>system\nVOCABULARY: lingyun<|im_end|>"),
        "context must appear between the system start tag and <|im_end|>; got: {prompt:?}"
    );
    assert!(prompt.contains("<|audio_pad|><|audio_pad|><|audio_end|>"));
    assert!(prompt.ends_with("language English<asr_text>"));
}

#[test]
fn audio_pad_count_matches_num_audio_tokens() {
    let prompt = format_transcribe_prompt("ctx", 5, "Chinese");
    assert_eq!(prompt.matches("<|audio_pad|>").count(), 5);
}

#[test]
fn zero_audio_tokens_leaves_empty_audio_region() {
    let prompt = format_transcribe_prompt("", 0, "Chinese");
    assert!(prompt.contains("<|audio_start|><|audio_end|>"));
}
