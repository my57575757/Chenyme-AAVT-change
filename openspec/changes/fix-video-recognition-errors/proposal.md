## Why

视频识别页面在点击"开始识别"后报错：`An error occurred during transcription: FasterWhisperResult() takes 9 positional arguments but 10 were given`。这是因为调用 `runWhisperSeperateProc` 时传递了 10 个参数，但底层函数 `FasterWhisperResult` 只接受 9 个参数。

## What Changes

- 修复 `runWhisperSeperateProc` 调用时参数过多问题（移除多余的 `st.session_state.video_name_original` 参数）
- 确保所有调用 `runWhisperSeperateProc` 的地方都使用正确的 9 个参数

## Capabilities

### New Capabilities

<!-- No new capabilities - this is a bug fix change -->

### Modified Capabilities

<!-- No existing specs modified - this is implementation-level bug fixes -->

## Impact

- **Affected Files**: `page/project/video.py` (第 544-550 行)
- **Dependencies**: Faster-Whisper
- **Breaking Changes**: 无
- **User Impact**: 修复后视频识别功能将正常工作，不再报参数错误
