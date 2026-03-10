## Context

视频识别页面 (`page/project/video.py`) 在调用 `runWhisperSeperateProc` 函数时传递了 10 个参数，但该函数内部调用的 `FasterWhisperResult` 只接受 9 个参数，导致报错：

```
FasterWhisperResult() takes 9 positional arguments but 10 were given
```

**问题位置**：
- `video.py` 第 544-546 行（FasterWhisper - AutoDownload 模式）
- `video.py` 第 548-550 行（FasterWhisper - LocalModel 模式）

**多余参数**：`st.session_state.video_name_original` 在第 10 个参数位置，但 `FasterWhisperResult` 不需要此参数。

## Goals / Non-Goals

**Goals:**
- 移除 `runWhisperSeperateProc` 调用中多余的参数
- 确保所有调用点都使用正确的 9 个参数
- 保持现有代码结构和功能不变

**Non-Goals:**
- 不修改 `FasterWhisperResult` 函数签名
- 不添加新功能
- 不重构代码结构

## Decisions

### 决策 1：移除多余的参数
**现状**：调用时传递 10 个参数（第 10 个是 `st.session_state.video_name_original`）  
**修复**：移除第 10 个参数，保留前 9 个参数  
**理由**：`FasterWhisperResult` 函数签名只需要 9 个参数，第 10 个参数未被使用

**参数列表（9 个）**：
1. `file_path` - 文件路径
2. `device` - GPU/CPU 设置
3. `model_name` - 模型名称
4. `prompt` - 提示词
5. `temp` - 温度
6. `vad` - VAD 设置
7. `lang` - 语言
8. `beam_size` - Beam Size
9. `min_vad` - 最小 VAD 持续时间

## Risks / Trade-offs

**[风险] 移除参数可能影响其他功能**  
→ 缓解：该参数在当前代码中未被使用，移除无影响

**[风险] 修改核心逻辑可能引入新 bug**  
→ 缓解：修改后手动测试视频识别功能

## Migration Plan

1. 备份当前 `video.py` 文件
2. 移除第 544-546 行的第 10 个参数
3. 移除第 548-550 行的第 10 个参数
4. 手动测试视频识别功能

## Open Questions

无 - 这是明确的 bug 修复
