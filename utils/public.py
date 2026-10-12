import time
import json
import sysconfig
import subprocess
import requests
from pathlib import Path
from openai import OpenAI
from openai import OpenAIError
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple
from faster_whisper import WhisperModel
import anthropic
import pandas as pd
import google.generativeai as genai
import re
import cv2
import os
import shutil
import toml
import torch
import base64
import traceback
import streamlit as st
from typing import List
from multiprocessing import Process, Manager
import pysrt
import datetime
from moviepy.editor import *


def _ensure_ffmpeg_on_path():
    # winget 新装的 ffmpeg 在 WinGet\Links,启动器进程未重启时继承的旧 PATH 找不到它
    if shutil.which("ffmpeg"):
        return
    winget_links = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links")
    if os.path.isfile(os.path.join(winget_links, "ffmpeg.exe")):
        os.environ["PATH"] = winget_links + os.pathsep + os.environ.get("PATH", "")


_ensure_ffmpeg_on_path()


def _register_nvidia_dlls():
    # ctranslate2 在模型加载时才按需 LoadLibrary cuBLAS/cuDNN;pip 安装的运行库位于
    # site-packages/nvidia/*/bin,不在系统 PATH,需显式注册 DLL 目录
    if sys.platform != "win32":
        return
    nvidia_root = os.path.join(sysconfig.get_paths().get("purelib", ""), "nvidia")
    if not os.path.isdir(nvidia_root):
        return
    for bin_dir in Path(nvidia_root).glob("*/bin"):
        if any(bin_dir.glob("*.dll")):
            os.add_dll_directory(str(bin_dir))
            # ctranslate2 内部以传统搜索方式 LoadLibrary cuBLAS,add_dll_directory 对它无效,必须进 PATH
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_register_nvidia_dlls()


def get_global_info():
    path = os.getcwd().replace("utils", "") + "/"
    llms_path = path + "config/llms.toml"
    with open(llms_path, 'r', encoding="utf-8") as config_file:
        llms = toml.load(config_file)
    global_key = llms["Global"]["key"]
    global_url = llms["Global"]["url"]
    all_in_one = llms["Global"]["all"]
    return global_key, global_url, all_in_one


EXCLUDED_VIDEO_DIRS = {"云盘缓存文件"}


def findVideos(directory: str) -> List:
    videos = []
    fileExt = ['m4a', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm', 'mkv', 'avi']
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_VIDEO_DIRS]
        for file in files:
            if file.split('.')[-1] in fileExt:
                videos.append(os.path.join(root, file))
    print(f"找到{len(videos)}个视频文件")
    return videos

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def FileToMp3(log_level: str, input_dir: str, output_dir: str, output_name: str = "output.mp3"):
    input_path = Path(input_dir)
    output_path = Path(output_dir) / output_name

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if input_path.suffix.lower() == ".mp3":
        print("\033[1;34m🎧 文件已经是 MP3 格式，无需转换。\033[0m")
        return output_path
    if os.path.exists(output_path):
        print("\033[1;34m🎧 文件已存在，无需转换。\033[0m")
        return output_path
    try:
        command = [
            "ffmpeg", "-loglevel", log_level, "-i", str(input_path),
            "-vn", "-acodec", "libmp3lame", "-ab", "320k",
            "-f", "mp3", str(output_path)
        ]
        subprocess.run(command, check=True)
        print("\033[1;34m🎧 文件已成功转换为 MP3 格式！\033[0m")
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 执行失败: {e}")
        # raise EOFError(f"FFmpeg 执行失败: {e}")
    except Exception as e:
        raise EOFError(f"发生错误！可能是 FFmpeg 未被正确配置或上传文件格式不受支持。详细信息: {e}")


def OpenaiWhisperResult(key: str, url: str, path: str, model: str, prompt: str, temperature: float):
    print("\n\033[1;35m*** OpenAI Whisper API 调用模式 ***\033[0m\n")
    print(f"\033[1;34m🈴 识别模型: {model}\033[0m")
    global_key, global_url, all_in_one = get_global_info()

    if all_in_one:
        client = OpenAI(api_key=global_key, base_url=global_url)
        print(f"\033[1;34m🌐 代理地址: {global_url}\033[0m")
    else:
        client = OpenAI(api_key=key, base_url=url)
        if url != "https://api.openai.com/v1":
            print(f"\033[1;34m🌐 代理地址: {url}\033[0m")

    audio_path = Path(path)
    try:
        if audio_path.is_file():
            audio_file_path = audio_path
        else:
            audio_file_path = audio_path / "output.mp3"
        if not audio_file_path.is_file():
            raise FileNotFoundError(f"文件未找到：{audio_file_path}")
    except FileNotFoundError as e:
        return {"error": f"文件错误：{str(e)}"}

    try:
        with audio_file_path.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                prompt=prompt,
                temperature=temperature
            )
    except OpenAIError as e:
        return {"error": f"OpenAI Whisper API 请求失败：{str(e)}"}

    finally:
        audio_file.close()

    result = {'text': transcript.text, 'segments': transcript.segments}
    print(f"\033[1;34m📝 Whisper识别结果:\033[0m\n{result['text']}\n")
    return result


def FasterWhisperResultDict(segments, info):
    segments = list(segments)
    return {
        'lang': info.language,
        'text': ' '.join([segment.text for segment in segments]),
        'segments': [{
            'id': segment.id,
            'seek': segment.seek,
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'tokens': segment.tokens,
            'temperature': segment.temperature,
            'avg_logprob': segment.avg_logprob,
            'compression_ratio': segment.compression_ratio,
            'no_speech_prob': segment.no_speech_prob}
            for segment in segments
        ]
    }


FASTER_VALID_MODELS = ('tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1',
                       'large-v2', 'large-v3', 'large-v3-turbo', 'large', 'distil-small.en', 'distil-medium.en',
                       'distil-large-v2', 'distil-large-v3')

FASTER_ADVANCED_DEFAULTS = {
    "condition_prev_text": False,       # 关闭上文承接,防重复句幻觉
    "hallucination_silence": 2.0,       # 秒,0 关闭;片段开头静音即丢弃
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "vad_threshold": 0.5,
    "drop_no_speech_prob": 0.85,        # 事后过滤:1.0 关闭
    "dedupe_repeat": True,              # 事后过滤:去掉相邻重复行
    "hotwords": "",
    "hallucination_phrases": [
        # 日语 YouTube 片尾/频道套话类幻觉(社区实测高频,正片不可能出现)
        "ありがとうございました",
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "最後までご視聴ありがとうございました",
        "ご清聴ありがとうございました",
        "チャンネル登録お願いします",
        "高評価とチャンネル登録をお願いします",
        "また次回お会いしましょう",
        # 中文同类
        "感谢观看",
        "谢谢观看",
        "请订阅频道",
        "感谢您的观看",
        # 英语片尾套话 / 字幕站署名幻觉
        "Thank you for watching.",
        "Thanks for watching.",
        "Please subscribe.",
        "Please like and subscribe.",
        "Subtitles by Amara.org",
        "Transcription by CastingWords",
        "DimaTorzok",
    ],
}


def _load_faster_advanced_settings(model_name: str) -> dict:
    # 高级参数从 config/whisper.toml 读取,避免改 FasterWhisperResult 的位置参数(调用点多且 audio 页历史上有传参顺序坑)
    settings = dict(FASTER_ADVANCED_DEFAULTS)
    section = "Faster" if model_name in FASTER_VALID_MODELS else "Faster_Local"
    config_path = Path(__file__).resolve().parent.parent / "config" / "whisper.toml"
    try:
        with open(config_path, 'r', encoding="utf-8") as f:
            raw = toml.load(f).get(section, {})
        for key in settings:
            if key in raw:
                settings[key] = raw[key]
    except Exception as e:
        print(f"\033[1;33m⚠️ 高级识别参数读取失败,使用默认值: {type(e).__name__}: {e}\033[0m")
    return settings


def _is_phantom_text(text: str, phrases) -> bool:
    normalized = text.strip().strip("。.!?！？…・、, \t\r\n")
    if not normalized:
        return True
    return any(p.strip() and normalized == p.strip() for p in phrases)


def _filter_hallucinated_segments(segments: list, settings: dict) -> list:
    kept = []
    dropped = 0
    prev_norm = None
    for seg in segments:
        text = seg.text.strip()
        if settings["drop_no_speech_prob"] < 1.0 and seg.no_speech_prob >= settings["drop_no_speech_prob"]:
            dropped += 1
            continue
        if _is_phantom_text(text, settings["hallucination_phrases"]):
            dropped += 1
            continue
        norm = re.sub(r"\s+", "", text)
        if settings["dedupe_repeat"] and norm and norm == prev_norm:
            dropped += 1
            continue
        prev_norm = norm
        kept.append(seg)
    if dropped:
        print(f"\033[1;33m⚠️ 幻觉过滤: 丢弃 {dropped} 个可疑片段\033[0m")
    return kept


def FasterWhisperResult(file_path: str, device: bool, model_name: str, prompt: str, temp: float, vad: bool, lang: str,
                        beam_size: int, min_vad: int, fileName: str, returnList):
    valid_models = FASTER_VALID_MODELS
    if device:
        device = "cuda"
    else:
        device = "cpu"

    if model_name in valid_models:
        print("\n\033[1;35m*** Faster Whisper 自动下载模式 ***\033[0m\n")
    else:
        print("\n\033[1;35m*** Faster Whisper 本地模型模式 ***\033[0m\n")

    print(f"\033[1;34m🖥️ 运行模型: {model_name} \033[0m")
    print(f"\033[1;34m⚙️️ 运行方式: {device} \033[0m")
    print(f"\033[1;34m🎙️ VAD辅助: {vad} \033[0m")
    file_path = Path(file_path)
    try:
        if file_path.is_file():
            audio_file = file_path.open("rb")
        else:
            audio_file = (file_path / "output.mp3").open("rb")
    except FileNotFoundError as e:
        returnList[0] = {"error": f"文件错误：{str(e)}"}
        return {"error": f"文件错误：{str(e)}"}

    try:
        with audio_file:
            model = WhisperModel(model_name, device)
            settings = _load_faster_advanced_settings(model_name)
            transcribe_args = {
                "initial_prompt": prompt,
                "beam_size": beam_size,
                "temperature": temp,
                "condition_on_previous_text": settings["condition_prev_text"],
                "no_speech_threshold": settings["no_speech_threshold"],
                "log_prob_threshold": settings["log_prob_threshold"],
                "compression_ratio_threshold": settings["compression_ratio_threshold"],
            }
            if settings["hallucination_silence"] > 0:
                transcribe_args["hallucination_silence_threshold"] = settings["hallucination_silence"]
            if str(settings["hotwords"]).strip():
                transcribe_args["hotwords"] = str(settings["hotwords"]).strip()
            if vad:
                transcribe_args.update({
                    "vad_filter": vad,
                    "vad_parameters": dict(
                        threshold=settings["vad_threshold"],
                        min_silence_duration_ms=min_vad)
                })
            if lang != "自动识别":
                transcribe_args["language"] = lang

            print(f"\033[1;34m🛡️ 幻觉抑制: 上文承接={settings['condition_prev_text']}, "
                  f"静音阈值={settings['hallucination_silence']}s, 事后丢弃no_speech≥"
                  f"{settings['drop_no_speech_prob']}, 去重={settings['dedupe_repeat']}\033[0m")
            segments, info = model.transcribe(audio_file, **transcribe_args)
            segments = _filter_hallucinated_segments(list(segments), settings)

        result = FasterWhisperResultDict(segments, info)
        print(f"\033[1;34m📝 Whisper识别结果:\033[0m\n{result['text']}")
        returnList[0] = result
        return result

    except Exception as e:
        # 生成错误文件,防止再次扫描
        if fileName!=None:
            fileName = os.path.splitext(fileName)[0]
            erroFile = f"{fileName}_erro.txt"
            with open(erroFile, 'w', encoding='utf-8') as error_file:
                error_file.write(str(e))
        print(f"转录过程发生错误：{str(e)}")
        returnList[0] = {"error": f"转录过程发生错误：{str(e)}"}
        return {"error": f"转录过程发生错误：{str(e)}"}


def runWhisperSeperateProc(*args):
    try:
        # with ThreadPoolExecutor() as executor:
        #     future = executor.submit(FasterWhisperResult, *args)
        #     return future.result()
        with Manager() as manager:
            returnList = manager.list([None])
            p = Process(target=FasterWhisperResult, args=args + (returnList,))
            p.start()
            p.join()
            p.close()
            return returnList[0]
    except Exception as e:
        # 不调用 st.stop():目录批量调用方依赖 error dict 决定 continue,
        # st.stop() 会静默终止整个批量脚本(无红框、无 traceback)
        print(f"\033[1;31m❌ Whisper 独立进程异常: {type(e).__name__}: {e}\033[0m")
        traceback.print_exc()
        return {"error": f"Whisper 独立进程异常: {type(e).__name__}: {e}"}


QWEN_LANG_MAP = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "th": "Thai",
    "de": "German", "fr": "French", "ru": "Russian", "ko": "Korean",
    "vi": "Vietnamese", "it": "Italian", "ar": "Arabic", "es": "Spanish",
    "pt": "Portuguese", "hi": "Hindi",
}
QWEN_LANG_REVERSE = {v: k for k, v in QWEN_LANG_MAP.items()}

QWEN_SETTINGS_DEFAULTS = {
    "asr_path": "model/faster-whisper/Qwen3-ASR-1.7B",
    "aligner_path": "model/faster-whisper/Qwen3-ForcedAligner-0.6B",
    "gpu": True,
    "context": "",
    "max_new_tokens": 2048,
    "line_max_chars": 24,
    "line_gap_seconds": 1.0,
    "line_max_seconds": 7.0,
    # VAD 预筛:跳过纯非语音粗块(AV 素材喘息/BGM 占比极高),只把含语音块送模型
    "vad_enabled": True,
    "vad_threshold": 0.5,
    "vad_block_seconds": 30,
    "vad_min_speech_ratio": 0.02,
    # 稀疏幻觉过滤:强制日语时非语音段被编成文字,对齐器把字符摊到超长跨度上
    "sparse_item_min_seconds": 1.5,
    "sparse_item_max_seconds_per_char": 1.5,
    "sparse_line_min_seconds": 3.0,
    "sparse_line_min_chars_per_second": 0.5,
}


def _load_qwen_settings() -> dict:
    settings = dict(QWEN_SETTINGS_DEFAULTS)
    config_path = Path(__file__).resolve().parent.parent / "config" / "whisper.toml"
    try:
        with open(config_path, 'r', encoding="utf-8") as f:
            raw = toml.load(f).get("Qwen3ASR", {})
        for key in settings:
            if key in raw:
                settings[key] = raw[key]
    except Exception as e:
        print(f"\033[1;33m⚠️ Qwen3ASR 配置读取失败,使用默认值: {type(e).__name__}: {e}\033[0m")
    root = Path(__file__).resolve().parent.parent
    for path_key in ("asr_path", "aligner_path"):
        p = Path(str(settings[path_key]))
        if not p.is_absolute():
            settings[path_key] = str((root / p).resolve())
    return settings


def _join_qwen_item_texts(items: list) -> str:
    text = ""
    for item in items:
        t = item.text
        if text and re.match(r"^[A-Za-z0-9]", t) and not re.search(r"[\s(\[\{\"'‘“\-–—]$", text):
            text += " "
        text += t
    return re.sub(r"\s+([,.;:!?%])", r"\1", text).strip()


def _qwen_items_to_segments(items: list, max_chars: int, gap_seconds: float, max_line_seconds: float) -> list:
    AlignedItem = namedtuple("AlignedItem", ["text", "start_time", "end_time"])
    # 把 Qwen3-ForcedAligner 的字符/词级 items 聚成字幕行:
    # 句读标点断行优先,其次超长/时间空隙/单行最大时长兜底
    sentence_end = "。．.！？!?…"
    segments = []
    current = []

    def flush():
        if not current:
            return
        line_text = _join_qwen_item_texts(current)
        if line_text:
            segments.append({
                "start": float(current[0].start_time),
                "end": float(current[-1].end_time),
                "text": line_text,
            })

    for index, item in enumerate(items):
        item_text = item.text.strip()
        if not item_text:
            continue
        # 强制指定语言时模型对非语音段可能输出幻觉短句,对齐器无法对齐会给零时长 span,直接丢弃
        if float(item.end_time) <= float(item.start_time):
            continue
        # 空隙/超长在加入新 item 前断行,保证已产出的行不越界
        if current:
            gap = float(item.start_time) - float(current[-1].end_time)
            joined_duration = float(item.end_time) - float(current[0].start_time)
            if gap > gap_seconds or joined_duration > max_line_seconds:
                flush()
                current = []
        current.append(AlignedItem(item_text, float(item.start_time), float(item.end_time)))
        line_len = len(_join_qwen_item_texts(current))
        if item_text[-1:] in sentence_end or line_len >= max_chars:
            flush()
            current = []
    flush()
    return segments


def _qwen_filter_sparse_items(items: list, min_seconds: float = 1.5, max_seconds_per_char: float = 1.5) -> list:
    # 丢弃对齐器摊长的幻觉字符:时长超过门槛且每字占用超过 max_seconds_per_char。
    # 真实密集台词约 0.1~0.3s/字,实测幻觉粒子被摊到 2~97s/字;短呻吟/附和(<min_seconds)无条件保留。
    kept = []
    for item in items:
        duration = float(item.end_time) - float(item.start_time)
        text_len = len(item.text.strip())
        if duration > min_seconds and duration / max(text_len, 1) > max_seconds_per_char:
            continue
        kept.append(item)
    return kept


def _qwen_filter_sparse_segments(segments: list, min_seconds: float = 3.0,
                                 min_chars_per_second: float = 0.5) -> list:
    # 行级兜底:item 级漏网后,整行仍稀疏(长时长、极少字)则丢弃;短行不判罚。
    return [
        seg for seg in segments
        if (seg["end"] - seg["start"]) <= min_seconds
        or len(seg["text"]) / max(seg["end"] - seg["start"], 1e-6) >= min_chars_per_second
    ]


def _qwen_vad_blocks(total_seconds: float, windows: list, block_seconds: float = 60.0,
                     min_speech_ratio: float = 0.02) -> list:
    # VAD 粗筛:按 block_seconds 切块,块内 VAD 语音累计占比超过 min_speech_ratio 才保留该块。
    # windows 为秒级 (start,end) 列表;返回保留块在原音频中的 [(start,end), ...]。
    speech = [0.0] * (int(total_seconds // block_seconds) + 1)
    for start, end in windows:
        first = int(start // block_seconds)
        last = int(end // block_seconds)
        for idx in range(first, min(last + 1, len(speech))):
            block_start = idx * block_seconds
            speech[idx] += max(0.0, min(end, block_start + block_seconds) - max(start, block_start))
    blocks = []
    for idx, amount in enumerate(speech):
        start = idx * block_seconds
        if start >= total_seconds:
            break
        if amount / block_seconds >= min_speech_ratio:
            blocks.append((start, min(start + block_seconds, total_seconds)))
    return blocks


def _qwen_result_dict(lang_code: str, segments: list):
    return {
        'lang': lang_code,
        'text': ' '.join(segment['text'] for segment in segments),
        'segments': [{
            'id': index,
            'seek': 0,
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'tokens': [],
            'temperature': 0.0,
            'avg_logprob': 0.0,
            'compression_ratio': 0.0,
            'no_speech_prob': 0.0,
        } for index, segment in enumerate(segments)]
    }


def Qwen3ASRResult(file_path: str, lang: str, fileName: str, returnList):
    print("\n\033[1;35m*** Qwen3-ASR 本地模型模式 ***\033[0m\n")
    settings = _load_qwen_settings()
    file_path = Path(file_path)
    try:
        if file_path.is_file():
            audio_path = file_path
        else:
            audio_path = file_path / "output.mp3"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
    except FileNotFoundError as e:
        returnList[0] = {"error": f"文件错误：{str(e)}"}
        return {"error": f"文件错误：{str(e)}"}

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        msg = f"Qwen3-ASR 依赖未安装: {type(e).__name__}: {e}"
        print(f"\033[1;31m❌ {msg}\033[0m")
        returnList[0] = {"error": msg}
        return {"error": msg}

    qwen_lang = None
    if lang and lang != "自动识别":
        qwen_lang = QWEN_LANG_MAP.get(lang)
        if qwen_lang is None:
            print(f"\033[1;33m⚠️ 语言 {lang} 不在 Qwen3-ASR 支持表,回退自动识别\033[0m")

    try:
        import numpy as np
        import librosa

        use_gpu = bool(settings["gpu"]) and torch.cuda.is_available()
        device = "cuda:0" if use_gpu else "cpu"
        dtype = torch.bfloat16 if use_gpu else torch.float32
        print(f"\033[1;34m🖥️ 运行模型: Qwen3-ASR-1.7B + ForcedAligner\033[0m")
        print(f"\033[1;34m⚙️️ 运行方式: {device}\033[0m")
        print(f"\033[1;34m🌐 识别语言: {qwen_lang or '自动识别'}\033[0m")

        wav = librosa.load(str(audio_path), sr=16000, mono=True)[0].astype(np.float32)
        total_seconds = len(wav) / 16000
        blocks = [(0.0, total_seconds)]
        if bool(settings["vad_enabled"]):
            try:
                from faster_whisper.vad import get_vad_model, get_speech_timestamps, VadOptions
                vad_windows = get_speech_timestamps(
                    wav,
                    VadOptions(threshold=float(settings["vad_threshold"]),
                               min_silence_duration_ms=500, min_speech_duration_ms=250,
                               speech_pad_ms=200),
                )
                windows_sec = [(w["start"] / 16000, w["end"] / 16000) for w in vad_windows]
                blocks = _qwen_vad_blocks(
                    total_seconds, windows_sec,
                    block_seconds=float(settings["vad_block_seconds"]),
                    min_speech_ratio=float(settings["vad_min_speech_ratio"]),
                )
                print(f"\033[1;34m🔇 VAD预筛: {len(windows_sec)} 个语音窗口, "
                      f"送识别 {len(blocks)} 个音频块共 {sum(e-s for s,e in blocks):.0f}s "
                      f"/ 全长 {total_seconds:.0f}s\033[0m")
            except Exception as vad_err:
                print(f"\033[1;33m⚠️ VAD 预筛失败,回退整段识别: {type(vad_err).__name__}: {vad_err}\033[0m")
                blocks = [(0.0, total_seconds)]

        if not blocks:
            print("\033[1;33m⚠️ VAD 未检测到任何语音,跳过模型识别(纯非语音素材)\033[0m")
            result = _qwen_result_dict(lang if lang and lang != "自动识别" else "", [])
            returnList[0] = result
            return result

        model = Qwen3ASRModel.from_pretrained(
            settings["asr_path"],
            forced_aligner=settings["aligner_path"],
            forced_aligner_kwargs={"dtype": dtype, "device_map": device},
            dtype=dtype,
            device_map=device,
            max_new_tokens=int(settings["max_new_tokens"]),
        )

        Item = namedtuple("Item", ["text", "start_time", "end_time"])
        all_items = []
        detected_languages = []
        batch_blocks = 8
        for group_start in range(0, len(blocks), batch_blocks):
            group = blocks[group_start:group_start + batch_blocks]
            audios = [(wav[int(s * 16000):int(e * 16000)], 16000) for s, e in group]
            transcriptions = model.transcribe(
                audio=audios,
                language=qwen_lang,
                context=str(settings["context"]),
                return_time_stamps=True,
            )
            for (block_offset, _), transcription in zip(group, transcriptions):
                if transcription.language:
                    detected_languages.append(transcription.language)
                if transcription.time_stamps:
                    for item in transcription.time_stamps.items:
                        all_items.append(Item(
                            item.text,
                            float(item.start_time) + block_offset,
                            float(item.end_time) + block_offset,
                        ))
            print(f"\033[1;34m   已处理 {min(group_start + batch_blocks, len(blocks))}/{len(blocks)} 块\033[0m")

        items_before = len(all_items)
        all_items = _qwen_filter_sparse_items(
            all_items,
            min_seconds=float(settings["sparse_item_min_seconds"]),
            max_seconds_per_char=float(settings["sparse_item_max_seconds_per_char"]),
        )
        sparse_dropped = items_before - len(all_items)
        segments = _qwen_items_to_segments(
            all_items,
            max_chars=int(settings["line_max_chars"]),
            gap_seconds=float(settings["line_gap_seconds"]),
            max_line_seconds=float(settings["line_max_seconds"]),
        )
        segments = _qwen_filter_sparse_segments(
            segments,
            min_seconds=float(settings["sparse_line_min_seconds"]),
            min_chars_per_second=float(settings["sparse_line_min_chars_per_second"]),
        )
        if sparse_dropped:
            print(f"\033[1;33m⚠️ 稀疏幻觉过滤: 丢弃 {sparse_dropped} 个被摊长的字符/词\033[0m")
        if qwen_lang:
            detected_code = lang
        else:
            merged_language = max(set(detected_languages), key=detected_languages.count) if detected_languages else ""
            detected_code = QWEN_LANG_REVERSE.get(merged_language, "")
        result = _qwen_result_dict(detected_code, segments)
        if not segments:
            print("\033[1;33m⚠️ 未识别到任何语音片段(可能为纯静音/非语音素材)\033[0m")
        print(f"\033[1;34m📝 Qwen3-ASR 识别结果:\033[0m\n{result['text']}")
        returnList[0] = result
        return result

    except Exception as e:
        if fileName is not None:
            erroFile = f"{os.path.splitext(fileName)[0]}_erro.txt"
            with open(erroFile, 'w', encoding='utf-8') as error_file:
                error_file.write(str(e))
        traceback.print_exc()
        print(f"转录过程发生错误：{str(e)}")
        returnList[0] = {"error": f"转录过程发生错误：{str(e)}"}
        return {"error": f"转录过程发生错误：{str(e)}"}


def runQwenSeperateProc(*args):
    try:
        with Manager() as manager:
            returnList = manager.list([None])
            p = Process(target=Qwen3ASRResult, args=args + (returnList,))
            p.start()
            p.join()
            p.close()
            return returnList[0]
    except Exception as e:
        print(f"\033[1;31m❌ Qwen3ASR 独立进程异常: {type(e).__name__}: {e}\033[0m")
        traceback.print_exc()
        return {"error": f"Qwen3ASR 独立进程异常: {type(e).__name__}: {e}"}



BATCH_MAX_LINES = 150
BATCH_MAX_CHARS = 6000
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[|｜]\s*(.*)$")
_NUMBERED_ANY_RE = re.compile(r"^\s*\d+\s*[|｜]", re.M)
_REFUSAL_MARKERS = (
    "不能协助", "无法协助", "不能提供", "无法提供", "不能翻译", "无法翻译",
    "抱歉", "不便", "违反", "内容政策", "不适宜", "敏感内容",
    "i can't", "i can not", "i cannot", "i'm unable", "i am unable",
    "cannot assist", "can't assist", "unable to assist", "content policy",
)


class TranslationRefused(Exception):
    pass


def _looks_like_refusal(content):
    text = str(content or "").strip()
    if not text:
        return True
    if _NUMBERED_ANY_RE.search(text):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _chunk_translation_items(texts, max_lines=BATCH_MAX_LINES, max_chars=BATCH_MAX_CHARS):
    chunk = []
    char_count = 0
    for index, text in enumerate(texts):
        text = str(text)
        if chunk and (len(chunk) >= max_lines or char_count + len(text) > max_chars):
            yield chunk
            chunk = []
            char_count = 0
        chunk.append((index, text))
        char_count += len(text)
    if chunk:
        yield chunk


def _build_numbered_block(items):
    lines = []
    for index, text in items:
        flat = str(text).replace("\r", " ").replace("\n", " ").replace("|", "丨")
        lines.append(f"{index + 1}|{flat}")
    return "\n".join(lines)


def _parse_numbered_translations(content, expected_numbers):
    numbered = {}
    for line in str(content).splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            number = int(match.group(1))
            if number not in numbered:
                numbered[number] = match.group(2).strip()
    expected = list(expected_numbers)
    if all(number in numbered for number in expected):
        return {number: numbered[number] for number in expected}

    plain_lines = [line.strip() for line in str(content).splitlines() if line.strip()]
    if len(plain_lines) == len(expected):
        aligned = {}
        for number, line in zip(expected, plain_lines):
            match = _NUMBERED_LINE_RE.match(line)
            aligned[number] = match.group(2).strip() if match else line
        return aligned

    missing = [number for number in expected if number not in numbered][:20]
    raise ValueError(
        f"批量翻译结果无法与原文对齐:期望 {len(expected)} 条,按编号解析出 {len(numbered)} 条,"
        f"缺失编号示例 {missing}。请检查模型输出格式或调小批量分块大小。"
    )


def _build_batch_user_prompt(user_prompt, block):
    template = user_prompt.replace("{combined_text}", "")
    if "{subtitles}" in template:
        return template.replace("{subtitles}", block)
    return template.rstrip() + "\n\n" + block


def _translate_one(system_prompt, user_prompt, index, original, send):
    block = _build_numbered_block([(index, original)])
    content = send(system_prompt, _build_batch_user_prompt(user_prompt, block))
    if _looks_like_refusal(content):
        raise TranslationRefused(str(content)[:120])
    parsed = _parse_numbered_translations(content, [index + 1])
    return parsed[index + 1]


def _translate_chunk(system_prompt, user_prompt, items, send):
    block = _build_numbered_block(items)
    content = send(system_prompt, _build_batch_user_prompt(user_prompt, block))
    if _looks_like_refusal(content):
        raise TranslationRefused(str(content)[:120])
    parsed = _parse_numbered_translations(content, [index + 1 for index, _ in items])
    return {index: parsed[index + 1] for index, _ in items}


def _batch_translate(system_prompt, user_prompt, texts, send, wait_time=0,
                     max_lines=BATCH_MAX_LINES, max_chars=BATCH_MAX_CHARS):
    translations = [None] * len(texts)
    chunks = list(_chunk_translation_items(texts, max_lines, max_chars))
    total = len(chunks)
    for chunk_index, items in enumerate(chunks):
        try:
            mapping = _translate_chunk(system_prompt, user_prompt, items, send)
        except TranslationRefused:
            # 整批被安全策略拒答:不产出半成品,交由上层跳过该文件的翻译 SRT
            raise
        except ValueError:
            # 整批输出无法对齐(截断/格式漂移):逐条重试以隔离问题条目
            mapping = {}
            for index, original in items:
                try:
                    mapping[index] = _translate_one(system_prompt, user_prompt, index, original, send)
                except TranslationRefused:
                    raise
                except ValueError:
                    mapping[index] = original
            print(f"\033[1;33m⚠️ 本批 {len(items)} 条整批无法对齐,已逐条重试\033[0m")
        for index, _ in items:
            translations[index] = mapping[index]
        print(f"\033[1;34m📦 批量翻译进度: {chunk_index + 1}/{total} 块(本块 {len(items)} 条)\033[0m")
        if wait_time and chunk_index < total - 1:
            time.sleep(wait_time)
    return translations


def _compose_bilingual(original, translated, srt):
    if srt == "原始语言为首":
        return f"{original}\n{translated}"
    if srt == "目标语言为首":
        return f"{translated}\n{original}"
    return translated


def _provider_batch_translate(system_prompt, user_prompt, texts, api_key, base_url, model,
                              wait_time, all_in_one, global_key, global_url):
    print("\033[1;34m📝 翻译内容: \033[0m")

    if all_in_one:
        print(f"\033[1;34m🌐 代理地址: {global_url}\033[0m")
        client = OpenAI(api_key=global_key, base_url=global_url)

        def send(system, user_content):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_content}])
            return response.choices[0].message.content

        return _batch_translate(system_prompt, user_prompt, texts, send, wait_time)

    if "gpt" in model:
        if base_url != "https://api.openai.com/v1":
            print(f"\033[1;34m🌐 代理地址: {base_url}\033[0m")
        client = OpenAI(api_key=api_key, base_url=base_url)

        def send(system, user_content):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_content}])
            return response.choices[0].message.content

        return _batch_translate(system_prompt, user_prompt, texts, send, wait_time)

    if "claude" in model:
        if base_url != "https://api.anthropic.com/v1/messages":
            print(f"\033[1;34m🌐 代理地址: {base_url}\033[0m")
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

        def send(system, user_content):
            message = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user_content}])
            return message.content[0]["text"]

        return _batch_translate(system_prompt, user_prompt, texts, send, wait_time)

    if "gemini" in model:
        if base_url is None:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)

            def send(system, user_content):
                return gemini_model.generate_content(system + user_content).text

            return _batch_translate(system_prompt, user_prompt, texts, send, wait_time)

        print(f"\033[1;34m🌐 代理地址: {base_url}\033[0m")

        def send(system, user_content):
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.8
            })
            headers = {"Content-Type": "application/json", "Authorization": api_key}
            response = requests.request("POST", base_url, headers=headers, data=payload)
            return response.json()["choices"][0]["message"]["content"]

        return _batch_translate(system_prompt, user_prompt, texts, send, wait_time)

    print(f"\033[1;34m🌐 代理地址: {base_url}\033[0m")
    client = OpenAI(api_key=api_key, base_url=base_url)

    def send(system, user_content):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_content}])
        return response.choices[0].message.content

    return _batch_translate(system_prompt, user_prompt, texts, send, wait_time)


def translate(system_prompt, user_prompt, api_key, base_url, model, result, wait_time, srt):
    print("\n\033[1;35m*** API 调用翻译模式 ***\033[0m\n")
    print(f"\033[1;34m🈴 翻译模型: {model}\033[0m")
    global_key, global_url, all_in_one = get_global_info()

    segments = result['segments']
    texts = [str(segment['text']) for segment in segments]

    if result.get('lang') == 'zh':
        translations = list(texts)
    else:
        translations = _provider_batch_translate(
            system_prompt, user_prompt, texts, api_key, base_url, model,
            wait_time, all_in_one, global_key, global_url)
        for answer in translations:
            print(answer)

    for index, (original, translated) in enumerate(zip(texts, translations)):
        segments[index]['text'] = _compose_bilingual(original, translated, srt)
    return result


def local_translate(system_prompt, user_prompt, api_key, base_url, model, result, srt):
    print("\n\033[1;35m*** 本地大模型翻译模式 ***\033[0m\n")
    print(f"\033[1;34m🈴 翻译模型: {model}\033[0m")
    print(f"\033[1;34m📝 翻译内容: \033[0m")
    client = OpenAI(api_key=api_key, base_url=base_url)
    segments = result['segments']
    segment_id = 0

    context_size = 2
    previous_texts = []
    original_user_prompt = user_prompt

    for segment in segments:
        text = segment['text']
        if len(text) > 20:
            text = str(text)[1:20]

        if len(previous_texts) > 0:
            combined_text = "\n".join(previous_texts[-context_size:]) + "\n"
        else:
            combined_text = text
        current_user_prompt = original_user_prompt.replace("{combined_text}", combined_text).replace("{subtitles}", "")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_user_prompt + str(text)}
            ])

        answer = response.choices[0].message.content
        if srt == "原始语言为首":
            result['segments'][segment_id]['text'] = str(text) + "\n" + str(answer)
        elif srt == "目标语言为首":
            result['segments'][segment_id]['text'] = str(answer) + "\n" + str(text)
        else:
            result['segments'][segment_id]['text'] = answer
        segment_id += 1
        print(answer)

        previous_texts.append(text)
        if len(previous_texts) > context_size:
            previous_texts.pop(0)

    return result


def milliseconds_to_srt_time_format(milliseconds):  # 将毫秒表示的时间转换为SRT字幕的时间格式
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def generate_srt_from_result(result):  # 格式化为SRT字幕的形式
    segments = result['segments']
    srt_content = ''
    segment_id = 1
    for segment in segments:
        start_time = int(segment['start'] * 1000)
        end_time = int(segment['end'] * 1000)
        text = segment['text'].replace("Text:", "").replace("翻译：", "")

        srt_content += f"{segment_id}\n"
        srt_content += f"{milliseconds_to_srt_time_format(start_time)} --> {milliseconds_to_srt_time_format(end_time)}\n"
        srt_content += f"{text}\n\n"
        segment_id += 1
    return srt_content


def generate_srt_from_result_2(result, font, font_size, font_color):  # 格式化为SRT字幕的形式
    segments = result['segments']
    srt_content = ''
    segment_id = 1
    for segment in segments:
        start_time = int(segment['start'] * 1000)
        end_time = int(segment['end'] * 1000)
        text = segment['text']

        srt_content += f"{segment_id}\n"
        srt_content += f"{milliseconds_to_srt_time_format(start_time)} --> {milliseconds_to_srt_time_format(end_time)}\n"
        srt_content += f"<font color={font_color}><font face={font}><font size={font_size}> {text}\n\n"
        segment_id += 1
    return srt_content


def check_cuda_installed():
    if torch.cuda.is_available():
        return True
    else:
        return False


def check_ffmpeg_hwaccel():
    try:
        result = subprocess.run(["ffmpeg", "-hwaccels"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        hwaccels = result.stdout.splitlines()
        if 'cuda' in hwaccels:
            return True
        else:
            return False
    except Exception as e:
        return False


def srt_mv(log, name, crf, quality, setting, path, font, font_size, font_color, subtitle_model):  # 视频合成字幕
    font_color = font_color.lstrip('#')  # 去掉 '#' 符号
    bb = font_color[4:6]
    gg = font_color[2:4]
    rr = font_color[0:2]
    font_color = f"&H{bb}{gg}{rr}&"
    cuda_installed = check_cuda_installed()
    cuda_supported = check_ffmpeg_hwaccel() if cuda_installed else False

    if subtitle_model == "硬字幕":
        if cuda_supported:
            command = f"""ffmpeg -loglevel {log} -hwaccel cuda -i {name} -lavfi "subtitles=output.srt:force_style='FontName={font},FontSize={font_size},PrimaryColour={font_color},Outline=1,Shadow=0,BackColour=&H9C9C9C&,Bold=-1,Alignment=2'" -preset {quality} -c:v {setting} -crf {crf} -y -c:a copy output.mp4"""
        else:
            command = f"""ffmpeg -loglevel {log} -i {name} -lavfi "subtitles=output.srt:force_style='FontName={font},FontSize={font_size},PrimaryColour={font_color},Outline=1,Shadow=0,BackColour=&H9C9C9C&,Bold=-1,Alignment=2'" -preset {quality} -c:v libx264 -crf {crf} -y -c:a copy output.mp4"""
    else:
        if cuda_supported:
            command = f"""ffmpeg -loglevel {log} -hwaccel cuda -i {name} -i output_with_style.srt -c:v {setting} -crf {crf} -y -c:a copy -c:s mov_text -preset {quality} output.mp4"""
        else:
            command = f"""ffmpeg -loglevel {log} -i {name} -i output_with_style.srt -c:v libx264 -crf {crf} -y -c:a copy -c:s mov_text -preset {quality} output.mp4"""

    subprocess.run(command, shell=True, cwd=path)


def parse_srt_file(srt_content, srt_setting="关闭"):
    lines = srt_content.strip().split('\n')
    subtitles = []

    if srt_setting == "关闭":
        current_subtitle = {'索引': '', '起始': '', '结束': '', '字幕': ''}
    else:
        current_subtitle = {'索引': '', '起始': '', '结束': '', '字幕': '', '翻译字幕': ''}

    for line in lines:
        line = line.strip()

        if line.isdigit():
            if current_subtitle['索引']:
                subtitles.append(current_subtitle)
            if srt_setting == "关闭":
                current_subtitle = {'索引': str(line), '起始': '', '结束': '', '字幕': ''}
            else:
                current_subtitle = {'索引': str(line), '起始': '', '结束': '', '字幕': '', '翻译字幕': ''}
        elif '-->' in line:
            start_time, end_time = line.split('-->')
            current_subtitle['起始'] = start_time.strip()
            current_subtitle['结束'] = end_time.strip()
        elif line != '':
            if srt_setting == "关闭":
                current_subtitle['字幕'] += line + ' '
            elif srt_setting == "原始语言为首":
                if not current_subtitle['字幕']:
                    current_subtitle['字幕'] = line
                else:
                    current_subtitle['翻译字幕'] = line
            elif srt_setting == "目标语言为首":
                if not current_subtitle['翻译字幕']:
                    current_subtitle['翻译字幕'] = line
                else:
                    current_subtitle['字幕'] = line

    if current_subtitle['索引']:
        subtitles.append(current_subtitle)

    df = pd.DataFrame(subtitles)
    return df


def convert_to_srt(edited_data, srt_setting="关闭"):
    subtitles = ''
    for index, row in edited_data.iterrows():
        start_time = row['起始']
        end_time = row['结束']

        if srt_setting == "关闭":
            content = row.get('字幕', '')
            subtitle = f"{index + 1}\n{start_time} --> {end_time}\n{content.strip()}\n\n"
        elif srt_setting == "原始语言为首":
            content1 = row.get('字幕', '')
            content2 = row.get('翻译字幕', '')
            subtitle = f"{index + 1}\n{start_time} --> {end_time}\n{content1}\n{content2}\n\n"
        elif srt_setting == "目标语言为首":
            content1 = row.get('字幕', '')
            content2 = row.get('翻译字幕', '')
            subtitle = f"{index + 1}\n{start_time} --> {end_time}\n{content2}\n{content1}\n\n"

        subtitles += subtitle
    return subtitles


def show_video(path, name):
    video_file = open(path + "/" + name, 'rb')
    video_bytes = video_file.read()
    return video_bytes


def add_font_settings(srt_content, font_color, font_face, font_size, srt_setting="关闭"):
    timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})")
    lines = srt_content.split("\n")
    result = []
    is_translation = False

    for line in lines:
        if line.isdigit():
            result.append(line)
            is_translation = False
        elif timestamp_pattern.match(line):
            result.append(line)
            is_translation = False
        elif line.strip() == "":
            result.append(line)
            is_translation = False
        else:
            if srt_setting == "关闭":
                formatted_line = f'<font color="{font_color}" face="{font_face}" size="{font_size}">{line}</font>'
                result.append(formatted_line)
            elif srt_setting == "原始语言为首":
                if not is_translation:
                    formatted_line = f'<font color="{font_color}" face="{font_face}" size="{font_size}">{line}</font>'
                    is_translation = True
                else:
                    formatted_line = f'<font color="{font_color}" face="{font_face}" size="{font_size}">{line}</font>'
                result.append(formatted_line)
            elif srt_setting == "目标语言为首":
                if not is_translation:
                    formatted_line = f'<font color="{font_color}" face="{font_face}" size="{font_size}">{line}</font>'
                    is_translation = True
                else:
                    formatted_line = f'<font color="{font_color}" face="{font_face}" size="{font_size}">{line}</font>'
                result.append(formatted_line)
    return "\n".join(result)


def srt_to_vtt(srt_content):
    blocks = srt_content.strip().split('\n\n')
    vtt_lines = ['WEBVTT\n']

    for block in blocks:
        lines = block.strip().split('\n')

        if len(lines) >= 2:
            time_range = lines[1].replace(',', '.')
            text = '\n'.join(lines[2:])
            vtt_lines.append(f'{time_range}\n{text}\n')
    vtt_content = '\n'.join(vtt_lines)

    return vtt_content


def srt_to_ass(srt_content, fontname="Arial", size=20, color="&H00FFFFFF"):
    lines = srt_content.strip().split('\n\n')
    ass_header = (
        '[Script Info]\n'
        'Title: Converted from SRT\n'
        'ScriptType: v4.00+\n'
        'Collisions: Normal\n'
        'PlayDepth: 0\n\n'
        '[V4+ Styles]\n'
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n'
        f'Style: Default,{fontname},{size},{color},&H000000FF,&H00000000,-1,0,0,1,1.0,0.0,2,10,10,10,1\n\n'
        '[Events]\n'
        'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
    )
    ass_content = ass_header

    for line in lines:
        parts = line.strip().split('\n')
        start, end = parts[1].split(' --> ')
        start = start.replace(',', '.')
        end = end.replace(',', '.')
        text = '\\N'.join(parts[2:])
        ass_content += f'Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n'

    return ass_content


def srt_to_sbv(srt_content):
    lines = srt_content.strip().split('\n\n')
    sbv_content = ''

    for block in lines:
        parts = block.strip().split('\n')
        start, end = parts[1].split(' --> ')
        start = convert_srt_time_to_sbv(start)
        end = convert_srt_time_to_sbv(end)
        text = '\n'.join(parts[2:])
        sbv_content += f'{start},{end}\n{text}\n\n'

    return sbv_content


def convert_srt_time_to_sbv(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    milliseconds = int(milliseconds) / 10
    sbv_time = f'{hours}:{minutes}:{seconds}.{int(milliseconds):02d}'

    return sbv_time


def local_translate_srt(system_prompt, user_prompt, api_key, base_url, model, srt_content, srt):
    print("\n\033[1;35m*** 本地大模型翻译模式 ***\033[0m\n")
    print(f"\033[1;34m🈴 翻译模型: {model}\033[0m")
    print(f"\033[1;34m📝 翻译内容: \033[0m")
    client = OpenAI(api_key=api_key, base_url=base_url)
    segment_id = 0

    context_size = 2
    previous_texts = []
    original_user_prompt = user_prompt

    for segment in srt_content:
        text = segment['text']
        if len(text) > 20:
            text = str(text)[1:20]

        if len(previous_texts) > 0:
            combined_text = "\n".join(previous_texts[-context_size:]) + "\n"
        else:
            combined_text = text
        current_user_prompt = original_user_prompt.replace("{combined_text}", combined_text).replace("{subtitles}", "")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_user_prompt + str(text)}
            ])
        answer = response.choices[0].message.content
        if srt == "原始语言为首":
            srt_content[segment_id]['text'] = str(text) + "\n" + str(answer)
        elif srt == "目标语言为首":
            srt_content[segment_id]['text'] = str(answer) + "\n" + str(text)
        else:
            srt_content[segment_id]['text'] = answer
        segment_id += 1
        print(answer)

        previous_texts.append(text)
        if len(previous_texts) > context_size:
            previous_texts.pop(0)

    return srt_content


def translate_srt(system_prompt, user_prompt, api_key, base_url, model, srt_content, wait_time, srt):
    print("\n\033[1;35m*** API 调用翻译模式 ***\033[0m\n")
    print(f"\033[1;34m🈴 翻译模型: {model}\033[0m")
    global_key, global_url, all_in_one = get_global_info()

    texts = [str(item['text']) for item in srt_content]
    translations = _provider_batch_translate(
        system_prompt, user_prompt, texts, api_key, base_url, model,
        wait_time, all_in_one, global_key, global_url)
    for answer in translations:
        print(answer)

    for index, (original, translated) in enumerate(zip(texts, translations)):
        srt_content[index]['text'] = _compose_bilingual(original, translated, srt)
    return srt_content


def read_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        srt = file.read().split('\n\n')
        timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})")
        subtitles = []

        for block in srt:
            lines = block.split('\n')
            if len(lines) >= 3:
                num = lines[0]
                times = lines[1]
                text = "\n".join(lines[2:])

                if timestamp_pattern.match(times):
                    subtitles.append({
                        'number': num,
                        'time': times,
                        'text': text
                    })
    return subtitles


def extract_frames(video_path, output_dir, time_interval=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"\033[1;31m❌ 无法读取视频关键帧，请检查此目录是否存在:\033[0m\033[1;34m {video_path} \033[0m")
        st.toast("无法读取视频关键帧", icon=":material/release_alert:")
        st.stop()

    timestamp = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        current_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_timestamp >= timestamp + time_interval:
            timestamp = current_timestamp
            frame_filename = os.path.join(output_dir, f'frame_{int(timestamp)}.png')
            cv2.imwrite(frame_filename, frame)
    video.release()


def write_llms(view, language, key, url, model, text, token, temp):
    print(f"\033[1;34m🈴 写作模型: {model}\033[0m")
    global_key, global_url, all_in_one = get_global_info()
    if all_in_one:

        client = OpenAI(api_key=global_key, base_url=global_url)
        print(f"\033[1;34m🌐 代理地址: {url}\033[0m")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "你是一个知识渊博且擅长分析的写作助手。你的任务是帮助用户基于文本内容重新创作一篇文章或总结，需遵循以下要求：1. 确保文章结构合理 2. 根据用户的要求调整文章的语气、风格、视角 3.确保文章结构合理，逻辑清晰"},
                {"role": "user",
                 "content": f"请你根据下面的文本内容写一篇{language}的文章或总结，写作时请遵循以下规则：1. 请以{view}的视角写作 2. 总篇幅不得少于{token}字 3.确保逻辑清晰、核心内容完整。文本内容：\n{text}"}
            ])
        answer = response.choices[0].message.content
        print(f"\033[1;34m📝 写作内容: \033[0m")
        print(answer)

    else:
        if "gpt" in model:
            print(f"\033[1;34m🌐 代理地址: {url}\033[0m")
            print(f"\033[1;34m📝 写作内容: \033[0m")
            client = OpenAI(api_key=key, base_url=url)
            response = client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=[
                    {"role": "system",
                     "content": "你是一个知识渊博且擅长分析的写作助手。你的任务是帮助用户基于文本内容重新创作一篇文章或总结，需遵循以下要求：1. 确保文章结构合理 2. 根据用户的要求调整文章的语气、风格、视角 3.确保文章结构合理，逻辑清晰"},
                    {"role": "user",
                     "content": f"请你根据下面的文本内写一篇{language}的文章或总结，写作时请遵循以下规则：1. 请以{view}的视角写作 2. 总篇幅不得少于{token}字 3.确保逻辑清晰、核心内容完整。文本内容：\n{text}"}
                ]
            )
            answer = response.choices[0].message.content
            print(answer)

        elif "claude" in model:
            print(f"\033[1;34m🌐 代理地址: {url}\033[0m")
            print(f"\033[1;34m📝 写作内容: \033[0m")
            client = anthropic.Anthropic(api_key=key, base_url=url)
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=temp,
                system="你是一个知识渊博且擅长分析的写作助手。你的任务是帮助用户基于文本内容重新创作一篇文章或总结，需遵循以下要求：1. 确保文章结构合理 2. 根据用户的要求调整文章的语气、风格、视角 3.确保文章结构合理，逻辑清晰",
                messages=[{"role": "user",
                           "content": f"请你根据下面的文本内容写一篇{language}的文章或总结，写作时请遵循以下规则：1. 请以{view}的视角写作 2. 总篇幅不得少于{token}字 3.确保逻辑清晰、核心内容完整。文本内容：\n{text}"}]
            )

            answer = message.content[0]['text']

        elif "gemini" in model:
            if url is None:
                print("- Python SDK 调用：如果需要使用代理，请填写 BASE_URL，系统将自动启用代理模式！")
                print(f"\033[1;34m🌐 代理地址: {url}\033[0m")
                genai.configure(client_options={"api_key": key})
                model = genai.GenerativeModel(model)
                answer = model.generate_content(
                    "请你根据下面的文本内容写一篇{language}的文章或总结，写作时请遵循以下规则：1. 请以{view}的视角写作 2. 总篇幅不得少于{token}字 3.确保逻辑清晰、核心内容完整。文本内容：\n{text}")
                print(f"\033[1;34m📝 写作内容: \033[0m")
                print(answer)
            else:
                print("- Request 请求调用（适用于代理模式，若不想使用Request模式，请把BASE_URL留空！)")
                print(f"\033[1;34m🌐 代理地址: {url}\033[0m")
                print(f"\033[1;34m📝 写作内容: \033[0m")

                payload = json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system",
                         "content": "你是一个知识渊博且擅长分析的写作助手。你的任务是帮助用户基于文本内容重新创作一篇文章或总结，需遵循以下要求：1. 确保文章结构合理 2. 根据用户的要求调整文章的语气、风格、视角 3.确保文章结构合理，逻辑清晰"},
                        {"role": "user",
                         "content": f"请你根据下面的文本内容写一篇{language}的文章或总结，写作时请遵循以下规则：1. 请以{view}的视角写作 2. 总篇幅不得少于{token}字 3.确保逻辑清晰、核心内容完整。文本内容：\n{text}"}
                    ],
                    "temperature": 0.8
                })
                headers = {'Content-Type': 'application/json', 'Authorization': key}
                response = requests.request("POST", url, headers=headers, data=payload)
                answer = response.json()

                try:
                    answer = answer["choices"][0]["message"]["content"]
                    print(answer)
                except Exception as e:
                    print("\n\033[1;31m------- 请阅读下方报错内容 -------\033[0m\n")
                    print(f"\033[1;31m❌ 发生错误: {type(e).__name__}\033[0m")
                    print(f"\033[1;33m💬 错误详情: \033[0m\n{answer}\n")
                    print("\033[1;31m------- 请根据报错内容检查你的 API Key 或代理配置 -------\033[0m\n")
                    raise e

        else:
            client = OpenAI(api_key=key, base_url=url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "你是一个知识渊博且擅长分析的写作助手。你的任务是帮助用户基于文本内容重新创作一篇文章或总结，需遵循以下要求：1. 确保文章结构合理 2. 根据用户的要求调整文章的语气、风格、视角 3.确保文章结构合理，逻辑清晰"},
                    {"role": "user",
                     "content": f"请你根据下面的文本内容写一篇{language}的文章或总结，写作时请遵循以下规则：1. 请以{view}的视角写作 2. 总篇幅不得少于{token}字 3.确保逻辑清晰、核心内容完整。文本内容：\n{text}"}
                ])
            answer = response.choices[0].message.content
            print(f"\033[1;34m📝 写作内容: \033[0m")
            print(answer)

    return answer


def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def readSrt(srtName):
    transcribedSubtitles = []
    srts = pysrt.open(srtName)
    for index in range(len(srts)):
        text: str = srts[index].text
        startTime: float = srts[index].start
        endTime: float = srts[index].end
        transcribedSubtitles.append(((startTime, endTime), text))
    return transcribedSubtitles

def get_duration_from_moviepy(url):
    clip = VideoFileClip(url)
    try:
        return clip.duration
    finally:
        clip.close()