import sys
import unittest
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import public


Item = namedtuple("Item", ["text", "start_time", "end_time"])


def chars(text, start=0.0, step=0.1):
    items = []
    t = start
    for ch in text:
        items.append(Item(ch, t, t + step))
        t += step
    return items


class QwenSegmentTests(unittest.TestCase):
    def test_splits_on_sentence_end(self):
        items = chars("気持ちいい。") + chars("もっとして。", start=2.0)
        segments = public._qwen_items_to_segments(items, max_chars=24, gap_seconds=1.0, max_line_seconds=7.0)
        self.assertEqual([s["text"] for s in segments], ["気持ちいい。", "もっとして。"])
        self.assertAlmostEqual(segments[0]["start"], 0.0)
        self.assertAlmostEqual(segments[0]["end"], 0.6)
        self.assertAlmostEqual(segments[1]["start"], 2.0)

    def test_forces_break_on_max_chars(self):
        items = chars("あ" * 30)
        segments = public._qwen_items_to_segments(items, max_chars=24, gap_seconds=1.0, max_line_seconds=999)
        self.assertEqual([len(s["text"]) for s in segments], [24, 6])

    def test_forces_break_on_gap(self):
        items = chars("好き", start=0.0) + chars("嫌い", start=5.0)
        segments = public._qwen_items_to_segments(items, max_chars=24, gap_seconds=1.0, max_line_seconds=999)
        self.assertEqual([s["text"] for s in segments], ["好き", "嫌い"])

    def test_forces_break_on_max_duration(self):
        items = chars("おはよう", start=0.0, step=2.0)  # 每行 0-10s
        segments = public._qwen_items_to_segments(items, max_chars=999, gap_seconds=999, max_line_seconds=7.0)
        self.assertTrue(len(segments) >= 2)
        for segment in segments:
            self.assertLessEqual(segment["end"] - segment["start"], 7.0 + 1e-6)

    def test_blank_items_skipped(self):
        items = [Item("あ", 0.0, 0.1), Item(" ", 0.1, 0.3), Item("い", 0.3, 0.4)]
        segments = public._qwen_items_to_segments(items, max_chars=24, gap_seconds=1.0, max_line_seconds=7.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "あい")

    def test_empty_input(self):
        self.assertEqual(public._qwen_items_to_segments([], 24, 1.0, 7.0), [])

    def test_latin_word_spacing(self):
        items = [
            Item("Oh", 0.0, 0.2), Item("my", 0.2, 0.4), Item("god", 0.4, 0.6),
            Item(",", 0.6, 0.65), Item("yes", 0.8, 1.0), Item(".", 1.0, 1.05),
        ]
        segments = public._qwen_items_to_segments(items, max_chars=24, gap_seconds=1.0, max_line_seconds=7.0)
        self.assertEqual(segments[0]["text"], "Oh my god, yes.")

    def test_result_dict_shape_matches_faster_contract(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "はい"}]
        result = public._qwen_result_dict("ja", segments)
        self.assertEqual(set(result.keys()), {"lang", "text", "segments"})
        segment_keys = {"id", "seek", "start", "end", "text", "tokens",
                        "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"}
        self.assertEqual(set(result["segments"][0].keys()), segment_keys)
        self.assertEqual(result["lang"], "ja")


    def test_zero_duration_items_dropped(self):
        # 强制语言时非语音幻觉短句,对齐器给 [0,0] 零时长 span
        items = [Item("は", 0.0, 0.0), Item("い", 0.0, 0.0)]
        self.assertEqual(public._qwen_items_to_segments(items, 24, 1.0, 7.0), [])


    def test_stripped_item_text_no_inner_space(self):
        items = [Item("近く", 0.0, 0.2), Item(" に", 0.2, 0.4)]
        segments = public._qwen_items_to_segments(items, max_chars=24, gap_seconds=1.0, max_line_seconds=7.0)
        self.assertEqual(segments[0]["text"], "近くに")


class QwenSparseFilterTests(unittest.TestCase):
    # AV 场景:强制日语时模型在喘息/BGM 段编出幻觉文字,对齐器无法对齐,
    # 把单个字符均匀摊到很长的音频跨度上(实测单字 span 6~97s,正常台词 0.1~0.3s/字)

    def test_stretched_single_char_item_dropped(self):
        items = [Item("て", 1596.9, 1643.8)]  # 实测 47s 单粒子幻觉段
        self.assertEqual(public._qwen_filter_sparse_items(items), [])

    def test_real_short_moan_kept(self):
        # 真实呻吟/附和短但发音密集(时长短于门槛一律保留)
        items = [Item("うん", 10.0, 10.7), Item("はい", 12.0, 12.4)]
        self.assertEqual(public._qwen_filter_sparse_items(items), items)

    def test_dense_dialogue_items_kept(self):
        items = chars("格安の大手に帰ろうって息子さんが言ってるみたいでさ", start=0.0, step=0.18)
        self.assertEqual(len(public._qwen_filter_sparse_items(items)), len(items))

    def test_multi_char_stretched_item_dropped(self):
        # 2 字摊到 6s(3s/字)也是对齐器硬塞,丢弃
        self.assertEqual(public._qwen_filter_sparse_items([Item("さん", 29.8, 35.8)]), [])

    def test_sparse_segment_floor_drops_remaining_thin_line(self):
        # item 级漏网后,整行仍稀疏(>3s 且 <0.5字/s)的兜底
        segs = [{"start": 0.0, "end": 10.0, "text": "あい"}]
        self.assertEqual(public._qwen_filter_sparse_segments(segs), [])

    def test_short_sparse_segment_kept(self):
        # 短行不做密度判罚,避免误杀真实短句
        segs = [{"start": 0.0, "end": 2.0, "text": "うん"}]
        self.assertEqual(public._qwen_filter_sparse_segments(segs), segs)

    def test_pipeline_filters_stretched_but_keeps_dialogue(self):
        # 同一条音频上:真实台词 + 幻觉粒子,端到端只留台词
        items = chars("気持ちいい", start=0.0, step=0.2)
        items += [Item("を", 20.0, 71.0)]  # 51s 单粒子
        items += chars("もっとして", start=80.0, step=0.2)
        segs = public._qwen_items_to_segments(items, 24, 1.0, 7.0)
        segs = public._qwen_filter_sparse_segments(segs)
        texts = [s["text"] for s in segs]
        self.assertIn("気持ちいい", texts)
        self.assertIn("もっとして", texts)
        self.assertNotIn("を", texts)


class QwenVadBlockTests(unittest.TestCase):
    # 粗粒度跳过:60s 块内 VAD 语音占比 >2% 才送模型,纯喘息/BGM 整块跳过(实测可跳 80%)
    def test_no_windows_no_blocks(self):
        self.assertEqual(public._qwen_vad_blocks(600.0, [], 60.0, 0.02), [])

    def test_window_keeps_its_block(self):
        windows = [(7.4, 8.2), (10.4, 12.5)]
        self.assertEqual(public._qwen_vad_blocks(600.0, windows, 60.0, 0.02), [(0.0, 60.0)])

    def test_window_in_second_block(self):
        windows = [(140.0, 143.0)]
        self.assertEqual(public._qwen_vad_blocks(600.0, windows, 60.0, 0.02), [(120.0, 180.0)])

    def test_tiny_window_below_ratio_skipped(self):
        windows = [(10.0, 10.5)]  # 0.5s/60s = 0.8% < 2%
        self.assertEqual(public._qwen_vad_blocks(600.0, windows, 60.0, 0.02), [])

    def test_last_block_clipped_to_total(self):
        windows = [(8590.0, 8593.0)]
        self.assertEqual(public._qwen_vad_blocks(8600.0, windows, 60.0, 0.02), [(8580.0, 8600.0)])


if __name__ == "__main__":
    unittest.main()
