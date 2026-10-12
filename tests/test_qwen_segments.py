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


if __name__ == "__main__":
    unittest.main()
