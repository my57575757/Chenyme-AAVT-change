import sys
import unittest
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import public


Seg = namedtuple("Seg", ["text", "start", "end", "no_speech_prob"])


def seg(text, start, end, no_speech_prob=0.0):
    return Seg(text, start, end, no_speech_prob)


class FasterZombieSegmentTests(unittest.TestCase):
    # YMDS-293 实测:VAD 开启后残句时间轴冻结成数百~上千秒,
    # 文本只有几个字(65s/8字 ~ 1700s/12字),正常台词 0.22 字/s 以上
    def test_drops_65s_8chars_zombie(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        result = public._filter_hallucinated_segments(
            [seg("嘘じゃないよ", 100.0, 165.0)], settings
        )
        self.assertEqual(result, [])

    def test_drops_1700s_frozen_tail(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        result = public._filter_hallucinated_segments(
            [seg("飲みなさい、取りなさい。", 2400.0, 4100.8)], settings
        )
        self.assertEqual(result, [])

    def test_keeps_long_but_dense_segment(self):
        # 65s 但有 30 字(0.46 字/s):慢速长台词,不能误杀
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        s = seg("あ" * 30, 100.0, 165.0)
        self.assertEqual(public._filter_hallucinated_segments([s], settings), [s])

    def test_duration_boundary_exactly_60s_kept(self):
        # 阈值是「时长严格大于 60s」,恰好 60s 不进僵尸判罚
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        s = seg("短い", 0.0, 60.0)
        self.assertEqual(public._filter_hallucinated_segments([s], settings), [s])

    def test_density_boundary_kept_at_threshold(self):
        # 61s / 13 字 ≈ 0.213 字/s,高于 0.2 阈值,保留
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        s = seg("あ" * 13, 0.0, 61.0)
        self.assertEqual(public._filter_hallucinated_segments([s], settings), [s])

    def test_keeps_normal_short_segment(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        s = seg("こんにちは", 0.0, 2.0)
        self.assertEqual(public._filter_hallucinated_segments([s], settings), [s])

    def test_zombie_guard_disabled_when_min_seconds_zero(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        settings["zombie_min_seconds"] = 0
        s = seg("あ", 0.0, 1000.0)
        self.assertEqual(public._filter_hallucinated_segments([s], settings), [s])


class FasterThanksPhraseTests(unittest.TestCase):
    # 裸「ありがとうございました」是剧情真实道谢(YMDS-293 624.8s 采访台词误杀),
    # 只封杀与片尾词组合的 YouTube 套话
    def test_bare_thanks_not_phantom_by_default(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        s = seg("ありがとうございました", 620.0, 622.0)
        self.assertEqual(public._filter_hallucinated_segments([s], settings), [s])

    def test_bare_thanks_removed_from_default_phrases(self):
        self.assertNotIn("ありがとうございました", public.FASTER_ADVANCED_DEFAULTS["hallucination_phrases"])

    def test_outro_phrase_still_filtered(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        result = public._filter_hallucinated_segments(
            [seg("ご視聴ありがとうございました", 8000.0, 8002.0)], settings
        )
        self.assertEqual(result, [])

    def test_instory_thanks_with_prefix_kept(self):
        # 「お話ありがとうございます」整行精确匹配黑名单本就不命中,固化此行为
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        s = seg("お話ありがとうございます", 624.0, 625.5)
        self.assertEqual(public._filter_hallucinated_segments([s], settings), [s])


class FasterExistingFilterTests(unittest.TestCase):
    def test_no_speech_prob_still_dropped(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        result = public._filter_hallucinated_segments(
            [seg("普通の台詞", 0.0, 2.0, no_speech_prob=0.95)], settings
        )
        self.assertEqual(result, [])

    def test_adjacent_duplicate_still_dropped(self):
        settings = dict(public.FASTER_ADVANCED_DEFAULTS)
        s1 = seg("同じ文です", 0.0, 2.0)
        s2 = seg("同じ文です", 2.0, 4.0)
        self.assertEqual(public._filter_hallucinated_segments([s1, s2], settings), [s1])

    def test_defaults_contain_zombie_keys(self):
        self.assertEqual(public.FASTER_ADVANCED_DEFAULTS["zombie_min_seconds"], 60.0)
        self.assertEqual(public.FASTER_ADVANCED_DEFAULTS["zombie_max_chars_per_second"], 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
