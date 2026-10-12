import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import public


class ChunkTests(unittest.TestCase):
    def test_splits_by_line_limit(self):
        texts = [f"t{i}" for i in range(200)]
        chunks = list(public._chunk_translation_items(texts, max_lines=150, max_chars=10 ** 9))
        self.assertEqual([len(c) for c in chunks], [150, 50])
        # 全局编号连续:第二块首个条目仍是原始下标
        self.assertEqual(chunks[1][0][0], 150)

    def test_splits_by_char_limit(self):
        texts = ["a" * 40, "b" * 40, "c" * 40]
        chunks = list(public._chunk_translation_items(texts, max_lines=100, max_chars=50))
        self.assertEqual(len(chunks), 3)

    def test_single_oversized_line_is_its_own_chunk(self):
        texts = ["x" * 500]
        chunks = list(public._chunk_translation_items(texts, max_lines=100, max_chars=100))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], (0, "x" * 500))


class NumberedBlockTests(unittest.TestCase):
    def test_numbering_is_one_based(self):
        block = public._build_numbered_block([(0, "hello"), (1, "world")])
        self.assertEqual(block, "1|hello\n2|world")

    def test_multiline_text_flattened(self):
        block = public._build_numbered_block([(0, "line1\nline2")])
        self.assertEqual(block, "1|line1 line2")


class ParseTests(unittest.TestCase):
    def test_normal_numbered(self):
        out = public._parse_numbered_translations("1|你好\n2|世界", [1, 2])
        self.assertEqual(out, {1: "你好", 2: "世界"})

    def test_tolerates_prose_and_spacing(self):
        content = "好的,翻译如下:\n 1 | 你好 \n2｜世界\n以上"
        out = public._parse_numbered_translations(content, [1, 2])
        self.assertEqual(out[1], "你好")
        self.assertEqual(out[2], "世界")

    def test_missing_number_raises(self):
        with self.assertRaises(ValueError):
            public._parse_numbered_translations("1|你好", [1, 2])

    def test_positional_fallback_when_count_matches(self):
        # 模型未加编号但行数一致,按顺序兜底
        out = public._parse_numbered_translations("你好\n世界", [1, 2])
        self.assertEqual(out, {1: "你好", 2: "世界"})

    def test_duplicate_number_first_wins(self):
        out = public._parse_numbered_translations("1|甲\n1|乙", [1])
        self.assertEqual(out[1], "甲")


class BatchTranslateTests(unittest.TestCase):
    def test_aligned_across_chunks_with_subtitles_placeholder(self):
        texts = [f"日文{i}" for i in range(200)]
        seen_prompts = []

        def fake_send(system_prompt, user_content):
            seen_prompts.append(user_content)
            lines = []
            for line in user_content.splitlines():
                if "|" in line and line.split("|", 1)[0].strip().isdigit():
                    num = line.split("|", 1)[0].strip()
                    lines.append(f"{num}|译{num}")
            return "\n".join(lines)

        result = public._batch_translate("SYS", "译:\n{subtitles}", texts, fake_send,
                                         wait_time=0, max_lines=150, max_chars=10 ** 9)
        self.assertEqual(len(result), 200)
        self.assertEqual(result[0], "译1")
        self.assertEqual(result[199], "译200")
        self.assertEqual(len(seen_prompts), 2)
        self.assertNotIn("{subtitles}", seen_prompts[0])

    def test_appends_block_when_placeholder_absent(self):
        def fake_send(system_prompt, user_content):
            self.assertIn("1|abc", user_content)
            return "1|译文"

        public._batch_translate("SYS", "请翻译", ["abc"], fake_send, wait_time=0)

    def test_no_legacy_placeholder_leak(self):
        captured = {}

        def fake_send(system_prompt, user_content):
            captured["u"] = user_content
            return "1|译文"

        public._batch_translate("SYS", "旧模板 {combined_text} 文本:{subtitles}", ["abc"], fake_send)
        self.assertNotIn("{combined_text}", captured["u"])
        self.assertNotIn("{subtitles}", captured["u"])


class ComposeTests(unittest.TestCase):
    def test_layouts(self):
        self.assertEqual(public._compose_bilingual("原文", "译文", "原始语言为首"), "原文\n译文")
        self.assertEqual(public._compose_bilingual("原文", "译文", "目标语言为首"), "译文\n原文")
        self.assertEqual(public._compose_bilingual("原文", "译文", "仅译文"), "译文")


class FindVideosTests(unittest.TestCase):
    def test_excludes_cloud_cache_dir_at_any_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ATID-1").mkdir()
            (root / "ATID-1" / "a.mp4").write_text("", encoding="utf-8")
            (root / "normal" / "deep").mkdir(parents=True)
            (root / "normal" / "deep" / "b.mkv").write_text("", encoding="utf-8")
            cache = root / "云盘缓存文件" / "ODA3" / "我的云盘缓存"
            cache.mkdir(parents=True)
            (cache / "cached.mp4").write_text("", encoding="utf-8")

            found = public.findVideos(str(root))
            names = sorted(Path(f).name for f in found)
            self.assertEqual(names, ["a.mp4", "b.mkv"])
            self.assertFalse(any("云盘缓存文件" in f for f in found))


class RefusalTests(unittest.TestCase):
    def test_detects_refusal(self):
        self.assertTrue(public._looks_like_refusal("抱歉，我不能协助翻译涉及露骨性暴力的内容。"))
        self.assertTrue(public._looks_like_refusal(""))
        self.assertFalse(public._looks_like_refusal("1|你好\n2|世界"))
        self.assertFalse(public._looks_like_refusal("你好"))

    def test_batch_refusal_falls_back_per_line(self):
        texts = ["你好", "露骨内容A", "再见"]

        def fake_send(system_prompt, user_content):
            # 整批请求:拒答
            if user_content.count("|") >= 2:
                return "抱歉，我不能协助翻译此类内容。"
            # 逐条:含露骨关键字的拒答,其余正常
            num = int(user_content.split("|", 1)[0].strip().splitlines()[-1])
            if "露骨" in user_content:
                return "抱歉，我不能协助。"
            return f"{num}|译{num}"

        out = public._batch_translate("SYS", "{subtitles}", texts, fake_send, wait_time=0)
        self.assertEqual(out[0], "译1")
        self.assertEqual(out[1], "露骨内容A")  # 被拒条目保留原文
        self.assertEqual(out[2], "译3")


if __name__ == "__main__":
    unittest.main()
