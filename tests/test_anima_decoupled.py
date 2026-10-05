"""Tests for the decoupled conditioning helpers and the node contract."""

import importlib.util
import sys
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from anima_decoupled import (  # noqa: E402
    build_conditioning,
    common_prefix_len,
    join_prefix,
    qwen_ids,
    t5_ids_and_weights,
)


class FakeClip:
    """Minimal CLIP stand-in: whitespace tokenization, id = word hash, weight 1.0."""

    def __init__(self):
        self.calls = []

    def tokenize(self, text):
        self.calls.append(text)
        words = text.split()
        return {
            "qwen3_06b": [[(hash(w) % 10_000, 1.0) for w in words]],
            "t5xxl": [[(hash(w) % 10_000, 1.0) for w in words]],
        }

    def encode_from_tokens_scheduled(self, tokens):
        import torch

        rows = len(tokens["qwen3_06b"][0])
        cond = torch.arange(rows, dtype=torch.float32).reshape(1, rows, 1).repeat(1, 1, 4)
        return [[cond, {"pooled_output": None}]]


class NotAnimaClip:
    """CLIP stand-in for a non-Anima model: no qwen3_06b / t5xxl entries."""

    def tokenize(self, text):
        return {"l": [[(1, 1.0)]]}


class WholeStringClip(FakeClip):
    """Tokenizes each text as one opaque token, so prefix and joined text share no tokens."""

    def tokenize(self, text):
        self.calls.append(text)
        token = hash(text) % 10_000
        return {"qwen3_06b": [[(token, 1.0)]], "t5xxl": [[(token, 1.0)]]}


def load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "anima_decoupled_plugin", PLUGIN_ROOT / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["anima_decoupled_plugin"] = module
    spec.loader.exec_module(module)
    return module


class HelperTests(unittest.TestCase):
    def test_common_prefix_len(self):
        self.assertEqual(common_prefix_len([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(common_prefix_len([1, 2], [1, 2, 3]), 2)
        self.assertEqual(common_prefix_len([], [1]), 0)
        self.assertEqual(common_prefix_len([5, 1], [1, 5]), 0)

    def test_join_prefix(self):
        self.assertEqual(join_prefix("", "body"), "body")
        self.assertEqual(join_prefix("   ", "body"), "body")
        self.assertEqual(join_prefix("  pre  ", "  body"), "pre body")

    def test_token_extractors(self):
        clip = FakeClip()
        tokens = clip.tokenize("a b c")
        self.assertEqual(len(qwen_ids(tokens)), 3)
        ids, weights = t5_ids_and_weights(tokens)
        self.assertEqual(tuple(ids.shape), (3,))
        self.assertEqual(tuple(weights.shape), (3,))

    def test_non_anima_clip_reports_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Anima"):
            build_conditioning(NotAnimaClip(), "a b", "c d")


class BuildConditioningTests(unittest.TestCase):
    def test_target_side_decides_rows_and_ids(self):
        clip = FakeClip()
        cond, dropped, rows = build_conditioning(clip, "alpha beta gamma", "one two")
        self.assertEqual(dropped, 0)
        self.assertEqual(rows, 2)
        tensor, extra = cond[0]
        self.assertEqual(tensor.shape[1], 3)  # source rows are not driven by target
        self.assertEqual(tuple(extra["t5xxl_ids"].shape), (2,))
        self.assertEqual(tuple(extra["t5xxl_weights"].shape), (2,))

    def test_prefix_rows_are_dropped_by_default(self):
        clip = FakeClip()
        cond, dropped, _ = build_conditioning(clip, "body tail", "tags here", "p1 p2")
        self.assertEqual(dropped, 2)
        self.assertEqual(cond[0][0].shape[1], 2)  # 4 joined rows minus 2 prefix rows

    def test_repeated_prefix_keeps_only_the_last_copy(self):
        """PP【P】: encode "P P P", keep only the third copy as conditioning rows."""
        clip = FakeClip()
        cond, dropped, _ = build_conditioning(
            clip, "alpha beta", "tags", "alpha beta alpha beta"
        )
        self.assertEqual(clip.calls[0], "alpha beta alpha beta alpha beta")
        self.assertEqual(dropped, 4)
        self.assertEqual(cond[0][0].shape[1], 2)

    def test_strip_prefix_off_keeps_prefix_rows(self):
        clip = FakeClip()
        cond, dropped, _ = build_conditioning(clip, "body tail", "tags", "p1 p2", False)
        self.assertEqual(dropped, 0)
        self.assertEqual(cond[0][0].shape[1], 4)

    def test_empty_prefix_is_a_noop(self):
        clip = FakeClip()
        cond, dropped, _ = build_conditioning(clip, "body tail", "tags", "   ")
        self.assertEqual(dropped, 0)
        self.assertEqual(cond[0][0].shape[1], 2)

    def test_unmatched_prefix_is_rejected_when_stripping(self):
        with self.assertRaisesRegex(ValueError, "no removable token prefix"):
            build_conditioning(WholeStringClip(), "alpha beta", "tags", "zzz")


class NodeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = load_plugin_module()

    def test_registered_mapping(self):
        self.assertEqual(
            set(self.plugin.NODE_CLASS_MAPPINGS), {"AnimaDecoupledConditioning"}
        )
        self.assertEqual(
            self.plugin.NODE_DISPLAY_NAME_MAPPINGS["AnimaDecoupledConditioning"],
            "Anima Decoupled Conditioning",
        )

    def test_input_and_output_contract(self):
        node_cls = self.plugin.NODE_CLASS_MAPPINGS["AnimaDecoupledConditioning"]
        self.assertEqual(node_cls.RETURN_TYPES, ("CONDITIONING",))
        inputs = node_cls.INPUT_TYPES()
        self.assertEqual(set(inputs["required"]), {"clip", "qwen_input", "t5_input"})
        self.assertEqual(set(inputs["optional"]), {"qwen_prefix", "strip_prefix"})
        self.assertEqual(inputs["optional"]["strip_prefix"][1]["default"], True)

    def test_encode_returns_one_conditioning(self):
        node_cls = self.plugin.NODE_CLASS_MAPPINGS["AnimaDecoupledConditioning"]
        result = node_cls().encode(FakeClip(), "body", "tags", "p")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0][0].shape[1], 1)  # 2 joined rows minus 1 prefix row


if __name__ == "__main__":
    unittest.main()
