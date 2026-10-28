import unittest

from alebrije import llm
from alebrije.errors import LlmError

from . import sample_config


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(llm.extract_json('{"a": 1}'), {"a": 1})

    def test_json_in_prose(self):
        self.assertEqual(llm.extract_json('Sure! {"a": 1} hope that helps'),
                         {"a": 1})

    def test_json_in_fence(self):
        self.assertEqual(llm.extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_nested(self):
        self.assertEqual(llm.extract_json('{"a": {"b": [1,2]}}'), {"a": {"b": [1, 2]}})

    def test_no_json_raises(self):
        with self.assertRaises(LlmError) as ctx:
            llm.extract_json("no json here at all")
        self.assertEqual(ctx.exception.kind, "parse")

    def test_unterminated_raises(self):
        with self.assertRaises(LlmError):
            llm.extract_json('{"a": 1')

    def test_malformed_raises(self):
        with self.assertRaises(LlmError):
            llm.extract_json('{"a": 1,,}')


class TestPostJsonErrors(unittest.TestCase):
    def test_connect_refused(self):
        cfg = sample_config(vlm_url="http://127.0.0.1:1", llm_timeout=2)
        with self.assertRaises(LlmError) as ctx:
            llm._post_json(cfg["vlm_url"] + "/v1/chat/completions", {}, 1.0)
        self.assertEqual(ctx.exception.kind, "connect")

    def test_bad_port_dns_kind(self):
        # connection refused to a closed local port is "connect", not dns
        with self.assertRaises(LlmError) as ctx:
            llm._post_json("http://localhost:1/x", {}, 1.0)
        self.assertIn(ctx.exception.kind, ("connect", "dns"))


class TestEmbedBatching(unittest.TestCase):
    def test_embed_batches_in_chunks_of_32(self):
        calls = []

        def fake_post(url, payload, timeout):
            calls.append(len(payload["input"]))
            return {"data": [{"index": i, "embedding": [float(i)] * 4}
                             for i in range(len(payload["input"]))]}

        orig = llm._post_json
        llm._post_json = fake_post
        try:
            out = llm.embed(sample_config(), [str(i) for i in range(70)])
        finally:
            llm._post_json = orig
        self.assertEqual(calls, [32, 32, 6])
        self.assertEqual(len(out), 70)


if __name__ == "__main__":
    unittest.main()
