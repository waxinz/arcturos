"""Config loader: file precedence + env override + tuple-key contract.

The 2026-09-30 publish round shipped _flatten() with string keys while
get() looked up tuples — file values were silently dropped and every call
fell through to built-in defaults (the lab overlay never applied).
Locked in here so the two key shapes can never diverge again.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("src")))
from arcturos import config as cfg  # noqa: E402


class FlattenShape(unittest.TestCase):
    def test_flat_file_map_uses_tuple_keys(self):
        flat = cfg._flatten({"bench": {"server_url": "x"}, "demo": {"judge_url": "y"}})
        self.assertEqual(flat[("bench", "server_url")], "x")
        self.assertEqual(flat[("demo", "judge_url")], "y")

    def test_get_reads_own_output(self):
        flat = cfg._flatten({"bench": {"server_url": "x"}})
        with mock.patch.object(cfg, "_load_all", return_value=flat):
            self.assertEqual(cfg.get("bench", "server_url"), "x")
            self.assertIsNone(cfg.get("bench", "missing"))
            self.assertEqual(cfg.get("bench", "missing", "fallback"), "fallback")

    def test_env_overrides_file(self):
        flat = cfg._flatten({"bench": {"server_url": "from-file"}})
        merged = dict(flat)
        merged[("bench", "server_url")] = "from-env"
        with mock.patch.object(cfg, "_load_all", return_value=merged):
            self.assertEqual(cfg.get("bench", "server_url"), "from-env")


class EnvValueTypes(unittest.TestCase):
    """List/dict config keys need real types from env overrides (2026-11-10).

    ARCTUROS_DEMO__LIVE_TARGETS='["http://h:8000"]' used to stay a string;
    seed_demo then iterated characters and urlopen('[') crashed."""

    def test_json_list_env_becomes_a_list(self):
        self.assertEqual(cfg._parse_env_value('["http://h:8000"]'),
                         ["http://h:8000"])

    def test_json_object_env_becomes_a_dict(self):
        self.assertEqual(cfg._parse_env_value('{"a": 1}'), {"a": 1})

    def test_plain_string_env_is_unchanged(self):
        self.assertEqual(cfg._parse_env_value("http://h:8000"),
                         "http://h:8000")

    def test_malformed_bracket_value_stays_a_string(self):
        self.assertEqual(cfg._parse_env_value("[not json"), "[not json")

    def test_get_section_env_list_type(self):
        cfg._load_all.cache_clear()
        try:
            with mock.patch.dict(os.environ,
                                 {"ARCTUROS_DEMO__LIVE_TARGETS":
                                  '["http://h:8000"]'}):
                section = cfg.get_section("demo")
        finally:
            cfg._load_all.cache_clear()
        self.assertEqual(section["live_targets"], ["http://h:8000"])


if __name__ == "__main__":
    unittest.main()