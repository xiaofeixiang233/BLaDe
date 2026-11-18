"""ALFWorld public projection/menu/failure tests; no model calls or keys."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from blade import alfworld_adapter as adapter


class FakeState(dict):
    def __init__(self, feedback, commands, won=False):
        super().__init__(admissible_commands=commands, won=won,
            facts=["hidden gold fact"], policy_commands=["hidden optimal move"],
            **{"extra.expert_plan": ["hidden expert"]})
        self.feedback = feedback


class FakeNative:
    def __init__(self):
        self.initial = FakeState("You arrive in a room.\nYour task is to: put a mug on a table.",
            ["look", "inventory", "go to table 1"])
        self.next = FakeState("You arrive at table 1.",
            ["look", "inventory", "examine table 1"])
        self.commands = []
        self.closed = False
        self.score, self.done = 0, False

    def reset(self):
        return self.initial

    def step(self, action):
        self.commands.append(action)
        return self.next, self.score, self.done

    def close(self):
        self.closed = True


class PublicAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.split = Path(self.temporary.name) / "valid_unseen"
        self.file = self.split / "pick_and_place_simple-mug/trial_0/game.tw-pddl"
        self.file.parent.mkdir(parents=True)
        self.file.write_text('{"solvable":true}')
        self.case = {"id": "alfworld/fake", "environment": "alfworld",
            "task": "pick_and_place_simple", "profile": "paper", "horizon": 50,
            "game_relative_path": self.file.relative_to(self.split).as_posix(),
            "game_sha256": hashlib.sha256(self.file.read_bytes()).hexdigest()}
        self.native = FakeNative()
        self.request_infos = None

        def infos(**kwargs):
            self.request_infos = kwargs
            return SimpleNamespace(**kwargs)

        self.tw = SimpleNamespace(core=SimpleNamespace(Wrapper=object), EnvInfos=infos,
            start=lambda *args, **kwargs: self.native)
        self.cases_patch = patch.object(adapter, "make_cases", return_value=[self.case])
        self.bindings_patch = patch.object(adapter, "_native_bindings",
            return_value=(self.tw, object, {}))
        self.cases_patch.start()
        self.bindings_patch.start()
        self.addCleanup(self.cases_patch.stop)
        self.addCleanup(self.bindings_patch.stop)

    def game(self):
        value = adapter.AlfWorldGame(self.case, self.split)
        self.addCleanup(value.close)
        return value

    def test_only_public_feedback_and_menu_no_free_actions(self):
        game = self.game()
        public = game.observe()
        self.assertEqual(set(public), {"environment", "mission", "native_instruction",
            "observation", "action_descriptions"})
        self.assertEqual(public["observation"], self.native.initial.feedback)
        self.assertEqual(public["mission"], "put a mug on a table.")
        self.assertEqual(self.request_infos, {"admissible_commands": True, "won": True})
        encoded = json.dumps(public)
        for prohibited in ("hidden gold", "hidden optimal", "hidden expert",
                "facts", "policy_commands", "extra.expert_plan", "score", "game_sha256"):
            self.assertNotIn(prohibited, encoded)
        self.assertEqual(self.native.commands, [])
        game.step("go to table 1")
        self.assertEqual(self.native.commands, ["go to table 1"])
        self.assertEqual(game.observe()["observation"], self.native.next.feedback)
        self.assertNotIn("You arrive in a room", game.observe()["observation"])

    def test_complete_dynamic_menu_stable_sort_without_filtering(self):
        game = self.game()
        self.assertEqual(game.actions(), sorted(self.native.initial["admissible_commands"]))
        game.step("look")
        self.assertEqual(game.actions(), sorted(self.native.next["admissible_commands"]))
        self.assertNotIn("go to table 1", game.actions())
        self.assertEqual(list(game.observe()["action_descriptions"]), game.actions())
        copied = game.actions()
        copied.pop()
        self.assertEqual(len(game.actions()), 3)

    def test_overflow_retains_all_commands_before_any_dispatch(self):
        self.native.initial["admissible_commands"] = [f"command-{i:03d}" for i in range(256)]
        with self.assertRaises(adapter.NativeMenuOverflow) as raised:
            self.game()
        record = raised.exception.record
        self.assertEqual(record["menu_size"], 256)
        self.assertEqual(len(record["complete_native_menu"]), 256)
        self.assertFalse(record["request_sent"])
        self.assertEqual(record["actions_removed"], 0)
        self.assertEqual(json.loads(str(raised.exception)), record)
        self.assertTrue(self.native.closed)
        self.assertEqual(self.native.commands, [])

    def test_post_action_overflow_retains_executed_action_and_new_menu(self):
        game = self.game()
        self.native.next["admissible_commands"] = [f"command-{i:03d}" for i in range(256)]
        with self.assertRaises(adapter.NativeMenuOverflow) as raised:
            game.step("look")
        self.assertEqual(self.native.commands, ["look"])
        self.assertEqual(game.metrics()["steps"], 1)
        self.assertEqual(len(raised.exception.record["complete_native_menu"]), 256)
        self.assertEqual(raised.exception.record["native_decisions_completed"], 1)
        self.assertTrue(raised.exception.record["prior_request_costs_retained"])
        self.assertEqual(raised.exception.record["public_observation"]["observation"], self.native.next.feedback)

    def test_invalid_action_not_repaired_or_sent(self):
        game = self.game()
        with self.assertRaisesRegex(ValueError, "no fallback"):
            game.step("go to nonexistent table")
        self.assertEqual(game.metrics()["steps"], 0)
        self.assertEqual(self.native.commands, [])

    def test_native_success_and_50_action_truncation_are_eval_only(self):
        game = self.game()
        for _ in range(50):
            result = game.step("look")
        self.assertTrue(result["truncated"])
        self.assertFalse(result["won"])
        self.assertFalse(result["terminated"])
        self.assertEqual(game.metrics()["progression"], 0)
        with self.assertRaises(ValueError):
            game.step("look")
        self.assertEqual(len(self.native.commands), 50)

    def test_native_done_wins_over_outer_truncation(self):
        game = self.game()
        self.native.done, self.native.score = True, 1
        self.native.next["won"] = True
        result = game.step("look")
        self.assertTrue(result["won"])
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["reward"], 1)
        self.assertNotIn("won", game.observe())

    def test_exact_case_identity_and_file_hash_are_required(self):
        bad = deepcopy(self.case)
        bad["id"] = "foreign"
        with self.assertRaisesRegex(ValueError, "manifest"):
            adapter.AlfWorldGame(bad, self.split)
        self.file.write_text('{"solvable":false}')
        with self.assertRaisesRegex(ValueError, "hash changed"):
            adapter.AlfWorldGame(self.case, self.split)

    def test_generic_rules_do_not_contain_task_decomposition(self):
        instruction = adapter.native_instruction()
        for task in adapter.TASK_COUNTS:
            self.assertNotIn(task, instruction)
        self.assertNotIn("then place", instruction)
        self.assertNotIn("suitable", instruction)
        self.assertNotIn("optimal", instruction)


class InventoryTests(unittest.TestCase):
    def test_partial_data_rejected_even_for_smoke_and_no_profile_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "valid_unseen"
            game = split / "pick_and_place_simple-mug/trial_0/game.tw-pddl"
            game.parent.mkdir(parents=True)
            game.write_text('{"solvable":true}')
            for profile in ("paper", "smoke"):
                with self.assertRaisesRegex(ValueError, "complete134"):
                    adapter.make_cases(profile, split)
            for bad_profile, horizon in (("default", 50), ("paper", 100), ("paper", True)):
                with self.assertRaises(ValueError):
                    adapter.make_cases(bad_profile, split, horizon)

    @unittest.skipUnless(os.environ.get("BLADE_ALFWORLD_TEST_DATA"),
        "Set BLADE_ALFWORLD_TEST_DATA for native offline checks")
    def test_real_all134_hashes_groups_and_one_native_reset_step(self):
        root = os.environ["BLADE_ALFWORLD_TEST_DATA"]
        rows = adapter.make_cases("paper", root)
        self.assertEqual(len(rows), 134)
        self.assertEqual(len({v["group"] for v in rows}), 47)
        self.assertEqual(sum(v["known_previously_used_12_case"] for v in rows), 12)
        self.assertTrue(all(not Path(v["game_relative_path"]).is_absolute() for v in rows))
        self.assertEqual(rows, adapter.make_cases("paper", root))
        case = adapter.make_cases("smoke", root)[0]
        game = adapter.AlfWorldGame(case, root)
        try:
            public = game.observe()
            self.assertGreater(len(game.actions()), 2)
            game.step("look")
            self.assertEqual(game.metrics()["steps"], 1)
            self.assertEqual(game.observe()["mission"], public["mission"])
            self.assertEqual(game.identity()["available_game_count"], 134)
        finally:
            game.close()


if __name__ == "__main__":
    unittest.main()
