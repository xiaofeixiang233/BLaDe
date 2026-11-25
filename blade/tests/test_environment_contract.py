"""Offline identity and native dispatch checks; no model clients or credentials."""
from copy import deepcopy
import unittest

from blade.environment import BENCHMARKS, make_cases, _NLEFamily
from blade.policy import MAIN_METHODS, effective_decision_cap, goal_slots


class CaseContract(unittest.TestCase):
    def test_v3_native_case_counts_and_caps_without_local_alfworld_data(self):
        expected = {"babyai": 125, "minihack": 40, "babaisai": 200}
        self.assertEqual(BENCHMARKS, ("babyai", "minihack", "babaisai", "alfworld"))
        self.assertEqual({n: len(make_cases(n)) for n in expected}, expected)
        all_cases = [row for n in expected for row in make_cases(n)]
        self.assertEqual(len({r["id"] for r in all_cases}), 365)
        self.assertEqual(len(all_cases) * len(MAIN_METHODS), 3285)
        # ALFWorld file enumeration is tested independently by its adapter; this
        # generic offline contract does not require downloading its 134 games.
        from blade.alfworld_adapter import TASK_COUNTS
        self.assertEqual(sum(TASK_COUNTS.values()), 134)
        self.assertEqual((len(all_cases) + sum(TASK_COUNTS.values())) * len(MAIN_METHODS), 4491)
        for row in all_cases:
            if row["environment"] == "babyai":
                self.assertEqual(row["horizon"], 64 if row["task"] in ("goto", "pickup") else 128)
            else:
                self.assertEqual(row["horizon"], {"minihack": 100, "babaisai": 100}[row["environment"]])

    def test_deterministic_and_explicit_small_profile(self):
        for name in BENCHMARKS:
            if name == "alfworld": continue  # Covered by test_alfworld_adapter.
            self.assertEqual(make_cases(name), make_cases(name))
            self.assertEqual(len(make_cases(name, profile="smoke")), 1)
            self.assertEqual(make_cases(name, limit=2), make_cases(name)[:2])
        for retired in ("nethack", "nle"):
            with self.assertRaises(ValueError): make_cases(retired)
        self.assertEqual(make_cases("baba"), make_cases("babaisai"))
        with self.assertRaises(ValueError):
            make_cases("babyai", limit=0)

    def test_main_arms_share_decision_caps_without_changing_native_cases(self):
        cfg={'decision_caps':{'crafter':2000}}
        for name in BENCHMARKS:
            if name == "alfworld": continue  # Its dynamic-menu cases have a separate contract.
            for case in make_cases(name):
                frozen=deepcopy(case)
                expected=case['horizon']
                for method in MAIN_METHODS:
                    with self.subTest(environment=name,method=method):
                        self.assertEqual(effective_decision_cap(cfg,case),expected)
                self.assertEqual(case,frozen)


class PrimitiveNativeDispatch(unittest.TestCase):
    """Preserved historical NLE adapter regression checks; NLE is not in v3 scope."""
    @staticmethod
    def game(terminate_first=False):
        game = _NLEFamily.__new__(_NLEFamily)
        game.name = "nle"
        game.case = {"horizon": 100000}
        game.closed = game.done = game.terminated = game.truncated = game.won = False
        game.steps = 0
        game.total_reward = game.score = 0.
        game.menu = ["00", "attack", "north"]
        game.mission = game.native_instruction = "Native instructions"
        game.descriptions = {a: a for a in game.menu}
        game.info = {}
        game.text = {"long_term_context": "Waiting for a direction", "short_term_context": ""}

        class Wrapper:
            def __init__(self): self.calls = []
            def step(self, action):
                self.calls.append(action)
                return {"text": deepcopy(game.text)}, 0., bool(terminate_first), False, {}
            def get_stats(self): return {"score": 0}

        game.wrapper = Wrapper()
        return game

    def test_selected_two_digit_label_emits_only_its_literal_keys(self):
        game = self.game()
        result = game.step("00")
        self.assertEqual(game.wrapper.calls, ["0", "0"])
        self.assertEqual([r["action"] for r in result["native_subactions"]], ["0", "0"])
        self.assertEqual(game.steps, 1)

    def test_native_termination_stops_remaining_selected_keys(self):
        game = self.game(terminate_first=True)
        result = game.step("00")
        self.assertTrue(result["done"])
        self.assertEqual(game.wrapper.calls, ["0"])

    def test_public_message_does_not_filter_or_replace_model_action(self):
        game = self.game()
        menu = game.actions()
        # Public message asks for a direction, but the selected attack is
        # dispatched unchanged. This catches accidental interaction repair.
        game.step("attack")
        self.assertEqual(game.wrapper.calls, ["attack"])
        self.assertEqual(game.actions(), menu)
        with self.assertRaises(ValueError):
            game.step("unlisted action")
        self.assertEqual(game.wrapper.calls, ["attack"])


if __name__ == "__main__":
    unittest.main()
