"""Crafter metric regressions: scripted/offline observations, never model calls."""
from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import unittest

from blade.environment import BALROG, Game, _Base, _Crafter, make_cases


class CrafterMetricRegression(unittest.TestCase):
    @staticmethod
    def game(achievements, rewards):
        """Use the actual adapter's step/metrics with a deterministic receipt.

        Fabricating native statistics here tests an otherwise impractical all22
        boundary. It is a unit test, not a successful Crafter model trajectory.
        """
        game = _Crafter.__new__(_Crafter)
        _Base.__init__(game, "crafter", {"horizon": 2000})
        game.menu = ["Noop"]
        game.mission = game.native_instruction = "Public task"
        game.descriptions = {"Noop": "do nothing"}
        game.text = {"long_term_context": "Public surroundings", "short_term_context": "Public inventory"}
        game.native = SimpleNamespace(_step=1)
        game.reset_native_actions = 1

        class Wrapper:
            def __init__(self):
                self.rewards = iter(rewards)
                self.achievements = deepcopy(achievements)

            def step(self, action):
                game.native._step += 1
                return {"text": deepcopy(game.text)}, next(self.rewards), False, {"dead": False}

            def get_stats(self):
                count = sum(value > 0 for value in self.achievements.values())
                return {"score": float(count), "progression": count / 22,
                        "achievements": deepcopy(self.achievements)}

        game.wrapper = Wrapper()
        return game

    def test_official_achievement_score_is_separate_from_native_reward(self):
        game = self.game({"collect_wood": 3, "collect_drink": 0}, [.9, -.3])
        first = game.step("Noop")
        self.assertEqual(first["reward"], .9)
        self.assertEqual(game.metrics()["score"], 1)
        self.assertAlmostEqual(game.metrics()["native_reward"], .9)
        game.step("Noop")
        metrics = game.metrics()
        self.assertEqual(metrics["score"], 1)
        self.assertAlmostEqual(metrics["native_reward"], .6)
        self.assertEqual(metrics["achievement_count"], 1)
        self.assertEqual(metrics["progression"], 1/22)
        self.assertFalse(metrics["won"])

    def test_all22_step_result_and_metric_won_agree_without_forcing_native_done(self):
        achievements = {"achievement" + str(index): 1 for index in range(22)}
        game = self.game(achievements, [0])
        result = game.step("Noop")
        metrics = game.metrics()
        self.assertTrue(result["won"])
        self.assertTrue(metrics["won"])
        self.assertTrue(metrics["all_achievements_unlocked"])
        self.assertEqual(metrics["score"], 22)
        self.assertEqual(metrics["progression"], 1)
        self.assertEqual(metrics["native_reward"], 0)
        self.assertFalse(result["done"])
        self.assertFalse(metrics["done"])

    def test_21_achievements_are_progress_not_a_full_completion(self):
        achievements = {"achievement" + str(index): int(index < 21) for index in range(22)}
        game = self.game(achievements, [-.1])
        result = game.step("Noop")
        metrics = game.metrics()
        self.assertFalse(result["won"])
        self.assertFalse(metrics["won"])
        self.assertFalse(metrics["all_achievements_unlocked"])
        self.assertEqual(metrics["score"], 21)
        self.assertEqual(metrics["progression"], 21/22)
        self.assertEqual(metrics["native_reward"], -.1)

    def test_reporting_score_and_achievements_do_not_enter_public_inference_state(self):
        from blade.policy import public_view
        game = self.game({"collect_wood": 1}, [.9])
        game.step("Noop")
        public = public_view(game.observe())
        self.assertEqual(public["observation"], game.text)
        self.assertNotIn("score", public)
        self.assertNotIn("native_reward", public)
        self.assertNotIn("achievements", public)
        self.assertNotIn("progression", public)


@unittest.skipUnless(importlib.util.find_spec("crafter") is not None and
                     (BALROG / "balrog/environments/crafter/env.py").is_file(),
                     "Pinned native Crafter runtime is optional for the lightweight offline suite")
class NativeCrafterMetricRegression(unittest.TestCase):
    def test_scripted_native_counterexample_preserves_public_text_menu_and_official_score(self):
        """One real environment episode, no actor/provider/credential access."""
        import crafter.constants
        game = Game("crafter", make_cases("crafter")[0])
        try:
            native = game._game
            self.assertEqual(game.observe()["observation"], native.raw["text"])
            self.assertEqual(len(game.actions()), 17)
            native_order = [action.lower().replace("move west", "move_left")
                            .replace("move east", "move_right").replace("move north", "move_up")
                            .replace("move south", "move_down").replace(" ", "_")
                            for action in game.actions()]
            self.assertEqual(native_order, crafter.constants.actions)
            self.assertEqual(game.metrics()["native_steps"], 1)
            self.assertEqual(game.metrics()["decision_steps"], 0)
            reward_sum, found_difference = 0., False
            for action in ["Move West", "Move North", "Do", "Move East", "Move South", "Noop"] * 30:
                result = game.step(action)
                reward_sum += result["reward"]
                metrics = game.metrics()
                official = native.wrapper.get_stats()
                self.assertEqual(metrics["score"], official["score"])
                self.assertEqual(metrics["score"], metrics["achievement_count"])
                self.assertEqual(metrics["progression"], metrics["achievement_count"] / 22)
                self.assertAlmostEqual(metrics["native_reward"], reward_sum)
                self.assertEqual(result["won"], metrics["won"])
                self.assertEqual(game.observe()["observation"], native.raw["text"])
                found_difference |= metrics["score"] != metrics["native_reward"]
                if result["done"]:
                    break
            self.assertTrue(found_difference, "Regression needs a native score/reward counterexample")
        finally:
            game.close()


if __name__ == "__main__":
    unittest.main()
