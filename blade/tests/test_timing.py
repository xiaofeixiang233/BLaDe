"""Observed action timing counterexamples, with no environment or API calls."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from blade.support import digest
from blade.timing import summarize_timing, write_timing_report


class TimingReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)

    def save(self, eid, records, goals=None):
        stem = digest(eid)
        path = self.out / "episodes" / (stem + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        if goals is not None:
            path = self.out / "goals" / (stem + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in goals))

    def episode(self, eid, status="complete", decisions=2):
        return {"id": eid, "environment": "babyai", "method": "d3-w2", "status": status,
                "decisions": decisions, "case": {"task_type": "goto"}}

    def test_action_percentiles_and_goal_update_are_not_request_percentiles(self):
        self.save("one", [
            {"step": 0, "decision_wall_seconds": 2, "native_step_seconds": .5, "goal": None},
            {"step": 1, "decision_wall_seconds": 10, "native_step_seconds": 1, "goal": "desired"},
            {"step": 2, "decision_wall_seconds": 4, "native_step_seconds": 1, "goal": "desired"}],
            goals=[{"step": 1, "request_id": "r"}])
        result = write_timing_report(self.out, [self.episode("one", decisions=3)])
        all_actions = result["total"]["by_action_category"]["all"]
        self.assertEqual(all_actions["decision_wall_seconds"]["p50"], 4)
        self.assertAlmostEqual(all_actions["decision_wall_seconds"]["p95"], 9.4)
        self.assertEqual(all_actions["policy_wall_seconds"]["p50"], 3)
        self.assertAlmostEqual(all_actions["policy_wall_seconds"]["known_sum"], 13.5)
        self.assertEqual(result["total"]["by_action_category"]["goal_update"]["actions"], 1)
        self.assertEqual(result["total"]["by_action_category"]["goal_reuse"]["actions"], 1)
        self.assertEqual(result["by_task_method"]["babyai|goto|d3-w2"]["retained_actions"], 3)
        self.assertEqual(json.loads((self.out / "timing.json").read_text()), result)
        with (self.out / "timing.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        selected = [row for row in rows if row["scope"] == "total" and row["category"] == "all"
                    and row["metric"] == "decision_wall_seconds"]
        self.assertAlmostEqual(float(selected[0]["p95"]), 9.4)
        self.assertTrue((self.out / "timing-by-episode.csv").exists())

    def test_interrupted_episode_keeps_actions_and_does_not_impute_failed_latency(self):
        self.save("failed", [
            {"step": 0, "decision_wall_seconds": 2, "native_step_seconds": .5, "goal": None}],
            goals=[{"step": 1, "request_id": "goal-before-failure"}])
        result = summarize_timing(self.out, [self.episode("failed", status="failed", decisions=1),
                                             self.episode("pending", status="pending", decisions=None)])
        total = result["total"]
        self.assertEqual(total["episodes"], 2)
        self.assertFalse(total["complete_cohort"])
        self.assertEqual(total["retained_actions"], 1)
        self.assertEqual(total["interrupted_episodes"], 1)
        self.assertIsNone(total["interrupted_decision_latency_seconds"])
        self.assertEqual(total["goal_updates_without_retained_action"], 1)
        self.assertEqual(total["trajectory_files_missing"], 1)
        self.assertEqual(total["by_action_category"]["all"]["decision_wall_seconds"]["known_sum"], 2)

    def test_missing_and_negative_policy_duration_remain_unknown(self):
        self.save("one", [
            {"step": 0, "decision_wall_seconds": 1, "native_step_seconds": 2, "goal": "desired"},
            {"step": 1, "decision_wall_seconds": None, "native_step_seconds": .5, "goal": None}])
        total = summarize_timing(self.out, [self.episode("one")])["total"]
        self.assertEqual(total["negative_policy_duration_actions"], 1)
        metric = total["by_action_category"]["all"]["policy_wall_seconds"]
        self.assertEqual(metric["known_actions"], 0)
        self.assertEqual(metric["unknown_actions"], 2)
        self.assertIsNone(metric["p95"])
        self.assertEqual(total["by_action_category"]["goal_update_unknown"]["actions"], 1)

    def test_corrupt_or_duplicate_steps_cannot_silently_disappear(self):
        self.save("one", [{"step": 0}, {"step": 0}])
        with self.assertRaisesRegex(ValueError, "sequence"):
            summarize_timing(self.out, [self.episode("one")])
        path = self.out / "episodes" / (digest("one") + ".jsonl")
        path.write_text('{"step":0}\n{"step":')
        with self.assertRaisesRegex(ValueError, "JSONL"):
            summarize_timing(self.out, [self.episode("one")])


if __name__ == "__main__":
    unittest.main()
