"""Partial trajectories must keep action counts, timing and incurred bills."""
import json
from pathlib import Path
import tempfile
import unittest

from blade import policy, runner
from blade.ledger import BudgetHalt, Ledger
from blade.support import HERE, digest
from blade.tests.test_runner import FakeGame, choice


class PartialRunMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = json.loads((HERE / 'configs/ablation.json').read_text())
        self.cfg.update(output=self.temp.name, protocol_hash='a' * 64)
        self.cfg['budgets'] = {'jev_usd': 5, 'gpt_usd': 5}
        case = {'id': 'partial/fixed', 'environment': 'babyai', 'horizon': 20, 'group': 'fixed'}
        game = FakeGame('babyai', case)
        self.row = {'case': case, 'decision_cap': 20, 'goal_slots': policy.goal_slots(20, 4),
                    'initial_sha256': digest(game.observe()), 'actions': game.actions(),
                    'action_descriptions_sha256': digest(game.observe()['action_descriptions'])}
        self.ledger = Ledger(self.temp.name, self.cfg, providers={'jev': lambda p, s: choice(p)})
        self.addCleanup(self.ledger.close)
        self.eid = 'babyai|partial/fixed|jev-history-h4'

    def summary(self):
        path, _, _ = runner.episode_paths(self.temp.name, self.eid)
        return json.loads(path.read_text())

    def test_native_failure_retains_completed_steps_and_successful_request_before_failure(self):
        class Interrupted(FakeGame):
            def step(self, action):
                if self.n == 2:
                    raise RuntimeError('native interface interrupted after a billed decision')
                return super().step(action)

        with self.assertRaises(RuntimeError):
            runner.execute_episode(self.cfg, 'babyai', self.row, 'jev-history-h4', self.ledger, Interrupted)
        summary = self.summary()
        self.assertEqual((summary['status'], summary['decisions']), ('failed', 2))
        self.assertEqual(summary['metrics']['native_steps'], 2)
        self.assertGreaterEqual(summary['episode_wall_seconds'], 0)
        self.assertGreaterEqual(summary['attempt_wall_seconds'], summary['episode_wall_seconds'])
        self.assertEqual(self.ledger.statistics()['physical']['jev']['requests'], 3)
        attempt = runner.read_records(Path(self.temp.name) / 'episode-attempts' / (digest(self.eid) + '.jsonl'))
        self.assertEqual((attempt[0]['status'], attempt[0]['retained_completed_decisions']), ('failed', 2))
        self.assertTrue((Path(self.temp.name) / 'STOP').exists())

    def test_budget_pause_keeps_progress_and_does_not_invent_third_action(self):
        original = self.ledger.call

        def paused(kind, request, eid, step, role):
            if step == 2:
                raise BudgetHalt('reservation denied before dispatch')
            return original(kind, request, eid, step, role)

        self.ledger.call = paused
        with self.assertRaises(BudgetHalt):
            runner.execute_episode(self.cfg, 'babyai', self.row, 'jev-history-h4', self.ledger, FakeGame)
        summary = self.summary()
        self.assertEqual((summary['status'], summary['decisions']), ('budget_paused', 2))
        self.assertEqual(self.ledger.statistics()['physical']['jev']['requests'], 2)
        self.assertFalse((Path(self.temp.name) / 'STOP').exists())
        _, path, _ = runner.episode_paths(self.temp.name, self.eid)
        self.assertEqual(len(runner.read_records(path)), 2)
