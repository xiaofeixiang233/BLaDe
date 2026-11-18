"""Offline controls for the new dynamic interface and single-call ablations."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from blade import policy, runner
from blade.alfworld_adapter import NativeMenuOverflow
from blade.ledger import Halt
from blade.providers import ProviderError, Providers
from blade.support import HERE, digest, load_config
from blade.tests.test_kev_client import card_response, answer, payload
from blade.tests.test_runner import FakeGame, target


class RecordingLedger:
    def __init__(self):
        self.calls = []

    def call(self, kind, request, eid, step, role):
        self.calls.append((kind, step, role, deepcopy(request)))
        if role == 'goal':
            return str(len(self.calls)), target(request)
        menu = list(request['questions']['action']['criteria'])
        distribution = {action: float(index == 0) for index, action in enumerate(menu)}
        return str(len(self.calls)), {'model': request['model'], 'answers': {
            'action': {'type': 'choice', 'choice': menu[0], 'probabilities': distribution}}}


class ChangingMenu(FakeGame):
    def actions(self):
        return ['look', 'go to table 1'] if self.n % 2 == 0 else ['examine table 1', 'inventory']

    def observe(self):
        return {'environment': self.name, 'mission': 'Public task',
                'observation': 'Public feedback ' + str(self.n),
                'native_instruction': 'Native command syntax',
                'action_descriptions': {action: action for action in self.actions()}}

    def step(self, action):
        if action not in self.actions():
            raise AssertionError('Action was altered or stale')
        return super().step(action)


class V3IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cfg = json.loads((HERE / 'configs/ablation.json').read_text())
        self.cfg.pop('execution_scope', None)
        self.cfg['ablation_actors'] = ['jev', 'd3']
        self.cfg['version'] = 'blade-main-v1'
        self.cfg.update(output=self.temporary.name, protocol_hash='a' * 64)

    def row(self, name, factory, horizon=20, mode='static'):
        case = {'id': name + '/fake', 'environment': name, 'horizon': horizon, 'group': 'fixed'}
        game = factory(name, case)
        public = game.observe()
        return {'case': case, 'initial_sha256': digest(public), 'actions': game.actions(),
                'action_descriptions_sha256': digest(public['action_descriptions']),
                'decision_cap': horizon, 'goal_slots': policy.goal_slots(horizon, 4),
                'action_menu_mode': mode}

    def test_alfworld_all_roles_receive_complete_current_menu_and_dispatch_exact_choice(self):
        row = self.row('alfworld', ChangingMenu, mode='dynamic_native_admissible')
        ledger = RecordingLedger()
        games = []

        def factory(name, case):
            game = ChangingMenu(name, case)
            games.append(game)
            return game

        runner.execute_episode(self.cfg, 'alfworld', row, 'jev-w2', ledger, factory)
        self.assertEqual(games[0].executed,
                         ['look', 'examine table 1'] * 10)
        for kind, step, role, request in ledger.calls:
            expected = ['look', 'go to table 1'] if step % 2 == 0 else ['examine table 1', 'inventory']
            if role == 'goal':
                self.assertEqual(list(json.loads(request['input'])['native_action_descriptions']), expected)
            else:
                self.assertEqual(list(request['questions']['action']['criteria']), expected)
        _, steps, _ = runner.episode_paths(self.cfg['output'], 'alfworld|alfworld/fake|jev-w2')
        self.assertEqual([r['action_menu'] for r in runner.read_records(steps)],
                         [['look', 'go to table 1'], ['examine table 1', 'inventory']] * 10)

    def test_alfworld_menu_requires_authorization_before_any_model_request(self):
        self.cfg['alfworld']['native_menu_authorized'] = False
        row = self.row('alfworld', ChangingMenu, mode='dynamic_native_admissible')
        ledger = RecordingLedger()
        with self.assertRaises(Halt):
            runner.execute_episode(self.cfg, 'alfworld', row, 'jev-flat-h0', ledger, ChangingMenu)
        self.assertEqual(ledger.calls, [])
        self.assertTrue((Path(self.cfg['output']) / 'STOP').is_file())

    def test_post_action_menu_overflow_retains_paid_dependencies_and_complete_failure(self):
        class Overflow(ChangingMenu):
            def step(self, action):
                super().step(action)
                raise NativeMenuOverflow(self.case['id'], [f'command-{i}' for i in range(256)],
                                         self.observe(), self.n)

        row = self.row('alfworld', Overflow, mode='dynamic_native_admissible')
        ledger = RecordingLedger()
        with self.assertRaises(NativeMenuOverflow):
            runner.execute_episode(self.cfg, 'alfworld', row, 'jev-flat-h0', ledger, Overflow)
        self.assertEqual(len(ledger.calls), 1)
        failure = json.loads(next((Path(self.cfg['output']) / 'failures').glob('*.json')).read_text())
        self.assertEqual(len(failure['complete_native_menu']), 256)
        self.assertEqual(failure['native_decisions_completed'], 1)
        self.assertTrue(failure['prior_request_costs_retained'])
        summary, _, _ = runner.episode_paths(self.cfg['output'], 'alfworld|alfworld/fake|jev-flat-h0')
        self.assertEqual(json.loads(summary.read_text())['status'], 'failed')

    def test_history_and_joint_differ_only_in_goal_input_after_warmup(self):
        row = self.row('babyai', FakeGame)
        ledgers = {}
        for suffix in ('history-h4', 'joint-h4'):
            ledger = RecordingLedger()
            runner.execute_episode(self.cfg, 'babyai', row, 'jev-' + suffix, ledger, FakeGame)
            ledgers[suffix] = ledger
        history = ledgers['history-h4']
        self.assertEqual([kind for kind, _, _, _ in history.calls], ['jev'] * 20)
        actor_joint = [request for kind, _, _, request in ledgers['joint-h4'].calls if kind == 'jev']
        for step, ((_, _, _, plain), joint) in enumerate(zip(history.calls, actor_joint)):
            joint = deepcopy(joint)
            goal = joint['state'].pop('desired_state', None)
            self.assertEqual(plain, joint)
            self.assertEqual(goal is not None, step >= 8)
            self.assertEqual(len(plain['state']['previous_interactions']), min(step, 4))
        self.assertEqual(sum(kind == 'gpt' for kind, _, _, _ in ledgers['joint-h4'].calls), 4)

    def test_native_runtime_override_preserves_venv_python_symlink(self):
        root = Path(self.temporary.name)
        executable = root / 'venv/bin/python'
        executable.parent.mkdir(parents=True)
        executable.symlink_to('/usr/bin/python3')
        config = root / 'config.json'
        self.cfg.update(json.loads((HERE / 'configs/ablation.json').read_text()))
        self.cfg['native_runtimes'] = {'alfworld': 'venv/bin/python'}
        config.write_text(json.dumps(self.cfg))
        loaded = load_config(config)
        self.assertEqual(loaded['native_runtimes']['alfworld'], str(executable))
        self.assertNotEqual(loaded['native_runtimes']['alfworld'], str(executable.resolve()))

    def test_kev_provider_does_not_forget_runtime_identity_between_calls(self):
        card = card_response()
        sent = []

        def transport(kind, endpoint, request):
            if request is None:
                return deepcopy(card)
            sent.append(deepcopy(request))
            return answer()

        client = Providers(self.cfg, transport=transport)
        client.call('kev', payload())
        card['runtime_identity']['versions']['torch'] = 'changed-kernel-version'
        with self.assertRaises(ProviderError):
            client.call('kev', payload())
        self.assertEqual(len(sent), 1)

    def test_kev_actual_runtime_is_frozen_across_new_workers_or_resume(self):
        card = card_response()

        def client():
            return Providers(self.cfg, transport=lambda *args: deepcopy(card))

        original = runner.freeze_kev_runtime(self.cfg, client())
        self.assertEqual(runner.freeze_kev_runtime(self.cfg, client()), original)
        card['runtime_identity']['versions']['transformers'] = 'different-runtime'
        with self.assertRaises(ValueError):
            runner.freeze_kev_runtime(self.cfg, client())
        recorded = json.loads((Path(self.cfg['output']) / 'runtime-identities/kev.json').read_text())
        self.assertEqual(recorded, original)


if __name__ == '__main__':
    unittest.main()
