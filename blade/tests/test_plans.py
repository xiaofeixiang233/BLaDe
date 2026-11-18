"""Plan assignment, entry-point and no-dispatch guards."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from blade import policy, runner
from blade.plans import PLANS, planned_counts, selected_methods
from blade.support import HERE, ROOT, load_config, scientific_config


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = json.loads((HERE / 'configs/ablation.json').read_text())
        self.cfg['output'] = str(Path(self.tmp.name) / 'output')
        self.path = Path(self.tmp.name) / 'config.json'
        self.path.write_text(json.dumps(self.cfg))

    def test_exact_counts_and_only_four_new_methods(self):
        cfg = load_config(self.path)
        self.assertEqual(planned_counts(cfg)['total_episodes'], 1460)
        self.assertEqual(planned_counts(cfg)['distinct_cases'], 365)
        self.assertEqual(planned_counts(cfg)['episodes_by_environment'],
                         {'babyai': 500, 'minihack': 160, 'babaisai': 800})
        self.assertEqual(len(selected_methods(cfg, 'all')), 4)
        for actor in ('jev', 'd3'):
            self.assertEqual(len(selected_methods(cfg, actor)), 2)
        self.assertEqual(policy.methods_for_environment(cfg, 'crafter'), [])
        self.assertEqual(policy.methods_for_environment(cfg, 'nle'), [])
        cfg['profile'] = 'smoke'
        self.assertEqual(planned_counts(cfg)['total_episodes'], 12)

    def test_scope_cannot_expand_in_configuration(self):
        for kind in ('crafter', 'ablation', 'jev_short'):
            cfg = deepcopy(self.cfg)
            if kind == 'crafter':
                cfg['environments'].append('crafter')
                cfg['execution_scope']['crafter'] = ['kev-flat-h0']
            elif kind == 'ablation':
                cfg['execution_scope']['babyai'].append('kev-joint-h4')
            else:
                cfg['execution_scope']['babyai'].append('jev-flat-h0')
            self.path.write_text(json.dumps(cfg))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_config(self.path)

    def test_initialization_retains_existing_file_and_creates_null_budget(self):
        runner.initialize(self.path, 'ablation')
        self.assertEqual(json.loads(self.path.read_text()), self.cfg)
        new = Path(self.tmp.name) / 'nested/new-config.json'
        runner.initialize(new, 'ablation')
        loaded = load_config(new)
        self.assertEqual(loaded['budgets'], {'jev_usd': None, 'gpt_usd': None})
        self.assertEqual(loaded['runtime_root'], str(HERE / '.runtime'))
        self.assertEqual(loaded['execution_scope'], PLANS['ablation']['scope'])

    def test_entry_requires_explicit_execute_before_any_runner_call(self):
        with patch.object(runner, 'run') as run, self.assertRaises(SystemExit) as stop:
            runner.main(['run', '--config', str(self.path), '--methods', 'jev-history-h4'])
        self.assertEqual(stop.exception.code, 2)
        run.assert_not_called()

    def test_main_plan_covers_table_one(self):
        main = Path(self.tmp.name) / 'main.json'
        runner.initialize(main, 'main')
        cfg = load_config(main)
        counts = planned_counts(cfg)
        self.assertEqual(counts['distinct_cases'], 499)
        self.assertEqual(counts['total_episodes'], 499 * 9)
        self.assertEqual(selected_methods(cfg, 'all'), list(policy.MAIN_METHODS))
        self.assertEqual(len(selected_methods(cfg, 'kev')), 3)
        self.assertEqual(policy.methods_for_environment(cfg, 'alfworld'), list(policy.MAIN_METHODS))
        self.assertEqual(policy.methods_for_environment(cfg, 'crafter'), [])
        raw = json.loads(main.read_text())
        raw['execution_scope']['alfworld'].append('jev-joint-h4')
        main.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            load_config(main)

    def test_credential_alias_rotation_does_not_change_science_identity(self):
        first = deepcopy(self.cfg)
        first['credential_slots']['gpt'] = 'gpt-key-01'
        second = deepcopy(first)
        second['credential_slots']['gpt'] = 'gpt-key-02'
        self.assertEqual(scientific_config(first), scientific_config(second))

    def test_coordinator_dispatches_only_assigned_environments_for_each_stage(self):
        cfg = load_config(self.path)
        cfg['budgets'] = {'jev_usd': 5, 'gpt_usd': 5}
        manifests = Path(cfg['output']) / 'manifests'
        manifests.mkdir(parents=True)
        for name in cfg['environments']:
            (manifests / (name + '.json')).write_text(json.dumps({'cases': [{'case': {'id': 'fake'}}]}))
        for actor, expected in (('jev', ['babyai','minihack','babaisai']),
                                ('d3', ['babyai','minihack','babaisai'])):
            dispatched = []

            def process(args, **kwargs):
                name = args[args.index('--environment') + 1]
                dispatched.append(name)
                return SimpleNamespace(returncode=0)

            with patch.object(runner, 'check_freeze'), patch.object(runner, 'preload_credentials'), \
                    patch.object(runner.Providers, 'health',return_value={'runtime_identity':{'offline':'mock'}}), \
                    patch.object(runner, 'freeze_kev_runtime',return_value={}), \
                    patch.object(runner.subprocess, 'run', side_effect=process), \
                    patch('blade.report.report'):
                runner._run_locked(cfg, selected_methods(cfg, actor))
            self.assertCountEqual(dispatched, expected)
        events=[json.loads(line) for line in (Path(cfg['output'])/'execution-events.jsonl').read_text().splitlines()]
        self.assertEqual([e['event'] for e in events],['run_start','run_stage_complete']*2)
        self.assertTrue(all(e['stage_wall_seconds']>=0 for e in events if e['event']=='run_stage_complete'))

    def test_d3_launcher_retains_virtualenv_interpreter_path(self):
        from blade import launch_d3
        import sys
        root=Path(self.tmp.name)
        python=root/'venv/bin/python';python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        model=root/'model';model.mkdir();(model/'MODEL_MANIFEST.json').write_text('{}')
        commands=[]

        def process(args,**kwargs):
            commands.append(args)
            return SimpleNamespace(returncode=1 if 'has-session' in args else 0)

        argv=['launch_d3','--python',str(python),'--model-dir',str(model),
              '--gpu-pairs','0,1','--ports','8902','--log-dir',str(root/'logs')]
        with patch.object(sys,'argv',argv),patch.object(launch_d3.shutil,'which',return_value='/usr/bin/tmux'), \
                patch.object(launch_d3.subprocess,'run',side_effect=process):
            launch_d3.main()
        shell=next(args[-1] for args in commands if 'new-session' in args)
        self.assertIn(str(python),shell)
        self.assertNotEqual(str(python),str(python.resolve()))

    def test_pending_report_uses_only_twelve_scoped_smoke_episodes(self):
        from blade.report import report
        cfg = load_config(self.path)
        manifests = Path(cfg['output']) / 'manifests'
        manifests.mkdir(parents=True)
        for name in cfg['environments']:
            row = {'case': {'id': name + '/fake', 'group': 'fake'}, 'decision_cap': 50}
            (manifests / (name + '.json')).write_text(json.dumps({'cases': [row]}))
        result = report(cfg, bootstrap=False)
        self.assertEqual(result['target_episodes'], 12)
        self.assertEqual(len(result['methods']), 12)
        self.assertEqual(len(result['ablation_comparisons']), 6)
        self.assertEqual(result['comparisons'], [])
        self.assertEqual(result['complete'], 0)
        self.assertEqual(result['status'], 'provisional')
        self.assertFalse((Path(cfg['output']) / 'requests.sqlite3').exists())


if __name__ == '__main__':
    unittest.main()
