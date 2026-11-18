"""Keep main v3 scope/caps explicit and incompatible with earlier freezes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from blade.support import HERE, load_config


class CapConfigurationTests(unittest.TestCase):
    def test_fixed_caps_change_scientific_identity_and_old_version_is_rejected(self):
        source = json.loads((HERE / 'configs/ablation.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.local.json'
            path.write_text(json.dumps(source))
            loaded = load_config(path)
            self.assertEqual(loaded['decision_caps'], {'crafter': 2000})
            self.assertEqual(loaded['version'], 'blade-ablation-v1')
            for cap in ({'crafter': 2000, 'nle': 2000}, {'crafter': 500},
                        {'crafter': 2000, 'alfworld': 50}, {}, {'crafter': 2000.}):
                changed = deepcopy(source)
                changed['decision_caps'] = cap
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_config(path)
            for old_version in ('balrog-six-methods-v1', 'balrog-six-methods-v2-cap2000'):
                changed = deepcopy(source); changed['version'] = old_version
                path.write_text(json.dumps(changed))
                with self.subTest(version=old_version), self.assertRaises(ValueError):
                    load_config(path)

    def test_main_and_ablation_scope_is_fixed_before_evaluation(self):
        from blade import policy
        source = json.loads((HERE / 'configs/ablation.json').read_text())
        self.assertEqual(source['environments'], ['babyai', 'minihack', 'babaisai'])
        self.assertEqual(source['ablation_actors'], ['jev', 'd3'])
        self.assertEqual(source['methods'], list(policy.METHODS))
        self.assertEqual(len(policy.MAIN_METHODS), 9)
        self.assertEqual(len(policy.METHODS), 13)
        for name in ('babyai', 'minihack', 'babaisai'):
            self.assertEqual(policy.methods_for_environment(source, name),
                ['jev-history-h4','jev-joint-h4','d3-history-h4','d3-joint-h4'])
        self.assertEqual(policy.methods_for_environment(source, 'alfworld'), [])
        self.assertEqual(policy.methods_for_environment(source, 'crafter'), [])
        self.assertNotIn('kev-history-h4', policy.METHODS)
        self.assertNotIn('kev-joint-h4', policy.METHODS)

    def test_budget_template_does_not_authorize_a_run(self):
        source = json.loads((HERE / 'configs/ablation.json').read_text())
        self.assertEqual(source['budgets'], {'jev_usd': None, 'gpt_usd': None})
        self.assertEqual(source['output'], '../../outputs/ablation')


if __name__ == '__main__':
    unittest.main()
