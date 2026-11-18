"""Frozen experiment plans: which methods run on which benchmark.

A configuration names exactly one plan through its ``version`` field and must
repeat that plan's environments and execution scope verbatim. No game state is
read here.
"""
from .policy import ABLATION_METHODS, MAIN_METHODS

COUNTS = dict(babyai=125, minihack=40, babaisai=200, alfworld=134)
SHORT = ('babyai', 'minihack', 'babaisai')
STAGES = ('jev', 'd3', 'kev', 'all')

PLANS = {
    # Table 1: System I vs. BLaDe (w = 0.5, w = 2) for all three actors.
    'main': {
        'version': 'blade-main-v1',
        'environments': SHORT + ('alfworld',),
        'methods': MAIN_METHODS,
    },
    # History-only and joint-prompt controls for Jev and d3.
    'ablation': {
        'version': 'blade-ablation-v1',
        'environments': SHORT,
        'methods': ABLATION_METHODS,
    },
}
for _plan in PLANS.values():
    _plan['scope'] = {name: list(_plan['methods']) for name in _plan['environments']}
VERSIONS = {plan['version']: name for name, plan in PLANS.items()}


def plan_for(cfg):
    """Return the frozen plan a configuration claims, or raise ValueError."""
    name = VERSIONS.get(cfg.get('version'))
    if name is None:
        raise ValueError('Unknown plan version; use one of ' + ', '.join(sorted(VERSIONS)))
    return PLANS[name]


def matches_plan(cfg):
    plan = plan_for(cfg)
    return (cfg.get('environments') == list(plan['environments'])
            and cfg.get('execution_scope') == plan['scope'])


def selected_methods(cfg, stage):
    if stage not in STAGES:
        raise ValueError('Choose jev, d3, kev or all')
    methods = [method for method in plan_for(cfg)['methods']
               if any(method in cfg['execution_scope'][name] for name in cfg['environments'])]
    if stage == 'all':
        return methods
    return [method for method in methods if method.startswith(stage + '-')]


def planned_counts(cfg):
    cases = {name: 1 if cfg['profile'] == 'smoke' else COUNTS[name]
             for name in cfg['environments']}
    return {'cases': cases, 'distinct_cases': sum(cases.values()),
            'methods': selected_methods(cfg, 'all'),
            'episodes_by_environment': {name: count * len(cfg['execution_scope'][name])
                                        for name, count in cases.items()},
            'total_episodes': sum(count * len(cfg['execution_scope'][name])
                                  for name, count in cases.items())}
