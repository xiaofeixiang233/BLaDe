"""Evaluation-only summaries. No value here is passed to model inference."""
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean

from . import policy
from .ledger import Ledger
from .support import digest,write_json


def aggregate_usage(rows,kind):
    values=[r.get(kind,{}) for r in rows]
    unknown=sum(v.get('unknown_billing_requests',0) for v in values)
    known=sum(v.get('known_cost_subtotal_usd') or 0 for v in values)
    result={key:sum(v.get(key,0) or 0 for v in values) for key in (
        'requests','failed','input_tokens','output_tokens','cached_input_tokens','reasoning_tokens',
        'unknown_input_usage_requests','unknown_output_usage_requests','receipt_latency_seconds','unknown_latency_requests')}
    result.update(unknown_billing_requests=unknown,known_cost_subtotal_usd=known if kind not in ('d3','kev') else None,
                  estimated_cost_usd=known if kind not in ('d3','kev') and not unknown else None)
    return result


def paired(groups,repeats=2000):
    """Paired bootstrap over the frozen manifest's case groups, seed20261001."""
    if not groups:return None
    values=list(groups.values());total=sum(len(v) for v in values)
    point=sum(sum(v) for v in values)/total
    rng=random.Random(20261001);samples=[]
    for _ in range(repeats):
        sampled=[rng.choice(values) for _ in values]
        samples.append(sum(sum(v) for v in sampled)/sum(len(v) for v in sampled))
    samples.sort()
    return {'difference':point,'bootstrap_95_interval':[samples[int(.025*repeats)],samples[min(repeats-1,int(.975*repeats))]],
            'pairs':total,'groups':len(values),'resamples':repeats,'bootstrap_seed':20261001}


def report(cfg,bootstrap=True,include_timing=True):
    out=Path(cfg['output']);rows=[];by_key={}
    for name in cfg['environments']:
        manifest=json.loads((out/'manifests'/f'{name}.json').read_text())
        for entry in manifest['cases']:
            case=entry['case']
            for method in policy.methods_for_environment(cfg,name):
                eid=name+'|'+case['id']+'|'+method
                p=out/'episodes'/f'{digest(eid)}.json'
                row=json.loads(p.read_text()) if p.exists() else {
                    'id':eid,'environment':name,'case':case,'method':method,
                    'decision_cap':entry['decision_cap'],'status':'pending'}
                rows.append(row);by_key[(name,case['id'],method)]=row
    statistics={'physical':{},'logical_by_episode':{}};efficiency_rows={'requests':[],'logical_uses':[]}
    if (out/'requests.sqlite3').exists():
        ledger=Ledger(out,cfg)
        try:
            statistics=ledger.statistics();efficiency_rows=ledger.efficiency_rows()
        finally:ledger.close()
    summaries=[];comparisons=[]
    for name in cfg['environments']:
        for method in policy.methods_for_environment(cfg,name):
            selected=[r for r in rows if r['environment']==name and r['method']==method]
            complete=[r for r in selected if r['status']=='complete'];metrics=[r['metrics'] for r in complete]
            wins=sum(bool(m.get('won')) for m in metrics)
            usage=[statistics['logical_by_episode'].get(r['id'],{}) for r in selected]
            steps=[r['decisions'] for r in complete]
            success_steps=[r['decisions'] for r in complete if r['metrics'].get('won')]
            native_steps=[m['native_steps'] for m in metrics]
            success_native_steps=[m['native_steps'] for m in metrics if m.get('won')]
            row={'environment':name,'method':method,'target':len(selected),'complete':len(complete),
                 'failed':sum(r['status']=='failed' for r in selected),
                 'budget_paused':sum(r['status']=='budget_paused' for r in selected),
                 'pending_or_running':sum(r['status'] in ('pending','running') for r in selected),
                 'decision_caps':sorted({r['decision_cap'] for r in selected}),
                 'decision_cap_episodes':sum(bool(r.get('decision_cap_reached')) for r in complete),
                 'native_done_episodes':sum(bool(r.get('native_done')) for r in complete),
                 'wins':wins,'wins_over_frozen_denominator':wins/len(selected),
                 'primary_metric':'progression' if name=='crafter' else 'success',
                 'average_achievement_count_completed':mean(m.get('achievement_count',0) for m in metrics) if name=='crafter' and metrics else None,
                 'final_comparison_ready':len(complete)==len(selected),
                 'average_score_completed':mean(m.get('score',0) for m in metrics) if metrics else None,
                 'average_progress_completed':mean(m.get('progression',0) for m in metrics) if metrics else None,
                 'average_decisions_completed':mean(steps) if steps else None,
                 'average_decisions_successful':mean(success_steps) if success_steps else None,
                 'total_decisions_completed':sum(steps),
                 'average_native_steps_completed':mean(native_steps) if native_steps else None,
                 'average_native_steps_successful':mean(success_native_steps) if success_native_steps else None,
                 'total_native_steps_completed':sum(native_steps),
                 'logical_usage':{k:aggregate_usage(usage,k) for k in ('jev','gpt','d3','kev')}}
            summaries.append(row)
            baseline=policy.actor_kind(method)+'-flat-h0'
            if not policy.is_baseline(method) and baseline in policy.methods_for_environment(cfg,name):
                baseline=policy.actor_kind(method)+'-flat-h0';diffs={};progress_diffs={};repaired=damaged=0;score_diffs=[]
                for r in complete:
                    b=by_key[(name,r['case']['id'],baseline)]
                    if b['status']!='complete':continue
                    d=int(bool(r['metrics'].get('won')))-int(bool(b['metrics'].get('won')))
                    diffs.setdefault(r['case']['group'],[]).append(d)
                    progress_diffs.setdefault(r['case']['group'],[]).append(r['metrics'].get('progression',0)-b['metrics'].get('progression',0))
                    repaired+=d>0;damaged+=d<0
                    score_diffs.append(r['metrics'].get('score',0)-b['metrics'].get('score',0))
                comparisons.append({'environment':name,'method':method,'baseline':baseline,
                    'repaired':repaired,'damaged':damaged,'paired_success_difference':paired(diffs) if bootstrap else None,
                    'paired_score_difference':mean(score_diffs) if score_diffs else None,
                    'paired_progress_difference':paired(progress_diffs) if bootstrap else None,
                    'complete_cohort':len(complete)==len(selected) and all(by_key[(name,r['case']['id'],baseline)]['status']=='complete' for r in selected)})
    ablation_comparisons=[]
    for name in cfg['environments']:
        if name not in policy.SHORT_BENCHMARKS:continue
        for actor in cfg.get('ablation_actors',[]):
            reference=actor+'-history-h4'
            for suffix in ('-joint-h4',):
                method=actor+suffix;groups={};progress_groups={};repaired=damaged=0;score_differences=[]
                cohort=[r for r in rows if r['environment']==name and r['method']==method]
                if not cohort or reference not in policy.methods_for_environment(cfg,name):continue
                for r in cohort:
                    b=by_key[(name,r['case']['id'],reference)]
                    if r['status']!='complete' or b['status']!='complete':continue
                    difference=int(bool(r['metrics'].get('won')))-int(bool(b['metrics'].get('won')))
                    groups.setdefault(r['case']['group'],[]).append(difference)
                    repaired+=difference>0;damaged+=difference<0
                    score_differences.append(r['metrics'].get('score',0)-b['metrics'].get('score',0))
                    progress_groups.setdefault(r['case']['group'],[]).append(r['metrics'].get('progression',0)-b['metrics'].get('progression',0))
                ablation_comparisons.append({'environment':name,'method':method,'reference':reference,
                    'repaired':repaired,'damaged':damaged,
                    'paired_score_difference':mean(score_differences) if score_differences else None,
                    'paired_success_difference':paired(groups) if bootstrap else None,
                    'paired_progress_difference':paired(progress_groups) if bootstrap else None,
                    'complete_cohort':bool(cohort) and all(r['status']=='complete' and by_key[(name,r['case']['id'],reference)]['status']=='complete' for r in cohort)})
    from .efficiency import summarize_efficiency
    efficiency=summarize_efficiency(rows,efficiency_rows['requests'],efficiency_rows['logical_uses'])
    write_json(out/'efficiency.json',efficiency)
    from .timing import write_timing_report
    timing=write_timing_report(out,rows) if include_timing else None
    result={'status':'complete' if all(r['status']=='complete' for r in rows) else 'provisional',
            'target_episodes':len(rows),'complete':sum(r['status']=='complete' for r in rows),
            'decision_cap_episodes':sum(r['status']=='complete' and bool(r.get('decision_cap_reached')) for r in rows),
            'native_done_episodes':sum(r['status']=='complete' and bool(r.get('native_done')) for r in rows),
            'configured_decision_caps':cfg.get('decision_caps',{}),
            'failed':sum(r['status']=='failed' for r in rows),'methods':summaries,'comparisons':comparisons,
            'request_accounting':statistics,'efficiency':efficiency,'timing':timing,'ablation_comparisons':ablation_comparisons,
            'execution_scope':cfg.get('execution_scope'),
            'main_methods':[method for method in policy.MAIN_METHODS if any(
                method in policy.methods_for_environment(cfg,name) for name in cfg['environments'])],
            'executed_methods':sorted({method for name in cfg['environments']
                for method in policy.methods_for_environment(cfg,name)}),
            'historical_controls_imported':False,
            'ablation_actors':cfg.get('ablation_actors',[]),
            'interpretation':'Only history-H4 and joint-H4 are executed; old Flat and weighted controls are not rerun or imported. '
                             'All denominators use the frozen cohort. Pending/interrupted cases prevent a final comparison. '
                             'Score/progress/steps means are explicitly completed-case summaries. Success-only steps are conditional. '
                             'Only the frozen configured environment-method assignment is included. '
                             'A decision-cap ending is a complete time-limited evaluation, not a game win or the original paper full-horizon evaluation. '
                             'Native engine horizons and native done/terminated/truncated/won metrics remain unchanged. '
                             'Joint versus H4 compares closed-loop policies with the same public input interface and fixed target schedule. '
                             'Prior H0/weighted results are external references only; historical run identities need separate verification.',
            'latency_definition':'Logical receipt latencies count every dependency at full independent-run latency; '
                                 'cached physical receipt time is not the observed wall time. '
                                 'Episode JSONL decision_wall_seconds records actual sequential execution/waiting.',
            'bootstrap_computed':bootstrap,'timing_report_current':include_timing,'local_gpu_cost_usd':None}
    write_json(out/'report.json',result)
    flat=[{k:v for k,v in r.items() if k!='logical_usage'} for r in summaries]
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print('REPORT',result['status'],'completed',result['complete'],'/',len(rows),'failed',result['failed'],flush=True)
    return result
