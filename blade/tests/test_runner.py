from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blade import policy,runner
from blade.ledger import Ledger,Halt
from blade.support import HERE,digest


class FakeGame:
    def __init__(self,name,case):self.name=name;self.case=case;self.n=0;self.executed=[]
    def observe(self):return {'environment':self.name,'mission':'native task','observation':'blocked '+str(self.n),
        'native_instruction':'native description','action_descriptions':{'blocked':'try blocked action','turn':'turn'}}
    def actions(self):return ['blocked','turn']
    def step(self,action):
        self.executed.append(action);self.n+=1
        return {'done':self.n>=self.case['horizon'],'reward':0,'observation':self.observe()}
    def metrics(self):return {'won':False,'done':self.n>=self.case['horizon'],'score':0,'progression':0,'steps':self.n,'native_steps':self.n}
    def close(self):pass


class NeverDoneGame(FakeGame):
    """Public synthetic engine: native steps intentionally differ from decisions."""
    def step(self,action):
        self.executed.append(action);self.n+=1
        return {'done':False,'reward':0,'observation':self.observe()}
    def metrics(self):
        return {'won':False,'done':False,'terminated':False,'truncated':False,
                'score':0,'progression':0,'steps':self.n,'native_steps':2*self.n+1,
                'horizon':self.case['horizon']}


class EarlyDoneGame(NeverDoneGame):
    def step(self,action):
        result=super().step(action);return {**result,'done':self.n>=13}
    def metrics(self):
        return {**super().metrics(),'done':self.n>=13,'terminated':self.n>=13}


class FastLedger:
    """Pure in-process replies for actual 2000-decision boundary checks."""
    def __init__(self):self.calls=[]
    def call(self,kind,request,eid,step,role):
        state=json.loads(request['input']) if role=='goal' else request['state']
        self.calls.append((kind,step,role,state['remaining_episode_actions'],
                           state['remaining_goal_horizon']))
        return str(len(self.calls)),target(request) if role=='goal' else choice(request)


def choice(payload):
    return {'model':payload['model'],'answers':{'action':{'type':'choice','choice':'blocked','probabilities':{'blocked':.8,'turn':.2}}},'usage':{'input_tokens':500,'output_tokens':0}}


def target(payload):
    return {'status':'completed','model':payload['model'],'error':None,'incomplete_details':None,'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':'{"goal":"public desired state"}'}]}],'usage':{'input_tokens':500,'output_tokens':100}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.cfg=json.loads((HERE/'configs/ablation.json').read_text());self.cfg.pop('execution_scope',None);self.cfg['ablation_actors']=['jev','d3'];self.cfg.update(output=self.tmp.name,protocol_hash='a'*64)
        self.cfg['budgets']={'jev_usd':10,'gpt_usd':10};self.case={'id':'fake/case','environment':'babyai','horizon':20,'group':'fixed'}
        g=FakeGame('babyai',self.case);p=g.observe()
        self.row={'case':self.case,'initial_sha256':digest(p),'actions':g.actions(),
                  'action_descriptions_sha256':digest(p['action_descriptions']),
                  'decision_cap':20,'goal_slots':policy.goal_slots(20,4)}
        self.calls=[]
        def wrapped(kind,func):
            def call(payload,settings):self.calls.append((kind,deepcopy(payload)));return func(payload)
            return call
        self.ledger=Ledger(self.tmp.name,self.cfg,providers={k:wrapped(k,f) for k,f in [('jev',choice),('d3',choice),('kev',choice),('gpt',target)]})
        self.addCleanup(self.ledger.close)

    def cap_fixture(self,name='crafter',case_id='cap/case'):
        case={'id':case_id,'environment':name,'horizon':2000,'group':'fixed'}
        game=NeverDoneGame(name,case);public=game.observe()
        return {'case':case,'initial_sha256':digest(public),'actions':game.actions(),
                'action_descriptions_sha256':digest(public['action_descriptions']),
                'decision_cap':2000,'goal_slots':policy.goal_slots(2000,8)}

    def test_h0_uses_no_teacher_and_resume_sends_no_new_request(self):
        runner.execute_episode(self.cfg,'babyai',self.row,'jev-flat-h0',self.ledger,FakeGame)
        self.assertEqual(len(self.calls),20)
        for kind,p in self.calls:
            self.assertEqual(kind,'jev');self.assertNotIn('previous_interactions',p['state']);self.assertNotIn('desired_state',p['state'])
        runner.execute_episode(self.cfg,'babyai',self.row,'jev-flat-h0',self.ledger,FakeGame)
        self.assertEqual(len(self.calls),20)

    def test_prepare_freezes_effective_cap_and_fixed_goal_schedule(self):
        from blade import environment
        class PreparedGame(NeverDoneGame):
            def identity(self):return {'native':'unchanged'}
        for name in ('crafter',):
            case=self.cap_fixture(name)['case'];before=deepcopy(case)
            with patch.object(environment,'make_cases',return_value=[case]),\
                    patch.object(environment,'Game',PreparedGame),\
                    patch.object(runner,'native_identity',return_value={'native':'identity'}),\
                    patch.object(policy,'payload',wraps=policy.payload) as payload,\
                    patch('builtins.print'):
                runner.prepare_native(self.cfg,name)
            manifest=json.loads((Path(self.tmp.name)/'manifests'/f'{name}.json').read_text())
            self.assertEqual(manifest['cases'][0]['decision_cap'],2000)
            self.assertEqual(manifest['cases'][0]['goal_slots'],[8,257,506,755,1004,1253,1502,1751])
            self.assertEqual(payload.call_args.args[5]['remaining_episode_actions'],2000)
            self.assertEqual(case,before)
        self.assertEqual(self.calls,[])

    def test_guided_complete_dependencies_and_no_loop_correction(self):
        games=[]
        def make(n,c):g=FakeGame(n,c);games.append(g);return g
        runner.execute_episode(self.cfg,'babyai',self.row,'jev-w2',self.ledger,make)
        self.assertEqual(games[0].executed,['blocked']*20)
        actor=[p for k,p in self.calls if k=='jev'];goals=[p for k,p in self.calls if k=='gpt']
        self.assertEqual(len(actor),8+12*3);self.assertEqual(len(goals),4)
        for request in actor[:8]:self.assertIn('previous_interactions',request['state']);self.assertNotIn('desired_state',request['state'])
        for p0,ph,q in zip(actor[8::3],actor[9::3],actor[10::3]):
            self.assertNotIn('previous_interactions',p0['state']);self.assertNotIn('desired_state',p0['state'])
            self.assertIn('previous_interactions',ph['state']);self.assertNotIn('desired_state',ph['state'])
            self.assertNotIn('previous_interactions',q['state']);self.assertIn('desired_state',q['state'])
        statistics=self.ledger.statistics();uses=next(iter(statistics['logical_by_episode'].values()))
        self.assertEqual(uses['jev']['requests'],44);self.assertEqual(uses['gpt']['requests'],4)

    def test_history_and_joint_ablation_inputs_and_dependency_counts(self):
        for actor in ('jev', 'd3'):
            for suffix in ('-history-h4', '-joint-h4'):
                method = actor + suffix
                before = len(self.calls)
                runner.execute_episode(self.cfg, 'babyai', self.row, method, self.ledger, FakeGame)
                recorded = [p for k, p in self.calls[before:] if k == actor]
                # Both arms make exactly one logical actor request per action;
                # a cached warm-up may reduce the number of physical calls.
                eid = 'babyai|fake/case|' + method
                uses = self.ledger.statistics()['logical_by_episode'][eid]
                self.assertEqual(uses[actor]['requests'], 20)
                self.assertEqual(uses['gpt']['requests'], 4 if suffix == '-joint-h4' else 0)
                for request in recorded:
                    state = request['state']
                    self.assertIn('previous_interactions', state)
                    if suffix == '-joint-h4' and state['actions_taken'] >= 8:
                        self.assertIn('desired_state', state)
                    else:
                        self.assertNotIn('desired_state', state)

    def test_short_benchmark_ablations_cannot_run_on_crafter(self):
        row = self.cap_fixture()
        for actor in ('jev', 'd3'):
            for suffix in ('-history-h4', '-joint-h4'):
                with self.subTest(method=actor + suffix), self.assertRaises(Halt):
                    runner.execute_episode(self.cfg, 'crafter', row, actor + suffix, self.ledger, NeverDoneGame)
        self.assertEqual(self.calls, [])

    def test_interface_failure_remains_failed_no_resampling(self):
        class BadGame(FakeGame):
            def step(self,a):raise RuntimeError('retained native interface failure')
        with self.assertRaises(RuntimeError):runner.execute_episode(self.cfg,'babyai',self.row,'jev-flat-h0',self.ledger,BadGame)
        self.assertTrue((Path(self.tmp.name)/'STOP').exists())
        self.assertEqual(runner.execute_episode(self.cfg,'babyai',self.row,'jev-flat-h0',self.ledger,FakeGame),'failed')

    def test_all_thirteen_short_benchmark_methods_shared_receipts_and_logical_cost(self):
        from blade.report import report
        self.cfg['environments']=['babyai']
        manifest=Path(self.tmp.name)/'manifests/babyai.json';manifest.parent.mkdir()
        manifest.write_text(json.dumps({'cases':[self.row]}))
        for method in policy.METHODS:
            runner.execute_episode(self.cfg,'babyai',self.row,method,self.ledger,FakeGame)
        result=report(self.cfg)
        self.assertEqual(result['status'],'complete');self.assertEqual(result['complete'],13)
        logical=result['request_accounting']['logical_by_episode']
        self.assertEqual(sum(x['gpt']['requests'] for x in logical.values()),32)
        self.assertEqual(result['request_accounting']['physical']['gpt']['requests'],4)
        for row in result['methods']:
            self.assertEqual(row['target'],1);self.assertEqual(row['complete'],1)
            if not policy.needs_goal(row['method']):self.assertEqual(row['logical_usage']['gpt']['requests'],0)
            else:self.assertEqual(row['logical_usage']['gpt']['requests'],4)
            self.assertEqual(row['wins'],0)
            self.assertEqual(row['decision_cap_episodes'],0)
            self.assertEqual(row['native_done_episodes'],1)

    def test_all_nine_main_methods_stop_at_2000_and_report_custom_caps(self):
        from blade.report import report
        cfg=deepcopy(self.cfg);cfg['output']=str(Path(self.tmp.name)/'boundary')
        cfg['environments']=['crafter']
        manifest_dir=Path(cfg['output'])/'manifests';manifest_dir.mkdir(parents=True)
        with patch.object(runner.os,'fsync'),patch('builtins.print'):
            for name in cfg['environments']:
                row=self.cap_fixture(name)
                (manifest_dir/f'{name}.json').write_text(json.dumps({'cases':[row]}))
                for method in policy.MAIN_METHODS:
                    with self.subTest(environment=name,method=method):
                        ledger=FastLedger();games=[]
                        def make(n,c):
                            game=NeverDoneGame(n,c);games.append(game);return game
                        # Interrupt after persisting the last allowed action.
                        # Resume must replay and finalize with no new request.
                        if name=='crafter' and method=='jev-w2':
                            original=runner.append
                            def interrupt(path,value):
                                original(path,value)
                                if '/episodes/' in str(path) and value.get('step')==1999:
                                    raise KeyboardInterrupt()
                            with patch.object(runner,'append',side_effect=interrupt):
                                with self.assertRaises(KeyboardInterrupt):
                                    runner.execute_episode(cfg,name,row,method,ledger,make)
                            before_resume=len(ledger.calls)
                            runner.execute_episode(cfg,name,row,method,ledger,make)
                            self.assertEqual(len(ledger.calls),before_resume)
                        else:
                            runner.execute_episode(cfg,name,row,method,ledger,make)
                        self.assertTrue(all(len(g.executed)==2000 for g in games))
                        self.assertTrue(all(g.case['horizon']==row['case']['horizon'] for g in games))
                        self.assertEqual(max(c[1] for c in ledger.calls),1999)
                        self.assertTrue(all(remaining==2000-step for _,step,_,remaining,_ in ledger.calls))
                        goals=[c[1] for c in ledger.calls if c[2]=='goal']
                        self.assertEqual(goals,[] if policy.is_baseline(method) else row['goal_slots'])
                        self.assertEqual(len(ledger.calls),2000 if policy.is_baseline(method) else 8+1992*3+8)
                        summary_path,step_path,_=runner.episode_paths(cfg['output'],f'{name}|cap/case|{method}')
                        summary=json.loads(summary_path.read_text())
                        self.assertEqual(summary['status'],'complete')
                        self.assertEqual(summary['decisions'],2000)
                        self.assertEqual(summary['decision_cap'],2000)
                        self.assertEqual(summary['native_steps'],4001)
                        self.assertEqual(summary['termination_reason'],'decision_cap')
                        self.assertTrue(summary['decision_cap_reached'])
                        self.assertFalse(summary['native_done'])
                        for flag in ('won','done','terminated','truncated'):
                            self.assertFalse(summary['metrics'][flag])
                        self.assertEqual(len(runner.read_records(step_path)),2000)
                        before_resume=len(ledger.calls)
                        runner.execute_episode(cfg,name,row,method,ledger,make)
                        self.assertEqual(len(ledger.calls),before_resume)
            result=report(cfg,bootstrap=False)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['decision_cap_episodes'],9)
        self.assertEqual(result['native_done_episodes'],0)
        self.assertIn('not a game win',result['interpretation'])
        self.assertIn('original paper full-horizon',result['interpretation'])
        for row in result['methods']:
            self.assertEqual(row['decision_caps'],[2000])
            self.assertEqual(row['decision_cap_episodes'],1)
            self.assertEqual(row['wins'],0)
            self.assertEqual(row['failed'],0)
            self.assertEqual(row['average_decisions_completed'],2000)
            self.assertEqual(row['average_native_steps_completed'],4001)
            self.assertEqual(row['total_decisions_completed'],2000)
            self.assertEqual(row['total_native_steps_completed'],4001)

    def test_native_done_takes_priority_before_or_at_decision_cap(self):
        row=self.cap_fixture();cfg=deepcopy(self.cfg)
        with patch.object(runner.os,'fsync'),patch('builtins.print'):
            for method in policy.MAIN_METHODS:
                ledger=FastLedger()
                runner.execute_episode(cfg,'crafter',row,method,ledger,EarlyDoneGame)
                path,_,_=runner.episode_paths(cfg['output'],f'crafter|cap/case|{method}')
                summary=json.loads(path.read_text())
                self.assertEqual(summary['decisions'],13)
                self.assertEqual(summary['termination_reason'],'native_done')
                self.assertTrue(summary['native_done'])
                self.assertFalse(summary['decision_cap_reached'])
                self.assertEqual([c[1] for c in ledger.calls if c[2]=='goal'],
                                 [] if policy.is_baseline(method) else [8])
            row=self.cap_fixture('crafter','native/at-cap');ledger=FastLedger()
            runner.execute_episode(cfg,'crafter',row,'d3-flat-h0',ledger,FakeGame)
        path,_,_=runner.episode_paths(cfg['output'],'crafter|native/at-cap|d3-flat-h0')
        summary=json.loads(path.read_text())
        self.assertEqual(summary['decisions'],2000)
        self.assertEqual(summary['termination_reason'],'native_done')
        self.assertFalse(summary['decision_cap_reached'])

    def test_frozen_cap_schedule_and_resume_identity_mismatches_are_rejected(self):
        for field,value in [('decision_cap',19),('goal_slots',[8,12,15,18])]:
            row=deepcopy(self.row);row[field]=value
            with self.subTest(field=field),self.assertRaises(Halt):
                runner.execute_episode(self.cfg,'babyai',row,'jev-w2',self.ledger,FakeGame)
        self.assertEqual(self.calls,[])
        runner.execute_episode(self.cfg,'babyai',self.row,'jev-w2',self.ledger,FakeGame)
        path,_,goal_path=runner.episode_paths(self.tmp.name,'babyai|fake/case|jev-w2')
        original=json.loads(path.read_text());before=len(self.calls)
        for field,value in [('decision_cap',19),('goal_slots',[8,12,15,18]),
                            ('episode_identity_sha256','wrong')]:
            path.write_text(json.dumps({**original,field:value}))
            with self.subTest(field=field),self.assertRaises(Halt):
                runner.execute_episode(self.cfg,'babyai',self.row,'jev-w2',self.ledger,FakeGame)
        path.write_text(json.dumps(original))
        altered=deepcopy(self.row);altered['case']['group']='different identity'
        with self.assertRaises(Halt):
            runner.execute_episode(self.cfg,'babyai',altered,'jev-w2',self.ledger,FakeGame)
        goals=runner.read_records(goal_path);goals[0]['step']=9
        goal_path.write_text(''.join(json.dumps(g)+'\n' for g in goals))
        with self.assertRaises(Halt):
            runner.execute_episode(self.cfg,'babyai',self.row,'jev-w2',self.ledger,FakeGame)
        self.assertEqual(len(self.calls),before)

    def test_trajectory_beyond_cap_is_rejected_before_requests(self):
        runner.execute_episode(self.cfg,'babyai',self.row,'jev-flat-h0',self.ledger,NeverDoneGame)
        _,step_path,_=runner.episode_paths(self.tmp.name,'babyai|fake/case|jev-flat-h0')
        extra=runner.read_records(step_path)[-1];extra['step']=20
        runner.append(step_path,extra);before=len(self.calls)
        with self.assertRaisesRegex(Halt,'exceeds frozen decision cap'):
            runner.execute_episode(self.cfg,'babyai',self.row,'jev-flat-h0',self.ledger,NeverDoneGame)
        self.assertEqual(len(self.calls),before)

    def test_partial_resume_replays_past_actions_without_new_requests_for_them(self):
        original=runner.append
        def interrupt(path,value):
            original(path,value)
            if '/episodes/' in str(path) and value.get('step')==9:raise KeyboardInterrupt()
        with patch.object(runner,'append',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                runner.execute_episode(self.cfg,'babyai',self.row,'jev-w2',self.ledger,FakeGame)
        partial=len(self.calls)
        self.assertGreater(partial,0)
        runner.execute_episode(self.cfg,'babyai',self.row,'jev-w2',self.ledger,FakeGame)
        self.assertEqual(len(self.calls),44+4)
        summary,_,_=runner.episode_paths(self.tmp.name,'babyai|fake/case|jev-w2')
        value=json.loads(summary.read_text())
        self.assertEqual(value['resumed_steps'],10);self.assertEqual(value['steps'],20)

    def test_one_coordinator_per_output(self):
        import fcntl
        with (Path(self.tmp.name)/'coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(Halt):runner.run(self.cfg,['jev-flat-h0'])
            fcntl.flock(lock,fcntl.LOCK_UN)

    def test_credentials_are_fully_validated_before_any_model_request(self):
        import os
        from blade.providers import ProviderError
        with patch.dict(os.environ,{'JEV_KEYS':'placeholder','OPENAI_API_KEY':'x'*17000},clear=True):
            with self.assertRaises(ProviderError):runner.preload_credentials(self.cfg,['jev-flat-h0','jev-w2'])
        self.assertEqual(len(self.calls),0)

    def test_model_manifest_cannot_be_self_asserted(self):
        from blade.d3_model import verify
        p=Path(self.tmp.name)/'MODEL_MANIFEST.json';p.write_text(json.dumps({'identity':{'model_sha256':'0e439bebbbbf80d9e1b601c56c55a6fc3a6502b4e396cbbb5f3c3f3acc1d607b'},'files_sha256':{},'files_bytes':{}}))
        with self.assertRaises(ValueError):verify(self.tmp.name)


if __name__=='__main__':unittest.main()
