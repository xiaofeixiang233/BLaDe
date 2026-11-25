"""Prepare and run only the new Jev/D3 history and joint short-game arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from collections import deque
import getpass
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import stat
import subprocess
import sys
import threading
import time

from . import policy
from .providers import Providers,parse_distribution,extract_goal,ProviderError,redact
from .ledger import Ledger,Halt,BudgetHalt,ProviderFailure
from .support import (HERE,ROOT,digest,dumps,write_json,immutable_json,source_identity,
                      scientific_config,load_config,child_environment)


def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(dumps(value)+'\n');f.flush();os.fsync(f.fileno())


def iter_records(path):
    path=Path(path)
    if path.exists():
        with path.open() as f:
            for line in f:yield json.loads(line)


def read_records(path):
    return list(iter_records(path))


def native_python(cfg,name):
    if name in cfg['native_runtimes']:return cfg['native_runtimes'][name]
    group='grid' if name in ('babyai','babaisai') else 'alfworld' if name=='alfworld' else 'nle'
    return str(Path(cfg['runtime_root'])/'venvs'/group/'bin/python')


def native_identity(name):
    from . import environment
    packages={'babyai':['minigrid','gymnasium','numpy'],'babaisai':['baba','gym','numpy'],
              'minihack':['minihack','nle','gym','numpy'],'nle':['nle','gym','numpy'],
              'alfworld':['alfworld','textworld','numpy']}[name]
    versions={}
    for p in packages:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]='source-only'
    return {'wrappers':environment.wrapper_identity(),'packages':versions,'python':sys.version.split()[0]}


def prepare_native(cfg,name):
    from . import environment
    rows=[];installation=None
    for case in environment.make_cases(name,profile=cfg['profile'],seed_start=cfg['seed_start']):
        game=None
        try:
            game=environment.Game(name,case);public=game.observe();menu=game.actions()
            actual_installation=game.identity()
            if installation is None:installation=actual_installation
            elif installation!=actual_installation:raise Halt('Native installation changed during preparation')
            cap=policy.effective_decision_cap(cfg,case)
            clock=policy.goal_time(cap,0,max_goals=cfg['max_goals'][name])
            payload=policy.payload('pi0',public,menu,[],None,clock)
            from .providers import validate_choice_payload
            validate_choice_payload(payload,'d3')
            rows.append({'case':case,'initial_sha256':digest(public),'actions':menu,
                         'action_descriptions_sha256':digest(public['action_descriptions']),
                         'native_installation_sha256':digest(actual_installation),
                         'decision_cap':cap,
                         'goal_slots':policy.goal_slots(cap,cfg['max_goals'][name]),
                         'action_menu_mode':'dynamic_native_admissible' if name=='alfworld' else 'static'})
        finally:
            if game is not None:game.close()
    manifest={'environment':name,'source_identity':native_identity(name),'actual_installation':installation,'cases':rows}
    immutable_json(Path(cfg['output'])/'manifests'/f'{name}.json',manifest)
    print('PREPARED',name,len(rows),'native cases; no model requests',flush=True)


def native_command(cfg,command,name=None,case_index=None,methods=None,endpoint=None,kev_endpoint=None):
    args=[native_python(cfg,name),'-m','blade',command,
          '--config',cfg['_config_path'],'--environment',name]
    if case_index is not None:args+=['--case-index',str(case_index)]
    if methods:args+=['--methods',*methods]
    if endpoint:args+=['--d3-endpoint',endpoint]
    if kev_endpoint:args+=['--kev-endpoint',kev_endpoint]
    return args


def kev_pool(cfg):
    return cfg['providers']['kev'].get('endpoints') or [cfg['providers']['kev']['endpoint']]


def prepare(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    for name in cfg['environments']:
        subprocess.run(native_command(cfg,'_prepare-env',name),cwd=ROOT,
                       env=child_environment(cfg),check=True)
    manifests={name:json.loads((out/'manifests'/f'{name}.json').read_text()) for name in cfg['environments']}
    freeze={'protocol_hash':cfg['protocol_hash'],'config':scientific_config(cfg),
            'sources':source_identity(),'manifest_sha256':{n:digest(m) for n,m in manifests.items()},
            'case_count_total':sum(len(m['cases']) for m in manifests.values()),
            'case_count_by_method':{method:sum(len(m['cases']) for n,m in manifests.items()
                                             if method in policy.methods_for_environment(cfg,n))
                                    for method in dict.fromkeys(method for name in manifests
                                        for method in policy.methods_for_environment(cfg,name))},
            'methods_by_environment':{n:policy.methods_for_environment(cfg,n) for n in manifests},
            'main_episode_count':sum(len(m['cases'])*sum(method in policy.MAIN_METHODS
                                   for method in policy.methods_for_environment(cfg,n))
                                   for n,m in manifests.items()),
            'episode_count':sum(len(m['cases'])*len(policy.methods_for_environment(cfg,n)) for n,m in manifests.items()),
            'paid_requests_at_preparation':0}
    immutable_json(out/'freeze.json',freeze)
    print('FROZEN',freeze['episode_count'],'episodes',flush=True)


def check_freeze(cfg):
    out=Path(cfg['output']);frozen=json.loads((out/'freeze.json').read_text())
    if (frozen['protocol_hash']!=cfg['protocol_hash'] or frozen['sources']!=source_identity()
            or frozen['config']!=scientific_config(cfg)):
        raise Halt('Scientific configuration or source changed; use a new output directory')
    for name,h in frozen['manifest_sha256'].items():
        if digest(json.loads((out/'manifests'/f'{name}.json').read_text()))!=h:
            raise Halt('Frozen case manifest changed')
    if (out/'STOP').exists():raise Halt('Retained interface interruption; inspect STOP before recovery')
    return frozen


def endpoint_pool(cfg):
    values=cfg['providers']['d3'].get('endpoints') or [cfg['providers']['d3']['endpoint']]
    if not isinstance(values,list) or not values or len(set(values))!=len(values):
        raise ValueError('D3 endpoints must be a distinct nonempty configured list')
    for endpoint in values:
        local=deepcopy(cfg);local['providers']['d3']['endpoint']=endpoint
        from .providers import settings_for
        settings_for(local,'d3')
    return values


def episode_paths(out,eid):
    stem=digest(eid)
    return (Path(out)/'episodes'/f'{stem}.json',Path(out)/'episodes'/f'{stem}.jsonl',
            Path(out)/'goals'/f'{stem}.jsonl')


def execute_episode(cfg,name,row,method,ledger,game_factory=None):
    """Only actual game.step(action) advances the native environment.

    No engine peek, predicted transition, goal checker or state-legal mask is
    available to this loop. Metrics are read only for saved evaluation records.
    """
    if game_factory is None:
        from .environment import Game
        game_factory=Game
    case=row['case'];cap=policy.effective_decision_cap(cfg,case);max_goals=cfg['max_goals'][name]
    if name!=case['environment'] or method not in policy.methods_for_environment(cfg,name):
        raise Halt('Episode environment or method identity differs')
    if type(row.get('decision_cap')) is not int or row['decision_cap']!=cap:
        raise Halt('Frozen decision cap differs from scientific configuration')
    slots=policy.goal_slots(cap,max_goals)
    if row.get('goal_slots')!=slots or any(type(p) is not int for p in row['goal_slots']):
        raise Halt('Frozen goal schedule differs from effective decision cap')
    eid=name+'|'+case['id']+'|'+method
    summary_path,step_path,goal_path=episode_paths(cfg['output'],eid)
    old=json.loads(summary_path.read_text()) if summary_path.exists() else None
    identity={'id':eid,'environment':name,'case':case,'method':method,
              'decision_cap':cap,'goal_slots':slots,
              'episode_identity_sha256':digest({'environment':name,'row':row,'method':method})}
    if old and any(old.get(key)!=value for key,value in identity.items()):
        raise Halt('Retained episode identity, decision cap or goal schedule differs')
    if old and old['status']=='failed':return 'failed'
    # Validate all retained boundaries before replay or any model/goal request,
    # including completed summaries. A cap change cannot legitimize an old run.
    retained_count=0;retained_done=False
    for record in iter_records(step_path):
        if type(record.get('step')) is not int or record['step']!=retained_count or retained_done:
            raise Halt('Retained trajectory sequence or native terminal boundary differs')
        retained_count+=1
        if retained_count>cap:raise Halt('Retained trajectory exceeds frozen decision cap')
        retained_done=bool(record['native_result']['done'])
    points=slots if policy.needs_goal(method) else []
    goals=read_records(goal_path)
    required=[p for p in points if p<retained_count]
    actual=[g.get('step') for g in goals]
    allowed=[required]
    if not retained_done and retained_count in points:allowed.append(required+[retained_count])
    if actual not in allowed or any(type(g.get('step')) is not int or type(g.get('slot')) is not int
                                  or g['slot']!=points.index(g['step']) for g in goals):
        raise Halt('Retained goals differ from frozen schedule or trajectory boundary')
    if old and old['status']=='complete':
        if (old.get('decisions')!=retained_count or old.get('steps')!=retained_count
                or not (retained_done or retained_count==cap)
                or old.get('native_done')!=retained_done
                or old.get('decision_cap_reached')!=(not retained_done and retained_count==cap)
                or old.get('termination_reason')!=('native_done' if retained_done else 'decision_cap')):
            raise Halt('Completed episode boundary differs from retained trajectory')
        return
    game=None;started=time.time();attempt_started=time.perf_counter();attempt_status='interrupted'
    summary={**identity,'status':'running','started':old['started'] if old else started}
    write_json(summary_path,summary)
    try:
        game=game_factory(name,case)
        if hasattr(game,'identity') and digest(game.identity())!=row['native_installation_sha256']:
            raise Halt('Native source/library identity differs from frozen installation')
        if digest(game.observe())!=row['initial_sha256'] or game.actions()!=row['actions']:
            raise Halt('Paired native reset or full menu identity changed')
        records=deque(maxlen=4);record_count=0
        for record in iter_records(step_path):
            if record['step']!=record_count:raise Halt('Retained trajectory sequence differs')
            if digest(game.observe())!=record['before_sha256']:raise Halt('Replay differs before native action')
            result=game.step(record['action'])
            if digest(game.observe())!=record['after_sha256'] or result!=record['native_result']:
                raise Halt('Replay differs after native action; no resampling')
            records.append(record);record_count+=1
        replayed_count=record_count
        done=bool(records and records[-1]['native_result']['done'])
        while not done and record_count<cap:
            if (Path(cfg['output'])/'STOP').exists():raise Halt('Another worker retained an interruption')
            if (Path(cfg['output'])/'PAUSE').exists():raise BudgetHalt('Another worker retained a budget pause')
            step=record_count;before=game.observe();menu=game.actions()
            mode=row.get('action_menu_mode','static')
            if mode=='static':
                if menu!=row['actions'] or digest(before['action_descriptions'])!=row['action_descriptions_sha256']:
                    raise Halt('Native full menu/description changed; no filtering allowed')
            elif mode=='dynamic_native_admissible' and name=='alfworld':
                if not cfg.get('alfworld',{}).get('native_menu_authorized'):
                    raise Halt('Official ALFWorld menu use requires explicit protocol authorization')
                # The adapter returns the complete official menu; no further filter.
                policy._criteria(before,menu)
            else:raise Halt('Unrecognized frozen action menu mode')
            history=list(records);at=max((p for p in points if p<=step),default=None)
            # H4 and H4+goal share the mechanical slot countdown. H4 does not
            # generate or receive a goal; this avoids changing a second input.
            clock_at=max((p for p in slots if p<=step),default=None) if not policy.is_baseline(method) else None
            clock=policy.goal_time(cap,step,clock_at,max_goals);begin=time.perf_counter()
            active=next((g for g in goals if g['step']==at),None)
            if step in points and active is None:
                request=policy.goal_payload(before,menu,history,clock,cfg['providers']['gpt'])
                rid,raw=ledger.call('gpt',request,eid,step,'goal')
                active={'step':step,'slot':points.index(step),'request_id':rid,'value':extract_goal(raw)}
                append(goal_path,active);goals.append(active)
            if at is not None and active is None:raise Halt('Scheduled goal missing; no fallback')
            goal=active['value'] if active else None;distributions={};dependencies=[]
            kind=policy.actor_kind(method)
            for role in policy.roles_for(method,step):
                if (Path(cfg['output'])/'STOP').exists():raise Halt('Another worker retained an interruption')
                if (Path(cfg['output'])/'PAUSE').exists():raise BudgetHalt('Another worker retained a budget pause')
                request=policy.payload(role,before,menu,history,goal,clock,cfg['providers'][kind]['model'])
                rid,raw=ledger.call(kind,request,eid,step,role)
                distributions[role]=parse_distribution(raw,request)
                dependencies.append({'role':role,'request_id':rid})
            scores=None
            if policy.is_baseline(method):
                combined=policy.normalize_distribution(menu,distributions['pi0'])
            elif method.endswith('-history-h4') or step<8:
                combined=policy.smooth_distribution(menu,distributions['ph'])
            elif method.endswith('-joint-h4'):
                combined=policy.normalize_distribution(menu,distributions['joint'])
            else:
                scores=policy.log_action_scores(method,menu,ph=distributions['ph'],q=distributions['q'],p0=distributions['pi0'])
                combined=policy.combine_distributions(method,menu,ph=distributions['ph'],q=distributions['q'],p0=distributions['pi0'])
            action=policy.choose_action(menu,combined)
            policy_wall_seconds=time.perf_counter()-begin
            # Sole real transition. Failed in-game actions remain in history.
            native_begin=time.perf_counter();result=game.step(action);after=game.observe()
            record={'step':step,'action':action,'before_sha256':digest(before),'after_sha256':digest(after),
                    'after':policy.public_view(after),'native_result':result,'dependencies':dependencies,
                    'action_menu':menu,'action_menu_mode':mode,
                    'goal_request_id':active['request_id'] if active else None,
                    'goal':goal,'distributions':distributions,'log_action_scores':scores,
                    'selected_distribution':combined,'metrics':game.metrics(),
                    'policy_wall_seconds':policy_wall_seconds,
                    'native_step_seconds':time.perf_counter()-native_begin,
                    'decision_wall_seconds':time.perf_counter()-begin}
            append(step_path,record);records.append(record);record_count+=1;done=bool(result['done'])
            if record_count%25==0:
                write_json(summary_path,{**summary,'steps':record_count,'decisions':record_count,
                                         'metrics':game.metrics(),'updated':time.time()})
                print('RUNNING',name,method,'steps',record_count,flush=True)
        cap_reached=not done and record_count==cap
        metrics={**game.metrics(),'decisions':record_count,'decision_cap':cap,
                 'decision_cap_reached':cap_reached}
        write_json(summary_path,{**summary,'status':'complete','steps':record_count,
                                'decisions':record_count,'native_steps':metrics['native_steps'],
                                'native_done':done,'decision_cap_reached':cap_reached,
                                'termination_reason':'native_done' if done else 'decision_cap',
                                'metrics':metrics,
                                'goals':len(goals),'finished':time.time(),'resumed_steps':replayed_count,
                                'episode_wall_seconds':sum(r.get('decision_wall_seconds',0) for r in iter_records(step_path)),
                                'attempt_wall_seconds':time.perf_counter()-attempt_started,
                                'elapsed_since_first_start_seconds':time.time()-summary['started']})
        attempt_status='complete'
        print('COMPLETE',name,method,case['id'],'steps',record_count,flush=True)
    except BudgetHalt:
        attempt_status='budget_paused'
        write_json(summary_path,{**summary,**retained_progress(step_path),
            'status':'budget_paused','attempt_wall_seconds':time.perf_counter()-attempt_started,
            'updated':time.time()});raise
    except Exception as exc:
        message=redact(type(exc).__name__+': '+str(exc))
        if hasattr(exc,'record'):write_json(Path(cfg['output'])/'failures'/f'{digest(eid)}.json',exc.record)
        attempt_status='failed'
        write_json(summary_path,{**summary,**retained_progress(step_path),
            'status':'failed','error':message,'attempt_wall_seconds':time.perf_counter()-attempt_started,
            'finished':time.time()})
        if isinstance(exc,(ProviderFailure,ProviderError)):
            print('FAILED',name,method,case['id'],message,flush=True)
            return 'failed'
        write_json(Path(cfg['output'])/'STOP',{'episode':eid,'error':message,'no_retry':True,'at':time.time()})
        raise
    finally:
        append(Path(cfg['output'])/'episode-attempts'/f'{digest(eid)}.jsonl',
            {'episode':eid,'status':attempt_status,'started':started,'finished':time.time(),
             'attempt_wall_seconds':time.perf_counter()-attempt_started,
             'retained_completed_decisions':sum(1 for _ in iter_records(step_path))})
        if game is not None:game.close()


def retained_progress(step_path):
    """Report completed retained actions, never invent an interrupted action."""
    rows=read_records(step_path)
    known=[r['decision_wall_seconds'] for r in rows if type(r.get('decision_wall_seconds')) in (int,float)]
    metrics=rows[-1].get('metrics') if rows else None
    return {'steps':len(rows),'decisions':len(rows),'metrics':metrics,
            'native_steps':metrics.get('native_steps') if isinstance(metrics,dict) else None,
            'episode_wall_seconds':sum(known) if len(known)==len(rows) else None,
            'retained_decision_wall_seconds_known_subtotal':sum(known),
            'retained_decision_wall_seconds_unknown_actions':len(rows)-len(known)}


def run_native(cfg,name,index,selected,endpoint,kev_endpoint=None):
    check_freeze(cfg);out=Path(cfg['output'])
    manifest=json.loads((out/'manifests'/f'{name}.json').read_text())
    if native_identity(name)!=manifest['source_identity']:raise Halt('Native runtime identity changed')
    if endpoint not in endpoint_pool(cfg):raise Halt('D3 endpoint outside declared pool')
    local=deepcopy(cfg);local['providers']['d3']['endpoint']=endpoint
    ledger=Ledger(out,local);failed=0
    try:
        if any(policy.actor_kind(m)=='kev' for m in selected):
            freeze_kev_runtime(local,ledger.providers)
        for method in selected:
            if method in policy.methods_for_environment(cfg,name):
                failed+=execute_episode(local,name,manifest['cases'][index],method,ledger)=='failed'
    finally:ledger.close()
    return failed


def freeze_kev_runtime(cfg,client):
    """Retain the actual inference identity across workers and later resumes."""
    identities={endpoint:value['runtime_identity'] for endpoint,value in client.kev_health_pool().items()}
    immutable_json(Path(cfg['output'])/'runtime-identities/kev.json',identities)
    return identities


def preload_credentials(cfg,selected):
    required={'JEV_KEYS' for m in selected if policy.actor_kind(m)=='jev'}
    if any(policy.needs_goal(m) for m in selected):required.add('OPENAI_API_KEY')
    p=cfg.get('credentials_file')
    if p:
        path=Path(p)
        if not stat.S_ISREG(path.lstat().st_mode):raise Halt('Private credentials must be a regular file')
        mode=stat.S_IMODE(path.stat().st_mode)
        if mode&0o077 or path.stat().st_uid!=os.getuid():raise Halt('Private credentials must be owned by this user and mode600')
        values=json.loads(path.read_text())
        for key in ['OPENAI_API_KEY','JEV_KEYS','D3_API_KEY','KEV_API_KEY']:
            if key in values and key not in os.environ:os.environ[key]=values[key]
    missing=[key for key in required if not os.environ.get(key)]
    if missing:raise Halt('Set '+', '.join(sorted(missing))+' in this terminal, or configure a private credentials_file')
    for key in required:
        if any(c.isspace() for c in os.environ[key]):raise Halt('Credential contains whitespace; enter only the key')
    client=Providers(cfg)
    for kind,key in [('jev','JEV_KEYS'),('gpt','OPENAI_API_KEY')]:
        if key in required:client._credential(kind)


def _run_locked(cfg,selected):
    from .plans import matches_plan
    try:frozen=matches_plan(cfg)
    except ValueError:frozen=False
    if not frozen:raise Halt('Real dispatch requires a configuration that equals a frozen plan')
    check_freeze(cfg)
    if len(set(selected))!=len(selected):raise Halt('Duplicate method selection')
    allowed={method for name in cfg['environments'] for method in policy.methods_for_environment(cfg,name)}
    if not selected or any(method not in allowed for method in selected):
        raise Halt('Selected methods fall outside the plan of this configuration')
    required={'jev' for m in selected if policy.actor_kind(m)=='jev'}
    if any(policy.needs_goal(m) for m in selected):required.add('gpt')
    for kind in required:
        value=cfg['budgets'][kind+'_usd']
        if value is None or value<=0:raise BudgetHalt('Set an explicit positive '+kind+'_usd budget before execution')
    preload_credentials(cfg,selected)
    # Check every configured D3 replica before any paid Jev or teacher request.
    pool=endpoint_pool(cfg);runtime_checks={}
    if any(policy.actor_kind(m)=='d3' for m in selected):
        for endpoint in pool:
            local=deepcopy(cfg);local['providers']['d3']['endpoint']=endpoint
            runtime_checks.setdefault('d3',{})[endpoint]=Providers(local).health()['runtime_identity']
    if any(policy.actor_kind(m)=='kev' for m in selected):
        runtime_checks['kev']=freeze_kev_runtime(cfg,Providers(cfg))
    out=Path(cfg['output']);jobs=[]
    # Complete schema creation before processes first open the shared WAL.
    initial_ledger=Ledger(out,cfg)
    initial_ledger.close()
    invocation_started=time.perf_counter()
    # D3 replicas bound only D3 concurrency; API-only Jev jobs use every requested worker.
    groups={}
    for method in selected:groups.setdefault(policy.actor_kind(method),[]).append(method)
    group_slots={kind:(min(cfg['workers'],len(pool)) if kind=='d3' else cfg['workers']) for kind in groups}
    append(out/'execution-events.jsonl',{'event':'run_start','unix_time':time.time(),
        'selected_methods':selected,'workers_requested':cfg['workers'],
        'workers_effective':group_slots,'verified_runtime_identities':runtime_checks})
    # A new explicit invocation may resume an operational budget pause.
    (out/'PAUSE').unlink(missing_ok=True)
    report_lock=threading.Lock();last_report=[time.monotonic()];streak=[0]
    def run_group(methods,slots):
        jobs=[]
        for name in cfg['environments']:
            if not any(method in policy.methods_for_environment(cfg,name) for method in methods):continue
            manifest=json.loads((out/'manifests'/f'{name}.json').read_text())
            jobs.extend((name,i) for i in range(len(manifest['cases'])))
        endpoint_queue=queue.Queue()
        for i in range(slots):endpoint_queue.put(pool[i%len(pool)])
        def work(job):
            endpoint=endpoint_queue.get()
            try:
                name,index=job
                env=child_environment(cfg);env['BLADE_EXECUTE']='1'
                result=subprocess.run(native_command(cfg,'_run-env',name,index,methods,endpoint),
                        cwd=ROOT,env=env)
                if result.returncode==3:
                    write_json(out/'PAUSE',{'reason':'worker budget pause','at':time.time()})
                    raise BudgetHalt('Retained budget pause; inspect spend before approving more budget')
                with report_lock:
                    streak[0]=streak[0]+1 if result.returncode==4 else 0
                    if streak[0]>=5:
                        write_json(out/'STOP',{'reason':'5 consecutive jobs with provider failures','at':time.time()})
                        raise Halt('Five consecutive jobs had provider failures; inspect provider health')
                if result.returncode not in (0,4):
                    if not (out/'STOP').exists():write_json(out/'STOP',{'reason':'native worker process exited','exit_code':result.returncode,'at':time.time()})
                    raise Halt('Retained native worker interruption; inspect output status')
            finally:endpoint_queue.put(endpoint)
        # Submit only bounded work; a failure prevents dispatch of the remaining cohort.
        iterator=iter(jobs)
        with ThreadPoolExecutor(max_workers=slots) as executor:
            active={}
            for _ in range(slots):
                job=next(iterator,None)
                if job is not None:active[executor.submit(work,job)]=job
            while active:
                completed,_=wait(active,return_when=FIRST_COMPLETED)
                # Inspect all returned results before submitting any further case.
                for future in completed:future.result()
                for future in completed:active.pop(future)
                with report_lock:
                    if time.monotonic()-last_report[0]>=300:
                        from .report import report
                        report(cfg,bootstrap=False,include_timing=False)
                        last_report[0]=time.monotonic()
                for _ in completed:
                    job=next(iterator,None)
                    if job is not None:active[executor.submit(work,job)]=job
    # Actor groups run concurrently; a failure in one stops dispatch in the other via STOP/PAUSE.
    with ThreadPoolExecutor(max_workers=len(groups)) as group_executor:
        futures=[group_executor.submit(run_group,methods,group_slots[kind]) for kind,methods in groups.items()]
        for future in futures:future.result()
    from .report import report
    report(cfg)
    append(out/'execution-events.jsonl',{'event':'run_stage_complete','unix_time':time.time(),
        'selected_methods':selected,'stage_wall_seconds':time.perf_counter()-invocation_started})


def run(cfg,selected):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'coordinator.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise Halt('Another coordinator owns this output; do not run duplicate episodes') from None
        try:_run_locked(cfg,selected)
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def save_credentials(path):
    path=Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT):raise Halt('Credential file must be outside the repository')
    values={}
    for label,key in [('OpenAI GPT','OPENAI_API_KEY'),('Jev','JEV_KEYS')]:
        value=getpass.getpass(label+' API key (hidden): ')
        if not value or any(c.isspace() for c in value):raise Halt('Enter only a nonempty key; no commands or spaces')
        values[key]=value
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as f:f.write(dumps(values)+'\n')
    print('Private credentials saved outside the repository; configure credentials_file:',path)


def initialize(path,plan):
    """Create a local configuration from a plan template; never overwrite one."""
    path=Path(path).expanduser().absolute()
    if path.exists():
        print('Existing configuration retained:',path);return
    cfg=json.loads((HERE/'configs'/(plan+'.json')).read_text())
    cfg['runtime_root']=os.path.relpath(HERE/'.runtime',path.parent)
    cfg['alfworld']['data_root']=os.path.relpath(HERE/'.runtime/upstream/alfworld-data',path.parent)
    cfg['output']=os.path.relpath(ROOT/'outputs'/plan,path.parent)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(cfg,indent=2)+'\n')
    print('Created',plan,'plan configuration with null budgets:',path)
    print('Nothing was installed, downloaded, requested or started.')


def main(argv=None):
    parser=argparse.ArgumentParser(prog='python -m blade',description=__doc__)
    parser.add_argument('command',choices=['init','doctor','prepare','run','report','credentials','_prepare-env','_run-env'])
    parser.add_argument('--config',type=Path,default=ROOT/'config.local.json')
    parser.add_argument('--plan',choices=['main','ablation'],default='main',help='Template used by init')
    parser.add_argument('--execute',action='store_true',help='Explicitly enable model inference')
    parser.add_argument('--methods',nargs='+',choices=policy.METHODS)
    parser.add_argument('--environment',choices=['babyai','minihack','babaisai','alfworld'])
    parser.add_argument('--case-index',type=int)
    parser.add_argument('--d3-endpoint')
    parser.add_argument('--kev-endpoint')
    parser.add_argument('--path',type=Path,default=Path.home()/'.config/blade/credentials.json')
    args=parser.parse_args(argv)
    try:
        if args.command=='init':initialize(args.config,args.plan);return
        if args.command=='credentials':save_credentials(args.path);return
        cfg=load_config(args.config)
        if args.command=='prepare':prepare(cfg)
        elif args.command=='doctor':
            from .plans import planned_counts
            print(json.dumps(planned_counts(cfg),indent=2))
            for name in cfg['environments']:
                print(name,'native Python:',native_python(cfg,name),'exists:',Path(native_python(cfg,name)).is_file())
            print('D3 endpoints:',endpoint_pool(cfg));print('Kev endpoints:',cfg['providers']['kev']['endpoints']);print('Budget configuration:',cfg['budgets'])
            print('Doctor is offline; no credentials read or model requests sent.')
        elif args.command=='_prepare-env':prepare_native(cfg,args.environment)
        elif args.command=='run':
            if not args.execute:raise Halt('Use --execute only after reviewing the frozen configuration and explicit budgets')
            from .plans import selected_methods
            run(cfg,args.methods or selected_methods(cfg,'all'))
        elif args.command=='_run-env':
            # Parent explicitly opted into execution; internal action is never a public substitute.
            if os.environ.get('BLADE_EXECUTE')!='1':raise Halt('Internal worker requires parent execution marker')
            if run_native(cfg,args.environment,args.case_index,args.methods,args.d3_endpoint,args.kev_endpoint):sys.exit(4)
        else:
            from .report import report
            report(cfg)
    except BudgetHalt as exc:
        print('BUDGET PAUSED:',redact(str(exc)),file=sys.stderr);sys.exit(3)
    except (Halt,ProviderError,ValueError,OSError,subprocess.CalledProcessError) as exc:
        print('HALTED:',redact(str(exc)),file=sys.stderr)
        sys.exit(2)


if __name__=='__main__':main()
