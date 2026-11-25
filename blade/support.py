"""Portable configuration, scientific identities and durable JSON artifacts."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def dumps(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError('Frozen artifact differs: '+str(path))
    else:write_json(path,value)


def source_identity():
    files=list(HERE.glob('*.py'))+list(HERE.glob('requirements-*.txt'))+[HERE/'upstream.lock.json']
    result={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.exists()}
    return result


def scientific_config(cfg):
    result=deepcopy(cfg)
    for key in ['budgets','workers','output','runtime_root','native_runtimes','credentials_file','credential_slots','_config_path','protocol_hash']:
        result.pop(key,None)
    result.get('alfworld',{}).pop('data_root',None)
    for provider in result['providers'].values():provider.pop('timeout_seconds',None)
    return result


def load_config(path):
    from .policy import METHODS
    from .providers import settings_for
    from .plans import matches_plan
    path=Path(path).resolve();cfg=json.loads(path.read_text())
    if cfg.get('methods')!=list(METHODS):raise ValueError('The complete v3 method registry is required')
    names=cfg.get('environments')
    if (not isinstance(names,list) or not names or len(set(names))!=len(names)
            or any(n not in ('babyai','minihack','babaisai','alfworld') for n in names)):
        raise ValueError('Select the five v3 benchmark names')
    if cfg.get('profile') not in ('paper','smoke'):raise ValueError('Use paper or smoke case profile')
    caps=cfg.get('decision_caps')
    if (not isinstance(caps,dict) or caps!={'crafter':2000}
            or any(type(value) is not int for value in caps.values())):
        raise ValueError('This protocol requires exactly 2000 outer action decisions for Crafter')
    if (cfg.get('history_length')!=4 or cfg.get('warmup_actions')!=8
            or cfg.get('epsilon')!=1e-6
            or cfg.get('max_goals')!={'babyai':4,'minihack':4,'babaisai':4,'crafter':8,'alfworld':4}):
        raise ValueError('Frozen history, warm-up, smoothing or goal count differs')
    # plan_for() rejects unknown versions; a plan cannot be widened in configuration.
    if not matches_plan(cfg) or cfg.get('ablation_actors')!=['jev','d3']:
        raise ValueError('Environments and execution_scope must equal the frozen plan named by version')
    alf=cfg.get('alfworld',{})
    if (alf.get('split')!='valid_unseen' or alf.get('expected_cases')!=134 or alf.get('horizon')!=50
            or alf.get('action_interface')!='native_admissible_commands' or alf.get('native_menu_authorized') is not True):
        raise ValueError('Use explicitly authorized, unfiltered native ALFWorld menus and the complete134-case split')
    if type(cfg.get('seed_start')) is not int or cfg['seed_start'] != 20271014:
        raise ValueError('Keep the main-experiment seed bank 20271014+i')
    if type(cfg.get('workers')) is not int or not 1<=cfg['workers']<=64:raise ValueError('workers must be1..64')
    for kind in ['jev','d3','kev','gpt']:settings_for(cfg,kind)
    budget=cfg.get('budgets',{})
    if set(budget)!={'jev_usd','gpt_usd'}:raise ValueError('Explicit per-provider budget fields required')
    for value in budget.values():
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
            raise ValueError('Budgets must be finite nonnegative numbers or null for preparation')
    for key in ['output','runtime_root']:
        value=cfg.get(key)
        if not isinstance(value,str) or not value:raise ValueError(key+' path required')
        p=Path(value).expanduser();cfg[key]=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    runtimes=cfg.get('native_runtimes',{})
    if not isinstance(runtimes,dict):raise ValueError('native_runtimes must be a mapping')
    # Keep a venv's executable path intact: resolving its python symlink to
    # the base interpreter discards the venv and its native dependencies.
    cfg['native_runtimes']={n:os.path.abspath(path.parent/Path(p).expanduser()) for n,p in runtimes.items()}
    p=Path(cfg['alfworld']['data_root']).expanduser()
    cfg['alfworld']['data_root']=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    if cfg.get('credentials_file'):
        p=Path(cfg['credentials_file']).expanduser()
        cfg['credentials_file']=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
        if Path(cfg['credentials_file']).is_relative_to(ROOT):
            raise ValueError('Private credential file must be outside the repository')
    cfg['_config_path']=str(path)
    cfg['protocol_hash']=digest({'config':scientific_config(cfg),'sources':source_identity()})
    return cfg


def child_environment(cfg):
    env=os.environ.copy()
    env.update(PYTHONHASHSEED='0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',
               TOKENIZERS_PARALLELISM='false',BLADE_RUNTIME_ROOT=cfg['runtime_root'])
    env['BLADE_ALFWORLD_DATA']=cfg['alfworld']['data_root']
    return env
