"""Start four independent two-GPU D3 servers in tmux, without API credentials."""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--gpu-pairs',nargs='+',default=['0,1','2,3','4,5','6,7'])
    p.add_argument('--ports',nargs='+',type=int,default=[8902,8903,8904,8905])
    p.add_argument('--session-prefix',default='blade-d3')
    p.add_argument('--log-dir',type=Path,default=Path(__file__).parent/'outputs/servers')
    args=p.parse_args()
    if not shutil.which('tmux'):p.error('Install tmux first')
    if len(args.gpu_pairs)!=len(args.ports) or len(set(args.ports))!=len(args.ports):p.error('One distinct port per GPU pair required')
    devices=[]
    for pair in args.gpu_pairs:
        if not re.fullmatch(r'\d+,\d+',pair):p.error('Each GPU pair must contain two numeric indices')
        devices.extend(pair.split(','))
    if len(set(devices))!=len(devices):p.error('GPU pairs must not overlap')
    if not re.fullmatch('[A-Za-z0-9_-]{1,40}',args.session_prefix):p.error('Invalid session prefix')
    python=Path(os.path.abspath(args.python.expanduser()));model=args.model_dir.resolve();logs=args.log_dir.resolve()
    if not python.is_file() or not (model/'MODEL_MANIFEST.json').is_file():p.error('Python or downloaded checkpoint missing')
    logs.mkdir(parents=True,exist_ok=True)
    names=[args.session_prefix+'-'+str(i) for i in range(len(args.ports))]
    for name in names:
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:
            p.error('Existing tmux session retained: '+name)
    root=Path(__file__).resolve().parent.parent
    for name,pair,port in zip(names,args.gpu_pairs,args.ports):
        command=['env','-u','OPENAI_API_KEY','-u','JEV_KEYS','-u','D3_API_KEY',
                 'CUDA_VISIBLE_DEVICES='+pair,str(python),'-m','blade.d3_server',
                 '--port',str(port),'--model-dir',str(model),'--identity-directory',str(logs)]
        shell='cd '+shlex.quote(str(root))+' && exec '+shlex.join(command)+' >'+shlex.quote(str(logs/(name+'.log')))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',name,shell],check=True)
        print('Started',name,'GPU pair',pair,'port',port)
    print('Loading is asynchronous. Check /health and server logs before running the cohort.')


if __name__=='__main__':main()
