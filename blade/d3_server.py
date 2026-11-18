"""Loopback-only text serving: official D3 numerics, explicit two-GPU placement.

The upstream loader hardcodes one device. A private factory changes only that
loader's device_map. The released checkpoint, tokenizer, attention hook,
readout and temperature are unchanged; no quantization or new decision head.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from .d3_model import REVISION, MODEL_SHA, verify
MAX_INPUT = 32768


def load_model(root):
    os.environ['D3_FAST'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import importlib.metadata
    if importlib.metadata.version('transformers') != '5.17.0':
        raise ValueError('D3 requires the independent pinned transformers5.17.0 runtime')
    if torch.cuda.device_count()!=2:
        raise ValueError('Expose exactly the two configured GPUs with CUDA_VISIBLE_DEVICES=0,1')
    for i in range(2):
        free,total = torch.cuda.mem_get_info(i)
        if free < 38*1024**3:
            raise ValueError('Configured GPU lacks38GiB free; do not evict another process')
    root = Path(root).resolve()
    identity = verify(root)
    if (identity['revision']!=REVISION or identity['repo_id']!='vllm-sr/d3'
            or identity['model_sha256']!=MODEL_SHA
            or identity['verification']!='all_manifest_files_full_sha256'):
        raise ValueError('Fully verified exact D3 snapshot is required')
    previous = list(sys.path)
    try:
        sys.path.insert(0,str(root))
        spec = importlib.util.spec_from_file_location('meta_jev_private_d3_runtime',root/'d3_runtime.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        original = module.backbone_class
        placement = {'visual':0,'language_model.embed_tokens':0,
            'language_model.rotary_emb':0,'language_model.norm':0}
        placement.update({'language_model.layers.'+str(i):0 if i<32 else 1 for i in range(64)})
        def backbone_class(kind):
            klass = original(kind)
            if kind!='qwen3_5':
                raise ValueError('Frozen D3 backbone type differs')
            class PlacementOnly:
                @staticmethod
                def from_pretrained(path,**kwargs):
                    if (Path(path)!=root or kwargs.get('device_map')!={'':'cuda:0'}
                            or kwargs.get('dtype')!=torch.bfloat16
                            or kwargs.get('attn_implementation')!='sdpa'):
                        raise ValueError('Original BF16/SDPA loader contract changed')
                    kwargs['device_map']=placement
                    return klass.from_pretrained(path,**kwargs)
            return PlacementOnly
        module.backbone_class=backbone_class
        model = module.D3.from_pretrained(root,device='cuda:0',batch_size=1,
            verify='fast',max_length=MAX_INPUT,model_name='d3',permutation_average=False)
    finally:
        sys.path[:] = previous
    if model.fast is not None or model.temperature!=1.0 or model.attention_mode!='noncausal_full_attention':
        raise ValueError('Frozen D3 decision numerics changed')
    devices = {str(p.device) for p in model.backbone.parameters()}
    if devices!={'cuda:0','cuda:1'}:
        raise ValueError('Model did not remain fully resident on the two GPUs')
    info = {'repo_id':'vllm-sr/d3','revision':REVISION,'model_sha256':MODEL_SHA,
        'format_id':'d3-code-readout-v1','max_input_tokens':MAX_INPUT,
        'checkpoint_declared_max_length':None,'dtype':'bfloat16','readout_dtype':'float32',
        'attention_mode':model.attention_mode,'temperature':model.temperature,
        'placement_adapter':'two_gpu_device_map_only_v1','device_map':placement,
        'fast_path':False,'permutation_average':False,'text_only':True,
        'versions':{p:importlib.metadata.version(p) for p in
            ('torch','transformers','accelerate','safetensors','huggingface-hub','flash-linear-attention')},
        'physical_devices':os.environ.get('CUDA_VISIBLE_DEVICES','all'),
        'server_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_manifest_sha256':identity['manifest_sha256'],'kernels':model.kernels}
    return model,info


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8902)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--identity-directory',type=Path)
    args=parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError('Use an unprivileged port')
    # Reserve our port before allocating a model. Never replace an existing service.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path!='/health':
                self.reply(404,{'error':'unknown_path'});return
            self.reply(200,{'status':'ready','model':'d3','authentication':False,
                'max_input_tokens':MAX_INPUT,'modalities':['text'],'runtime_identity':info})
        def do_POST(self):
            if self.path!='/v1/systemone':
                self.reply(404,{'error':'unknown_path'});return
            started=time.perf_counter()
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2*1024*1024:
                    raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size))
                if set(payload)!={'model','state','questions'} or payload['model']!='d3':
                    raise ValueError('Frozen text-only SystemOne request required')
                from .providers import validate_choice_payload
                validate_choice_payload(payload,'d3')
                for i in range(2):
                    model.torch.cuda.synchronize(i)
                result=model.system_one(state=payload['state'],questions=payload['questions'])
                for i in range(2):
                    model.torch.cuda.synchronize(i)
                result['runtime_identity']=info
                result['inference_seconds']=time.perf_counter()-started
                self.reply(200,result)
            except Exception as exc:
                self.reply(500,{'error':type(exc).__name__,'message':str(exc),
                    'runtime_identity':info,'inference_seconds':time.perf_counter()-started})
    server=HTTPServer(('127.0.0.1',args.port),Handler)
    try:
        model,info=load_model(args.model_dir)
        if args.identity_directory:
            args.identity_directory.mkdir(parents=True,exist_ok=True)
            (args.identity_directory/f'server.{args.port}.identity.json').write_text(json.dumps(info,sort_keys=True,indent=2)+'\n')
        print('D3_READY',json.dumps(info,sort_keys=True),flush=True)
        server.serve_forever()
    finally:
        server.server_close()


if __name__=='__main__':
    main()
