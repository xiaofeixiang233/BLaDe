"""Fetch and SHA256-verify one immutable public D3 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

REPO='vllm-sr/d3'
REVISION='cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75'
MODEL_SHA='0e439bebbbbf80d9e1b601c56c55a6fc3a6502b4e396cbbb5f3c3f3acc1d607b'
MANIFEST_SHA='830ee38ac3386ef7db03cef240a155f36b5d82c08781718ef5bf6393c528f747'


def verify(root):
    root=Path(root).resolve();manifest_path=root/'MODEL_MANIFEST.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=MANIFEST_SHA:
        raise ValueError('Exact upstream model manifest SHA256 differs')
    manifest=json.loads(manifest_path.read_text())
    if manifest['identity']['model_sha256']!=MODEL_SHA:raise ValueError('Frozen checkpoint identity differs')
    checked={}
    for name,expected in sorted(manifest['files_sha256'].items()):
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Model manifest path invalid')
        if path.stat().st_size!=manifest['files_bytes'][name]:raise ValueError('Model file size differs: '+name)
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Model SHA256 mismatch: '+name)
        checked[name]=expected
    return {'repo_id':REPO,'revision':REVISION,'model_sha256':MODEL_SHA,
            'manifest_sha256':hashlib.sha256((root/'MODEL_MANIFEST.json').read_bytes()).hexdigest(),
            'verification':'all_manifest_files_full_sha256','files_sha256':checked}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();root=args.model_dir.resolve()
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO,revision=REVISION,local_dir=str(root),token=False,max_workers=4)
    receipt=verify(root)
    target=root/'meta_jev_verified_identity.json'
    target.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('Verified D327B checkpoint:',root,flush=True)


if __name__=='__main__':main()
