"""Prepare/verify the released Kev-27B v2 and serve its original decision head.

Nothing is downloaded, imported from Kev, or loaded on a GPU on module import.
Only the explicit ``prepare --download`` command downloads the ~51 GB snapshot.
``serve`` requires verified local weights and the exact, clean upstream checkout.
The serving wrapper adds provenance to /v1/models; it does not change decisions.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

REPO = "jaredpalmer/kev-27b"
REVISION = "28be62e9c5ae0471bda2b7c636a55224b4f4b887"
SOURCE_REPO = "https://github.com/jaredpalmer/kev"
SOURCE_REVISION = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"
HEAD_SHA256 = "7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad"
MODEL_NAME = "kev-latest"
FORMAT_ID = "kev-systemone-native-v2"
TEMPERATURE = 1.32  # rounded published temperature; actual head value is retained.
MAX_STATE_TOKENS = 65536
MAX_ROW_TOKENS = 73728
MANIFEST_NAME = "meta_jev_kev_manifest.json"
IDENTITY_NAME = "meta_jev_kev_verified_identity.json"
_REQUIRED_SMALL = {"config.json", "model.safetensors.index.json", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja", "head.pt"}
_SHARD = re.compile(r"model-\d{5}-of-00011\.safetensors")
_SHARD_NAMES = {f"model-{i:05d}-of-00011.safetensors" for i in range(1, 12)}


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest(path, *, git_blob=False):
    size = path.stat().st_size
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_file(root, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Kev snapshot file must be a simple filename")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Kev snapshot file missing or outside model directory: " + name)
    return path


def prepare(root, *, download=False):
    """Explicit public download; expected file digests come from immutable Hub metadata.

    This function has no implicit authentication and never downloads a base model:
    the requested snapshot contains the full trained backbone, head and tokenizer.
    """
    if not download:
        raise ValueError("prepare requires --download; use verify for local-only checks")
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Hub did not resolve the frozen Kev revision")
    expected = {}
    for item in info.siblings:
        name = item.rfilename
        if name not in _REQUIRED_SMALL and not _SHARD.fullmatch(name):
            continue
        lfs = item.lfs
        expected[name] = {"bytes": item.size,
                          "sha256": getattr(lfs, "sha256", None) if lfs else None,
                          "git_blob_sha1": item.blob_id if not lfs else None}
    if set(expected) != _REQUIRED_SMALL | _SHARD_NAMES:
        raise ValueError("Frozen Kev snapshot layout differs")
    if expected["head.pt"]["sha256"] != HEAD_SHA256:
        raise ValueError("Published Kev head digest differs")
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / MANIFEST_NAME).exists():
        previous = json.loads((root / MANIFEST_NAME).read_text())
        if previous.get("files") != expected or previous.get("revision") != REVISION:
            raise ValueError("Existing Kev manifest differs; use an independent model directory")
    snapshot_download(REPO, revision=REVISION, local_dir=str(root),
                      allow_patterns=sorted(expected), token=False, max_workers=4)
    manifest = {"repo_id": REPO, "revision": REVISION, "head_sha256": HEAD_SHA256,
                "source_revision": SOURCE_REVISION,
                "provenance": "immutable_public_huggingface_file_metadata", "files": expected}
    (root / MANIFEST_NAME).write_text(_json(manifest))
    identity = verify(root)
    (root / IDENTITY_NAME).write_text(_json(identity))
    return identity


def verify(root):
    """Read and hash every declared file; never contact a network or import torch."""
    root = Path(root).expanduser().resolve()
    manifest_path = _safe_file(root, MANIFEST_NAME)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("repo_id") != REPO or manifest.get("revision") != REVISION
            or manifest.get("source_revision") != SOURCE_REVISION
            or manifest.get("head_sha256") != HEAD_SHA256
            or manifest.get("provenance") != "immutable_public_huggingface_file_metadata"):
        raise ValueError("Frozen Kev manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _REQUIRED_SMALL | _SHARD_NAMES:
        raise ValueError("Kev manifest must contain the full eleven-shard snapshot")
    for name, entry in sorted(files.items()):
        path = _safe_file(root, name)
        if not isinstance(entry, dict) or type(entry.get("bytes")) is not int or path.stat().st_size != entry["bytes"]:
            raise ValueError("Kev snapshot size differs: " + name)
        sha256, git_sha1 = entry.get("sha256"), entry.get("git_blob_sha1")
        if sha256:
            if not re.fullmatch(r"[0-9a-f]{64}", sha256) or _digest(path) != sha256:
                raise ValueError("Kev snapshot SHA256 differs: " + name)
        elif git_sha1:
            if not re.fullmatch(r"[0-9a-f]{40}", git_sha1) or _digest(path, git_blob=True) != git_sha1:
                raise ValueError("Kev snapshot Git-blob hash differs: " + name)
        else:
            raise ValueError("Kev snapshot expected digest missing: " + name)
    if _digest(root / "head.pt") != HEAD_SHA256:
        raise ValueError("Released Kev pointer head differs")
    cfg = json.loads((root / "config.json").read_text())
    if (cfg.get("dtype") or cfg.get("torch_dtype")) != "bfloat16":
        raise ValueError("Released full-weight Kev snapshot must remain BF16")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if set(index.get("weight_map", {}).values()) != {n for n in files if _SHARD.fullmatch(n)}:
        raise ValueError("Kev checkpoint index does not cover exactly all frozen shards")
    if (root / "adapter_config.json").exists():
        raise ValueError("Kev v2 is a full-weight checkpoint, not the v1 adapter")
    return {"repo_id": REPO, "revision": REVISION, "source_revision": SOURCE_REVISION,
            "head_sha256": HEAD_SHA256, "format_id": FORMAT_ID,
            "checkpoint_manifest_sha256": _digest(manifest_path),
            "verification": "all_manifest_files_verified", "weights": "full",
            "dtype": "bfloat16", "temperature_published": TEMPERATURE,
            "max_state_tokens": MAX_STATE_TOKENS, "max_row_tokens": MAX_ROW_TOKENS,
            "max_options": 255, "date_facts": False, "truncate_states": False}


def verify_source(root):
    root = Path(root).expanduser().resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != SOURCE_REVISION:
        raise ValueError("Exact frozen upstream Kev code revision required")
    # A local source edit (including an untracked module shadowing an import) must
    # not silently masquerade as the published inference implementation.
    if git("status", "--porcelain", "--untracked-files=all", "--", "kev", "pyproject.toml", "uv.lock"):
        raise ValueError("Upstream Kev inference source must be a clean checkout")
    if not (root / "kev" / "serve.py").is_file():
        raise ValueError("Upstream Kev serving source missing")
    return root


def _serve_options():
    """Freeze inference settings without parsing game state or modifying prompts."""
    from kev.checkpoint import LoadOptions
    import torch
    # Reference eager BF16 path, native saved head calibration; avoid package-
    # dependent optional fused kernels and graph padding changing numerics.
    return LoadOptions(dtype=torch.bfloat16, backend="torch", temperature=None,
                       attn="sdpa", cuda_graphs=False, fused=False)


def serve(model_dir, source_dir, *, port=8906):
    """Run verified released Kev on one explicitly exposed 80-GB-class GPU.

    Only /v1/models receives extra provenance. Native /v1/systemone and option
    encoding, probability rounding, context admission and pointer head are used
    unchanged. No candidate repair, date arithmetic or state truncation is used.
    """
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Use an unprivileged local Kev port")
    source = verify_source(source_dir)
    identity = verify(model_dir)
    # Fail before importing/loading the model if a service already owns the port.
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    if os.environ.get("KEV_TEMPERATURE"):
        raise ValueError("KEV_TEMPERATURE must be unset; native fitted calibration is frozen")
    for key in ("KEV_DATE_FACTS", "KEV_TRUNCATE_STATES"):
        if os.environ.get(key) not in (None, "", "0"):
            raise ValueError(key + " must be unset; calibrated native head/no preprocessing required")
    os.environ["KEV_DATE_FACTS"] = "0"
    os.environ["KEV_TRUNCATE_STATES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one available 80-GB-class CUDA GPU; no CPU fallback")
    free, total = torch.cuda.mem_get_info(0)
    if total < 75 * 1024**3 or free < 68 * 1024**3:
        raise ValueError("Kev requires one >=75-GiB CUDA GPU with >=68 GiB free; do not evict other work")
    upstream = importlib.import_module("kev.serve")
    if Path(upstream.__file__).resolve() != source / "kev" / "serve.py":
        raise ValueError("Kev imported outside the verified source checkout")
    from kev.checkpoint import Checkpoint
    # Verify local tokenizer equals the frozen base copy but load from the local
    # release snapshot. The official loader obtains its tokenizer from the base;
    # prepare stores the release tokenizer so serving can be entirely offline.
    checkpoint = Checkpoint(str(Path(model_dir).resolve()))
    from kev import checkpoint as ck_module
    original_tokenizer_loader = ck_module.load_tokenizer
    def local_tokenizer_loader(name, revision=None):
        if name != checkpoint.meta.base or revision != checkpoint.meta.base_revision:
            raise ValueError("Unexpected upstream tokenizer source identity")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(Path(model_dir).resolve()), local_files_only=True)
    ck_module.load_tokenizer = local_tokenizer_loader
    try:
        tok, model = checkpoint.load("cuda", _serve_options())
    finally:
        ck_module.load_tokenizer = original_tokenizer_loader
    if (round(float(model.head.temperature), 2) != TEMPERATURE or model.dtype != "bfloat16"
            or model.backend != "torch" or checkpoint.meta.weights != "full"):
        raise ValueError("Released Kev inference identity changed")
    identity.update(temperature=float(model.head.temperature), backend="torch", device="cuda",
                    cuda_graphs=False, fused=False,
                    gpu_name=torch.cuda.get_device_name(0),
                    compute_capability=list(torch.cuda.get_device_capability(0)),
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "accelerate", "safetensors", "huggingface-hub")})
    for package in ("flash-linear-attention", "causal-conv1d", "triton"):
        try:
            identity['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            identity['versions'][package] = None
    upstream.app.state.server = upstream.Server(checkpoint, tok, model, "cuda", truncate_states=False)
    original_models = upstream.models
    def models_with_identity():
        response = original_models()
        response["runtime_identity"] = identity
        return response
    for route in upstream.app.routes:
        if getattr(route, "path", None) == "/v1/models":
            route.endpoint = models_with_identity
            route.dependant.call = models_with_identity
    print("KEV_READY " + json.dumps(identity, sort_keys=True), flush=True)
    import uvicorn
    uvicorn.run(upstream.app, host="127.0.0.1", port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "serve"):
        child = commands.add_parser(name)
        child.add_argument("--model-dir", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--download", action="store_true", help="Explicitly download ~51 GB public weights")
        if name == "serve":
            child.add_argument("--source-dir", type=Path, required=True)
            child.add_argument("--port", type=int, default=8906)
    args = parser.parse_args()
    if args.command == "prepare":
        print(_json(prepare(args.model_dir, download=args.download)), end="")
    elif args.command == "verify":
        print(_json(verify(args.model_dir)), end="")
    else:
        serve(args.model_dir, args.source_dir, port=args.port)


if __name__ == "__main__":
    main()
