"""Install pinned ALFWorld text runtime/data; never loads model credentials.

Linux Python3.12 supported. Downloads only verified public environment sources
and134 text-game payloads; no THOR, detector, checkpoints, or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import venv
import zipfile

try:
    from . import alfworld_adapter as adapter
except ImportError:
    import alfworld_adapter as adapter

HERE = Path(__file__).resolve().parent
SOURCE_URL = "https://codeload.github.com/alfworld/alfworld/tar.gz/" + adapter.SOURCE_COMMIT
DATA_URL = "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
DEPENDENCIES = ("setuptools==75.8.0", "numpy==1.26.4", "textworld==1.7.0",
    "fast-downward-textworld==20.6.4", "TatSu==5.8.3", "PyYAML==6.0.2",
    "requests==2.34.2", "tiktoken==0.12.0")


def _checked_archive(root, name, url, expected, source_cache=None, offline=False):
    destination = root / "downloads" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if adapter._file_sha(destination) != expected:
            raise RuntimeError("Existing public archive hash mismatch: " + name)
        return destination
    candidate = Path(source_cache) / name if source_cache else None
    partial = destination.with_suffix(destination.suffix + ".partial")
    if candidate is not None and candidate.is_file():
        shutil.copyfile(candidate, partial)
    elif offline:
        raise RuntimeError("Offline source archive missing: " + name)
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "blade-setup"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if adapter._file_sha(partial) != expected:
        raise RuntimeError("Public archive SHA256 mismatch: " + name)
    partial.replace(destination)
    return destination


def _extract_source(archive, root):
    directory = root / "upstream" / ("alfworld-" + adapter.SOURCE_COMMIT)
    if directory.exists():
        for name, expected in adapter.SOURCE_FILES_SHA256.items():
            if adapter._file_sha(directory / "alfworld" / name) != expected:
                raise RuntimeError("Existing ALFWorld source differs: " + name)
        return directory
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / (directory.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(temporary, filter="data")
    tops = list(temporary.iterdir())
    if len(tops) != 1 or not tops[0].is_dir():
        raise RuntimeError("Unexpected checked ALFWorld source archive layout")
    tops[0].rename(directory)
    temporary.rmdir()
    return directory


def _extract_data(archive, root):
    target = root / "upstream/alfworld-data"
    if target.exists():
        adapter.make_cases("paper", target)
        return target
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        members = [member for member in source.infolist()
            if member.filename.startswith("json_2.1.1/valid_unseen/")
            and member.filename.endswith("/game.tw-pddl")]
        if len(members) != 134:
            raise RuntimeError("Checked ALFWorld archive no longer has134 unseen games")
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe public archive member")
            path = temporary.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read(member))
    adapter.make_cases("paper", temporary)
    temporary.rename(target)
    return target


def prepare_sources(runtime_root, source_cache=None, offline=False):
    root = Path(runtime_root).expanduser().resolve()
    source = _checked_archive(root, "alfworld-" + adapter.SOURCE_COMMIT + ".tar.gz",
        SOURCE_URL, adapter.SOURCE_ARCHIVE_SHA256, source_cache, offline)
    data = _checked_archive(root, "json_2.1.3_tw-pddl.zip", DATA_URL,
        adapter.DATA_ARCHIVE_SHA256, source_cache, offline)
    return _extract_source(source, root), _extract_data(data, root)


def install_runtime(root, source):
    directory = Path(root).resolve() / "venvs/alfworld"
    python = directory / "bin/python"
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(directory)
    subprocess.run([str(python), "-m", "pip", "install", *DEPENDENCIES], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
        "--no-build-isolation", str(source)], check=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=HERE / ".runtime")
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--offline", action="store_true",
        help="Require public source/data archives from cache; no download")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python3.12 for the isolated ALFWorld runtime")
    root = args.runtime_root.resolve()
    source, data = prepare_sources(root, args.source_cache, args.offline)
    if args.download_only:
        print("Checked ALFWorld source and all134 games ready; no model calls.")
        return
    python = install_runtime(root, source)
    if not args.skip_smoke:
        code = """import json
from blade.alfworld_adapter import AlfWorldGame,make_cases
game=AlfWorldGame(make_cases('smoke')[0])
try:
 print(json.dumps({'case_count':len(make_cases()),'initial_menu_size':len(game.actions()),'metrics':game.metrics()}))
finally:game.close()
"""
        env = dict(os.environ, BLADE_ALFWORLD_DATA=str(data), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([str(python), "-c", code], env=env, cwd=HERE.parent, check=True)
    path = root / "runtime_paths.json"
    paths = json.loads(path.read_text()) if path.is_file() else {}
    paths["alfworld"] = str(python)
    path.write_text(json.dumps(paths, indent=2) + "\n")
    print("Native ALFWorld preparation complete; no paid requests or credentials used.")
    print("BLADE_ALFWORLD_DATA=" + str(data))


if __name__ == "__main__":
    main()
