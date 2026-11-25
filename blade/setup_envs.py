"""Download checked upstream sources and install isolated native runtimes.

No credentials, model inference, package-global environment modification, or
automatic sudo. Linux Python 3.12 is the supported installation target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import venv

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "upstream.lock.json").read_text())
GROUPS = {"grid": ("balrog", "minigrid", "baba"),
    "nle": ("balrog", "balrog_nle", "minihack", "boxoban")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(name, root, cache=None):
    row = LOCK["sources"][name]
    archive = root / "downloads" / (row["folder"] + ".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_file():
        if sha(archive) != row["sha256"]:
            raise RuntimeError("Existing dependency archive has the wrong hash: " + str(archive))
        return archive
    temporary = archive.with_suffix(".partial")
    # Optional read-only cache import is verified identically to a download.
    candidate = Path(cache) / archive.name if cache else None
    if candidate and candidate.is_file():
        shutil.copyfile(candidate, temporary)
    else:
        request = urllib.request.Request(row["url"], headers={"User-Agent": "blade-setup"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
    if sha(temporary) != row["sha256"]:
        raise RuntimeError("Downloaded dependency failed SHA256 verification: " + name)
    temporary.replace(archive)
    return archive


def unpack(name, archive, root):
    row = LOCK["sources"][name]
    destination = root / "upstream" / row["folder"]
    marker = destination / ".blade-source-sha256"
    if destination.exists():
        if marker.is_file() and marker.read_text().strip() == row["sha256"]:
            return destination
        raise RuntimeError("Existing upstream directory is not verified; move it aside: " + str(destination))
    temporary = root / "upstream" / (row["folder"] + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    with tarfile.open(archive) as source:
        # data filter rejects escaping paths, device entries, and unsafe links.
        source.extractall(temporary, filter="data")
    tops = list(temporary.iterdir())
    if len(tops) != 1 or not tops[0].is_dir():
        raise RuntimeError("Unexpected upstream archive layout: " + name)
    tops[0].rename(destination)
    temporary.rmdir()
    marker.write_text(row["sha256"] + "\n")
    return destination


def run(args, env=None, cwd=None):
    subprocess.run([str(a) for a in args], check=True, env=env, cwd=cwd)


def install(group, root, sources):
    target = root / "venvs" / group
    if not (target / "bin/python").is_file():
        venv.EnvBuilder(with_pip=True).create(target)
    python = target / "bin/python"
    run([python, "-m", "pip", "install", "-r", HERE / f"requirements-{group}.txt"])
    if group == "grid":
        run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", sources["minigrid"], sources["baba"]])
    else:
        missing = [name for name in ("gcc", "g++", "make", "flex", "bison") if not shutil.which(name)]
        if missing:
            raise RuntimeError("Native NLE build tools missing: " + ", ".join(missing) +
                ". Install Ubuntu/Debian build-essential flex bison libncurses-dev zlib1g-dev libbz2-dev, then rerun.")
        build_env = dict(os.environ)
        build_env["PATH"] = str(target / "bin") + os.pathsep + os.environ.get("PATH", "")
        build_env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "4")
        run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", sources["balrog_nle"]], build_env)
        run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", sources["minihack"]], build_env)
        location = subprocess.check_output([str(python), "-c",
            "import pathlib,minihack;print(pathlib.Path(minihack.__file__).resolve().parent/'dat'/'boxoban-levels-master')"], text=True).strip()
        destination = Path(location)
        source = sources["boxoban"]
        # MiniHack normally fetches mutable master on first use. Install the
        # frozen bank before any episode instead; no level selection is changed.
        if destination.exists():
            source_files = {p.relative_to(source): sha(p) for p in source.rglob("*.txt")}
            actual = {p.relative_to(destination): sha(p) for p in destination.rglob("*.txt")}
            if source_files != actual:
                raise RuntimeError("Existing Boxoban bank differs from frozen source")
        else:
            shutil.copytree(source, destination)
    run([python, "-m", "pip", "install", "-r", HERE / "requirements-runner.txt"])
    return str(python)


def smoke_one(name, root, python):
    env = dict(os.environ, BLADE_RUNTIME_ROOT=str(root), SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy", PYTHONHASHSEED="0")
    code = """import json
from blade.environment import Game,make_cases
name=__import__('sys').argv[1]
game=Game(name,make_cases(name,profile='smoke')[0])
try:
 public=game.observe(); menu=game.actions()
 assert set(public)=={'environment','mission','native_instruction','observation','action_descriptions'}
 assert list(public['action_descriptions'])==menu
 before=len(menu);game.step(menu[0]);assert game.actions()==menu
 print(json.dumps({'environment':name,'native_menu_count':before,'metrics':game.metrics()}))
finally:game.close()
"""
    run([python, "-c", code, name], env, HERE.parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    parser.add_argument("--runtime-root", type=Path, default=HERE / ".runtime")
    parser.add_argument("--source-cache", type=Path, help="Optional directory of exact checked archives")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true", help="Install only; native verification remains required before evaluation")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 (tested); separate legacy gym runtimes will be created")
    root = args.runtime_root.resolve()
    names = sorted({name for group in args.groups for name in GROUPS[group]})
    sources = {name: unpack(name, download(name, root, args.source_cache), root) for name in names}
    if args.download_only:
        print("Checked source archives ready; no models or credentials used.")
        return
    paths = {group: install(group, root, sources) for group in args.groups}
    previous_path = root / "runtime_paths.json"
    combined = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    combined.update(paths)
    previous_path.write_text(json.dumps(combined, indent=2) + "\n")
    if not args.skip_smoke:
        for group, python in paths.items():
            for name in {"grid": ("babyai", "babaisai"), "nle": ("minihack",)}[group]:
                smoke_one(name, root, python)
    print("Native setup complete; no model calls were made.")


if __name__ == "__main__":
    main()
