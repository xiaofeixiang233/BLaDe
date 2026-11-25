"""Pinned native BALROG interfaces; no planning or decision heuristics.

Only text emitted by official wrappers reaches inference. The four BALROG
games use complete static language menus. ALFWorld uses its complete native
admissible menu, explicitly authorized and identical across methods.
Native rewards/statistics are returned solely for evaluation and never used to
finish a GPT target, select an action, or repair a loop.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "upstream.lock.json").read_text())
RUNTIME = Path(os.environ.get("BLADE_RUNTIME_ROOT", HERE / ".runtime")).resolve()
BALROG = RUNTIME / "upstream" / LOCK["sources"]["balrog"]["folder"]
BENCHMARKS = ("babyai", "minihack", "babaisai", "alfworld")
SEED_START = 20271014
CHOICE_LIMIT = 255
BABYAI_TASKS = ("goto", "pickup", "open", "putnext", "pick_up_seq_go_to")
BABYAI_KINDS = ("goto", "pickup", "open", "putnext", "pick up seq go to")
MINIHACK_TASKS = (
    "MiniHack-Boxoban-Hard-v0", "MiniHack-Boxoban-Medium-v0",
    "MiniHack-MazeWalk-9x9-v0", "MiniHack-MazeWalk-15x15-v0",
    "MiniHack-Corridor-R3-v0", "MiniHack-CorridorBattle-Dark-v0",
    "MiniHack-Quest-Easy-v0", "MiniHack-Quest-Medium-v0",
)
BABA_TASKS = tuple(LOCK["baba_tasks"])
MINIHACK_SETTINGS = dict(character="@", max_episode_steps=100, penalty_step=-.01,
    penalty_time=0., penalty_mode="constant", savedir=None, save_ttyrec_every=0,
    autopickup=False, skip_more=True)
NLE_SETTINGS = dict(character="@", max_episode_steps=100000,
    no_progress_timeout=150, savedir=None, save_ttyrec_every=0, skip_more=True)


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def canonical_name(name):
    return {"nethack": "nle", "baba": "babaisai"}.get(name, name)


def runtime_paths():
    return {name: str(RUNTIME / "venvs" / (
        "grid" if name in ("babyai", "babaisai") else
        "alfworld" if name == "alfworld" else "nle") / "bin/python")
        for name in BENCHMARKS}


def make_cases(name, profile="paper", limit=None, seed_start=SEED_START):
    """Task-major cases frozen by identity, never selected using outcomes.

    'paper' uses original-paper counts; 'default' uses pinned repository counts;
    'smoke' is one case per benchmark. Seeds are our fixed seeds, not a claim of
    reproducing the paper's precise RNG states. limit takes a deterministic
    prefix solely for interface checks and is not a full-benchmark run.
    """
    name = canonical_name(name)
    if name == "alfworld":
        from .alfworld_adapter import make_cases as alf_cases
        return alf_cases(profile=profile, horizon=50)
    if name not in BENCHMARKS or profile not in ("paper", "default", "smoke"):
        raise ValueError("Unknown benchmark or case profile")
    if type(seed_start) is not int or seed_start < 0:
        raise ValueError("seed_start must be a nonnegative integer")
    tasks = {"babyai": BABYAI_TASKS, "minihack": MINIHACK_TASKS,
        "babaisai": BABA_TASKS, "nle": ("NetHackChallenge-v0",)}[name]
    repetitions = {"babyai": 25 if profile == "paper" else 10,
        "minihack": 5, "babaisai": 5 if profile == "paper" else 3,
        "nle": 5}[name]
    rows = []
    # Sharing the seed bank across task types does not expose hidden state.
    for task in tasks:
        for i in range(repetitions):
            seed = seed_start + i
            horizon = (64 if task in ("goto", "pickup") else 128) if name == "babyai" else {
                "minihack": 100, "babaisai": 100, "nle": 100000}[name]
            rows.append({"id": f"{name}/{profile}/{task}/{seed}", "environment": name,
                "task": task, "task_type": task, "seed": seed, "profile": profile,
                "horizon": horizon, "group": f"{name}/{task}/{seed}",
                "paper_exact_seed_replication": False})
    if profile == "smoke":
        rows = rows[:1]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        rows = rows[:limit]
    return rows


def wrapper_identity():
    actual = {}
    for relative, expected in LOCK["balrog_files_sha256"].items():
        path = BALROG / relative
        if not path.is_file():
            raise RuntimeError("Upstream dependencies missing; run setup_envs.py")
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[relative] != expected:
            raise RuntimeError("Pinned BALROG source changed: " + relative)
    return {"balrog_revision": LOCK["sources"]["balrog"]["revision"],
        "source_files_sha256": actual,
        "native_revisions": {k: v["revision"] for k, v in LOCK["sources"].items()},
        "native_caps": {"babyai": [64, 128], "minihack": 100,
            "babaisai": 100, "crafter": 2000, "nle": 100000},
        "public_projection": "Complete original public text, no raw arrays/images/hidden state",
        "menus": "Complete static native language menus; no state legality filter",
        "compatibility_changes": [
            "BabyAI constructor RNG fixes mission category before unchanged native reset.",
            "Baba unused expert target_plan read omitted; original no-YOU reset retained with fresh public text serialization.",
            "NLE selected 00..99 menu labels mechanically emit both literal digits; all native subactions retained."],
        "custom_goal_completion_or_legality_checks": False,
        "crafter_engine_modified": False,
        "nle_calendar_caveat": "Pinned NLE retains native calendar events; paired initial public hashes must match."}


def _module(relative, name):
    path = BALROG / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("Pinned public constant absent: " + name)


def _prompt_module(name):
    """Execute only upstream constants/functions, without unrelated imports."""
    folder = "babyai_text" if name == "babyai" else name
    path = BALROG / f"balrog/environments/{folder}/__init__.py"
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
        and n.name == "get_instruction_prompt" or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in ("ACTIONS", "ACTION_DICT") for t in n.targets)]
    namespace = {}
    if name == "crafter":
        namespace["ACTIONS"] = list(_module("balrog/environments/crafter/env.py",
            "_six_crafter_for_prompt").ACTIONS)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _package():
    if str(BALROG) not in sys.path:
        sys.path.insert(0, str(BALROG))
    from balrog.environments import make_env
    if not Path(sys.modules[make_env.__module__].__file__).resolve().is_relative_to(BALROG.resolve()):
        raise RuntimeError("BALROG import points outside pinned source")
    return make_env


@contextmanager
def _python_seed(seed):
    previous = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(previous)


class _Base:
    def __init__(self, name, case):
        self.name, self.case = name, deepcopy(case)
        self.steps = 0
        self.total_reward = self.score = 0.
        self.done = self.terminated = self.truncated = self.won = False
        self.closed, self.info = False, {}

    def _menu(self, menu):
        self.menu = list(menu)
        if not self.menu or len(self.menu) != len(set(self.menu)) or any(not isinstance(a, str) or not a for a in self.menu):
            raise RuntimeError("Native menu must contain distinct nonempty strings")
        if len(self.menu) > CHOICE_LIMIT:
            raise RuntimeError(f"Native menu has {len(self.menu)} actions, exceeding Choice limit {CHOICE_LIMIT}; no actions removed")

    def observe(self):
        return {"environment": self.name, "mission": self.mission,
            "native_instruction": self.native_instruction,
            "observation": deepcopy(self.text),
            "action_descriptions": deepcopy(self.descriptions)}

    def actions(self):
        return list(self.menu)

    def _check(self, action):
        # This is schema/interface validation, never state-dependent legality.
        if self.closed or self.done:
            raise ValueError("Cannot step a closed or completed episode")
        if action not in self.menu:
            raise ValueError("Model action absent from complete static native menu; no fallback")

    def _project(self):
        self.text = deepcopy(self.raw["text"])

    def _result(self, reward):
        return {"reward": float(reward), "done": self.done,
            "terminated": self.terminated, "truncated": self.truncated,
            "won": self.won, "feedback": deepcopy(self.text),
            "observation": self.observe()}

    def metrics(self):
        return {"won": self.won, "done": self.done, "terminated": self.terminated,
            "truncated": self.truncated, "score": self.score,
            "native_reward": self.total_reward, "decision_steps": self.steps,
            "steps": self.steps, "native_steps": self.steps,
            "horizon": self.case["horizon"], "action_count": len(self.menu),
            "fixed_menu_sha256": _sha(self.menu)}

    def close(self):
        if not self.closed:
            obj = getattr(self, "wrapper", None) or getattr(self, "native", None)
            if obj is not None:
                obj.close()
            self.closed = True


class _BabyAI(_Base):
    def __init__(self, name, case):
        super().__init__(name, case)
        import numpy as np
        from minigrid.envs.babyai.mixed_seq_levels import Level_MixedTrainLocal
        official = _module("balrog/environments/babyai_text/clean_lang_wrapper.py", "_six_babyai")
        kind = BABYAI_KINDS[BABYAI_TASKS.index(case["task"])]
        constructor_seed = next(s for s in range(1000)
            if BABYAI_KINDS[int(np.random.default_rng(s).integers(0, 5))] == kind)

        class Constructor(Level_MixedTrainLocal):
            def __init__(self):
                self._np_random = np.random.default_rng(constructor_seed)
                super().__init__(num_dists=0, room_size=8, render_mode=None)

        self.native = Constructor()
        self.wrapper = official.BabyAITextCleanLangWrapper(self.native)
        self.raw, self.info = self.wrapper.reset(seed=case["seed"])
        if self.native.max_steps != case["horizon"] or self.native.action_kinds != [kind]:
            raise RuntimeError("Native BabyAI category/horizon differs")
        self._menu(self.wrapper.language_action_space)
        prompts = _prompt_module(name)
        self.descriptions = {a: prompts["ACTIONS"][a] for a in self.menu}
        self.mission = self.raw["mission"]
        self._project()
        self.native_instruction = prompts["get_instruction_prompt"](self.wrapper, self.mission)

    def step(self, action):
        self._check(action)
        self.raw, reward, terminated, truncated, self.info = self.wrapper.step(action)
        self.steps += 1
        self.total_reward += float(reward)
        self.score = self.total_reward
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self.done = self.terminated or self.truncated
        self.won = bool(self.won or self.terminated and reward > 0)
        self._project()
        return self._result(reward)

    def metrics(self):
        return {**super().metrics(), "native_steps": int(self.native.step_count),
            "progression": float(self.won)}


class _Baba(_Base):
    def __init__(self, name, case):
        super().__init__(name, case)
        import baba
        official = _module("balrog/environments/babaisai/base.py", "_six_baba")

        class PublicWrapper(official.BabaIsAIWrapper):
            """Serialization-only compatibility; official automatic reset kept."""
            def __init__(self, env):
                super().__init__(env, add_ruleset=True)
                self.auto_reset_count = self.reset_count = 0

            def reset(self, **kwargs):
                if self.reset_count:
                    self.auto_reset_count += 1
                self.reset_count += 1
                obs = self.env.reset(**kwargs)
                self.progression = 0.
                # Upstream target_plan is unused expert data; do not read it.
                return self.textworld_process_obsv(obs)

            def textworld_process_obsv(self, obs):
                text, reset = self.get_text_observation(obs)
                if reset:
                    fresh = deepcopy(obs if not isinstance(text, dict) else text)
                    fresh["text"]["long_term_context"] = (
                        "[...] IS YOU rule was broken, environment reset\n" +
                        fresh["text"]["long_term_context"])
                    return fresh
                return super().textworld_process_obsv(obs)

        self.native = baba.make(case["task"], max_steps=case["horizon"])
        self.wrapper = PublicWrapper(self.native)
        self.raw = self.wrapper.reset(seed=case["seed"])
        self._menu(self.wrapper.language_action_space)
        prompts = _prompt_module(name)
        self.descriptions = {a: prompts["ACTIONS"][a] for a in self.menu}
        self.native_instruction = prompts["get_instruction_prompt"](self.wrapper)
        self.mission = self.native_instruction
        self._project()

    def step(self, action):
        self._check(action)
        self.raw, reward, native_done, self.info = self.wrapper.step(action)
        self.steps += 1
        self.total_reward += float(reward)
        self.score = self.total_reward
        self.won = bool(self.won or native_done and reward > 0)
        # Official reset can restart the native step counter. The benchmark's
        # episode action cap is still 100; no target-based termination exists.
        self.done = bool(native_done or self.steps >= self.case["horizon"])
        self.truncated = bool(self.done and not self.won and (
            self.steps >= self.case["horizon"] or self.native.step_count >= self.native.max_steps))
        self.terminated = self.done and not self.truncated
        self._project()
        return self._result(reward)

    def metrics(self):
        return {**super().metrics(), "progression": float(self.wrapper.progression),
            "native_steps_since_last_reset": int(self.native.step_count),
            "automatic_resets": self.wrapper.auto_reset_count}


class _NLEFamily(_Base):
    def __init__(self, name, case):
        super().__init__(name, case)
        make = _package()
        if name == "minihack":
            from balrog.environments.minihack import ACTIONS, get_instruction_prompt
            options = deepcopy(MINIHACK_SETTINGS)
        else:
            from balrog.environments.nle import ACTIONS, get_instruction_prompt
            options = deepcopy(NLE_SETTINGS)
        cfg = SimpleNamespace(envs=SimpleNamespace(**{name + "_kwargs": options}),
            agent=SimpleNamespace(max_image_history=0))
        with _python_seed(case["seed"]):
            self.wrapper = make(name, case["task"], cfg)
            self.language = self.wrapper.env.gym_env.env
            self.native = self.language.env.unwrapped
            self.native.seed(core=case["seed"], disp=case["seed"], reseed=False)
            self.raw, self.info = self.wrapper.reset()
        self._menu(self.language.language_action_space)
        self.descriptions = {a: ACTIONS.get(a, a) for a in self.menu}
        self.native_instruction = get_instruction_prompt(self.language, case["task"]) if name == "minihack" else get_instruction_prompt()
        self.mission = self.native_instruction
        self._project()

    def step(self, action):
        self._check(action)
        # Upstream lists two-digit labels but only translates primitive keys.
        # Submit precisely the selected label's literal keys, without deciding
        # which native prompt to answer or inspecting state to replace actions.
        primitives = list(action) if self.name == "nle" and len(action) == 2 and action.isascii() and action.isdigit() else [action]
        retained, total_reward = [], 0.
        for primitive in primitives:
            self.raw, reward, terminated, truncated, self.info = self.wrapper.step(primitive)
            total_reward += float(reward)
            retained.append({"action": primitive, "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "public_observation": deepcopy(self.raw["text"])})
            if terminated or truncated:
                break
        self.steps += 1
        self.total_reward += total_reward
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self.done = self.terminated or self.truncated
        self.end_status = getattr(self.info.get("end_status"), "name", str(self.info.get("end_status")))
        self.won = self.end_status == "TASK_SUCCESSFUL" if self.name == "minihack" else bool(self.info.get("is_ascended", False))
        self._project()
        if self.name == "nle":
            self.score = float(self.wrapper.get_stats()["score"])
        else:
            self.score = self.total_reward
        result = self._result(total_reward)
        result["native_subactions"] = retained
        return result

    def metrics(self):
        value = {**super().metrics(), "native_steps": int(getattr(self.native, "_steps", self.steps)),
            "end_status": getattr(self, "end_status", None)}
        if self.name == "nle":
            stats = self.wrapper.get_stats()
            value.update({key: stats[key] for key in (
                "score", "progression", "episode_return", "depth", "experience_level", "end_reason")})
        else:
            value["progression"] = float(self.won)
        return value


class Game:
    """Construct inside the selected benchmark's isolated native runtime."""
    def __init__(self, name, case):
        name = canonical_name(name)
        if name == "alfworld":
            from .alfworld_adapter import AlfWorldGame
            self._game = AlfWorldGame(case)
            return
        if name not in BENCHMARKS or canonical_name(case.get("environment")) != name:
            raise ValueError("Case does not match a supported benchmark")
        tasks = {"babyai": BABYAI_TASKS, "minihack": MINIHACK_TASKS,
            "babaisai": BABA_TASKS, "nle": ("NetHackChallenge-v0",)}[name]
        if case.get("task") not in tasks:
            raise ValueError("Task absent from pinned benchmark bank")
        horizon = (64 if case["task"] in ("goto", "pickup") else 128) if name == "babyai" else {
            "minihack": 100, "babaisai": 100, "nle": 100000}[name]
        if case.get("horizon") != horizon or type(case.get("seed")) is not int:
            raise ValueError("Case native horizon or seed differs")
        wrapper_identity()
        cls = {"babyai": _BabyAI, "babaisai": _Baba, "minihack": _NLEFamily,
            "nle": _NLEFamily}[name]
        self._game = cls.__new__(cls)
        try:
            cls.__init__(self._game, name, case)
        except BaseException:
            self._game.close()
            raise

    def observe(self): return self._game.observe()
    def actions(self): return self._game.actions()
    def step(self, action): return self._game.step(action)
    def metrics(self): return self._game.metrics()
    def close(self): self._game.close()

    def identity(self):
        """Actual native installation facts for freezing/auditing manifests."""
        name = self._game.name
        if name == "alfworld":
            return self._game.identity()
        packages = ["numpy", "gym", "gymnasium", "Pillow"]
        packages += {"babyai": ["minigrid"], "babaisai": ["baba"],
            "minihack": ["minihack", "balrog-nle"], "nle": ["balrog-nle"]}[name]
        native_packages = {"babyai": ("minigrid",), "babaisai": ("baba",),
            "minihack": ("minihack", "nle"), "nle": ("nle",)}[name]
        hashes = {}
        for package in native_packages:
            folder = Path(sys.modules[package].__file__).resolve().parent
            # Python sources and native shared libraries record actual runtime
            # identity; absolute installation paths are not published metadata.
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix in (".py", ".so", ".yaml"):
                    hashes[package + "/" + path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {**wrapper_identity(), "actual_package_versions": {
            p: importlib.metadata.version(p) for p in packages},
            "actual_native_files_sha256": hashes}
