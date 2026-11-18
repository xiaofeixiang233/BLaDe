"""Pinned ALFWorld text-only unseen134 adapter with native admissible menus.

The dynamic menu is an explicitly disclosed engine interface, not a static
action-space experiment. No expert wrapper, plan, state facts, free look or
inventory action, custom target-completion check, or semantic action repair is
used. Native scores are evaluation-only and never appear in observe().
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import threading

HERE = Path(__file__).resolve().parent
VERSION = "official-alfworld-unseen134-native-admissible-v3"
SOURCE_COMMIT = "aaba6870f86c5be6a08a491f32a50b906227bc3e"
SOURCE_ARCHIVE_SHA256 = "5592fbb36124b08d24167c5f7612a55a2cc610e0c39170f638a69b628835ee3b"
DATA_ARCHIVE_SHA256 = "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"
GAME_INVENTORY_SHA256 = "54532d99be9e2f9e458bbfa0366de3316655becc629d399ff47b1063dac19fd5"
SPLIT_RELATIVE = Path("json_2.1.1/valid_unseen")
TASK_COUNTS = {
    "pick_and_place_simple": 24, "look_at_obj_in_light": 18,
    "pick_clean_then_place_in_recep": 31,
    "pick_heat_then_place_in_recep": 23,
    "pick_cool_then_place_in_recep": 21, "pick_two_obj_and_place": 17,
}
SOURCE_FILES_SHA256 = {
    "info.py": "fcf9941faed64d221260b531d54e0e84ffb1739ccc94f8a6168c4645b932edef",
    "agents/utils/misc.py": "28972caffbf77c3ab2cfad9789a91cc02ace472f4a90769937c11024e18f5e44",
    "agents/environment/alfred_tw_env.py": "4d9166363e32f0fa70c63530a900785ddb45c79387a02ebcb9a21252869a8924",
    "data/alfred.twl2": "c5b22f0f73309a7fd7c8e5b71de42ad4803b60766ab4a51cc2feb34162be036b",
    "data/alfred.pddl": "f1473041ed01da514de1f87adfd7ef0804862361f83ccf63c75e43633b0eb280",
    "gen/goal_library.py": "132b9407facbde7f7fc0f31e913f4678cc95a19bfb5f282cb827197da7f02dbc",
    "gen/constants.py": "2bd365f5bbca9bfa53b590421af36a0c58a1e074d0582cbc5f174078f16b7377",
}
KNOWN_PREVIOUS_CASE_ID_SHA256 = frozenset({
    "1f726d4a4336887adab3b3f8cccbcc969393bd66eca609ef7f5269cf882afd2c",
    "2d4d85077d5eeca03f7649e328562dfacf88247b19c9a1699d2fa45a344bc0ec",
    "431c0266a5853b31927fb649c9f72b17edeb534460ca119f3f7de3282e793839",
    "5166f3991cb179c51d678dcca9c27715a4f2ded411663a20ff61a012f47de665",
    "558693ebee65b0cfbe23bb5a98640a78dc0584b6250a0bc113501407b1401acd",
    "76703616e5dee1f68e208ca657a24266c234c572cb2f7f6b7d7252101f8e82e1",
    "91b8ad113eb21acb643cb34c7446360fb29d9ccbd68ad363d9f03e55b116a9de",
    "9af6b39929b251fd1904ba356307c8d2b655c64e6d618f98fb0663ea03cd2ee3",
    "aefbda0d9d1282460b4a80178483ae1c262ecaa6daadb6368ed446ffa84938ab",
    "ee81265ca6161be43d02b194603d6f891838ef0d372d7a27bcc792c7019d74f7",
    "fe8f951f508f813999931760867214203bf22930616a4338a0550fd03905e321",
    "ffdedefd02111c73a43ce5646fcfb20cc1198fb540ec208b3bdb4beaf74224fe",
})
CHOICE_LIMIT = 255
HORIZON = 50
_ENGINE_LOCK = threading.RLock()


def _sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_data_root(data_root=None):
    root = Path(data_root or os.environ.get("BLADE_ALFWORLD_DATA",
        HERE / ".runtime/upstream/alfworld-data")).expanduser().resolve()
    split = root if root.name == "valid_unseen" else root / SPLIT_RELATIVE
    if not split.is_dir():
        raise RuntimeError("Pinned ALFWorld unseen data missing; run setup_alfworld.py "
            "or set BLADE_ALFWORLD_DATA to the checked extracted data root")
    return split


def make_cases(profile="paper", data_root=None, horizon=HORIZON):
    """Verify ALL134 before returning either the full list or one smoke case.

    Portable paths are relative to valid_unseen; no model outcomes are read.
    The fixed inventory hash authenticates filenames and every game payload.
    """
    if profile not in ("paper", "smoke") or type(horizon) is not int or horizon != HORIZON:
        raise ValueError("ALFWorld profiles are paper/smoke with fixed50-action horizon")
    split = resolve_data_root(data_root)
    inventory, rows = {}, []
    for path in sorted(split.rglob("game.tw-pddl")):
        if path.is_symlink() or not path.resolve().is_relative_to(split):
            raise ValueError("ALFWorld game path escapes pinned data directory")
        relative = path.relative_to(split).as_posix()
        inventory[relative] = _file_sha(path)
        game = json.loads(path.read_text())
        family = path.parent.parent.name
        task = family.split("-", 1)[0]
        if task not in TASK_COUNTS or "movable" in family or "Sliced" in family:
            raise ValueError("Unexpected task in pinned complete134 data")
        if game.get("solvable") is not True:
            raise ValueError("Pinned official solvable=true flag changed")
        case_id = "alfworld/" + relative.removesuffix("/game.tw-pddl")
        group = _sha({"task_type": task, "official_task_family": family})
        rows.append({"id": case_id, "environment": "alfworld", "task": task,
            "task_type": task, "profile": profile, "split": "valid_unseen",
            "horizon": horizon, "seed": 0,
            "game_relative_path": relative, "game_sha256": inventory[relative],
            "official_task_family": family, "group": group,
            "bootstrap_group": group, "data_inventory_sha256": GAME_INVENTORY_SHA256,
            "known_previously_used_12_case": hashlib.sha256(case_id.encode()).hexdigest()
                in KNOWN_PREVIOUS_CASE_ID_SHA256,
            "external_history_fully_audited": False,
            "independent_confirmation": False,
            "menu_protocol": "complete_native_admissible_commands"})
    if (len(rows) != 134 or dict(Counter(v["task_type"] for v in rows)) != TASK_COUNTS
            or len({v["group"] for v in rows}) != 47
            or _sha(inventory) != GAME_INVENTORY_SHA256):
        raise ValueError("Pinned complete134 ALFWorld inventory/count/hash changed; no resampling")
    rows.sort(key=lambda row: row["id"])
    return deepcopy(rows[:1] if profile == "smoke" else rows)


def native_instruction():
    """Only interface grammar, without task-type decomposition or advice."""
    return (
        "ALFWorld is a text household environment. Follow the public task instruction. "
        "Choose one exact command from the current complete engine-generated admissible "
        "command menu, including its object numbering. Native command forms include look, "
        "inventory, help, go to <receptacle>, open <receptacle>, close <receptacle>, "
        "take <object> from <receptacle>, move <object> to <receptacle>, examine <object>, "
        "use <object>, heat <object> with <receptacle>, cool <object> with <receptacle>, "
        "clean <object> with <receptacle>, and slice <object> with <object>. "
        "The menu is supplied by the environment and can change after each action.")


def _native_bindings(data_root):
    # Set only the native data-cache location; never inspect model credentials.
    os.environ["ALFWORLD_DATA"] = str(Path(data_root).resolve())
    import alfworld
    import textworld
    from alfworld.agents.utils.misc import Demangler
    installed = Path(alfworld.__file__).resolve().parent
    for relative, expected in SOURCE_FILES_SHA256.items():
        path = installed / relative
        if not path.is_file() or _file_sha(path) != expected:
            raise RuntimeError("Installed ALFWorld source identity changed: " + relative)
    versions = {name: importlib.metadata.version(name) for name in
        ("alfworld", "textworld", "fast-downward-textworld", "numpy", "tatsu", "pyyaml")}
    expected = {"alfworld": "0.5.0", "textworld": "1.7.0",
        "fast-downward-textworld": "20.6.4", "numpy": "1.26.4",
        "tatsu": "5.8.3", "pyyaml": "6.0.2"}
    if versions != expected:
        raise RuntimeError("Pinned ALFWorld dependency versions changed")
    return textworld, Demangler, versions


class NativeMenuOverflow(RuntimeError):
    """Contains the COMPLETE public menu for retained failure records."""
    def __init__(self, case_id, commands, public_observation=None, decisions_completed=0):
        self.record = {"failure": "native_menu_exceeds_choice_limit",
            "case_id": case_id, "menu_size": len(commands),
            "choice_limit": CHOICE_LIMIT, "complete_native_menu": list(commands),
            "public_observation": deepcopy(public_observation),
            "request_sent": False, "request_scope": "next_choice_for_overflow_menu",
            "native_decisions_completed": decisions_completed,
            "prior_request_costs_retained": True, "actions_removed": 0}
        super().__init__(json.dumps(self.record, ensure_ascii=False, sort_keys=True))


class AlfWorldGame:
    """The Game interface, using exact native feedback and dynamic commands."""
    def __init__(self, case, data_root=None):
        self.case = deepcopy(case)
        self.name, self.steps = "alfworld", 0
        self.closed = False
        self.total_reward = self.score = 0.
        self.done = self.terminated = self.truncated = self.won = False
        self.native = None
        split = resolve_data_root(data_root)
        cases = make_cases(case.get("profile", "paper"), split, case.get("horizon"))
        if not any(candidate == case for candidate in cases):
            raise ValueError("ALFWorld case differs from pinned complete134 manifest")
        relative = Path(case["game_relative_path"])
        path = (split / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(split) or _file_sha(path) != case["game_sha256"]:
            raise ValueError("ALFWorld native case path/hash changed")
        textworld, Demangler, self.package_versions = _native_bindings(split.parent)

        class NativeNames(textworld.core.Wrapper):
            # Exact official AlfredDemangler naming behavior, no expert import.
            def load(self, *args, **kwargs):
                super().load(*args, **kwargs)
                names = Demangler(game_infos=self._entity_infos, shuffle=False)
                for info in self._entity_infos.values():
                    info.name = names.demangle_alfred_name(info.id)

        try:
            infos = textworld.EnvInfos(admissible_commands=True, won=True)
            with _ENGINE_LOCK:
                self.native = textworld.start(str(path), request_infos=infos, wrappers=[NativeNames])
                self.raw = self.native.reset()
            match = re.search(r"Your task is to:\s*(.+)", self.raw.feedback)
            if match is None or not match.group(1).strip() or "UNKNOWN GOAL" in match.group(1):
                raise ValueError("Public ALFWorld reset instruction is absent")
            self.mission = match.group(1).strip()
            self.native_instruction = native_instruction()
            self.won = bool(self.raw["won"])
            self.score = float(self.won)
            self.done = self.terminated = self.won
            self._project()
        except BaseException:
            self.close()
            raise

    def _project(self):
        if not isinstance(self.raw.feedback, str):
            raise ValueError("Native ALFWorld feedback must be text")
        self.text = self.raw.feedback  # exact current feedback, no free observation action
        commands = self.raw["admissible_commands"]
        if (not isinstance(commands, list) or not commands
                or any(not isinstance(a, str) or not a for a in commands)
                or len(commands) != len(set(commands))):
            raise ValueError("Native ALFWorld menu must contain distinct nonempty commands")
        self.menu = sorted(commands)
        self.descriptions = {command: command for command in self.menu}
        if len(self.menu) > CHOICE_LIMIT:
            raise NativeMenuOverflow(self.case["id"], self.menu, self.observe(), self.steps)

    def observe(self):
        return {"environment": self.name, "mission": self.mission,
            "native_instruction": self.native_instruction, "observation": self.text,
            "action_descriptions": dict(self.descriptions)}

    def actions(self):
        if len(self.menu) > CHOICE_LIMIT:
            raise NativeMenuOverflow(self.case["id"], self.menu, self.observe(), self.steps)
        return list(self.menu)

    def step(self, action):
        if self.closed or self.done:
            raise ValueError("Cannot step a closed or completed episode")
        if not isinstance(action, str) or action not in self.menu:
            raise ValueError("Selected command absent from complete native menu; no fallback")
        before = self.score
        with _ENGINE_LOCK:
            self.raw, raw_score, raw_done = self.native.step(action)
        if (not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score)
                or raw_score not in (0, 1)):
            raise ValueError("Native ALFWorld binary score semantics changed")
        self.steps += 1
        self.score = float(raw_score)
        self.won = bool(self.raw["won"])
        self.terminated = bool(raw_done)
        self.truncated = bool(self.steps >= self.case["horizon"] and not self.terminated)
        self.done = self.terminated or self.truncated
        reward = self.score - before
        self.total_reward += reward
        self._project()
        return {"reward": reward, "done": self.done,
            "terminated": self.terminated, "truncated": self.truncated,
            "won": self.won, "feedback": self.text, "observation": self.observe()}

    def metrics(self):
        return {"won": self.won, "done": self.done, "terminated": self.terminated,
            "truncated": self.truncated, "score": self.score,
            "native_reward": self.total_reward, "decision_steps": self.steps,
            "steps": self.steps, "native_steps": self.steps,
            "horizon": self.case["horizon"], "action_count": len(self.menu),
            "current_menu_sha256": _sha(self.menu),
            "menu_protocol": "complete_native_admissible_commands",
            "progression": float(self.won)}

    def identity(self):
        return {"adapter_version": VERSION, "official_repository": "https://github.com/alfworld/alfworld",
            "official_commit": SOURCE_COMMIT,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "data_archive_sha256": DATA_ARCHIVE_SHA256,
            "game_inventory_sha256": GAME_INVENTORY_SHA256,
            "source_files_sha256": dict(SOURCE_FILES_SHA256),
            "actual_package_versions": dict(self.package_versions),
            "split": "valid_unseen", "available_game_count": 134,
            "task_family_groups": 47, "native_action_horizon": HORIZON,
            "menu_protocol": "complete_native_admissible_commands",
            "menu_is_state_dependent": True,
            "custom_goal_completion_or_legality_checks": False,
            "expert_or_planner_requested": False,
            "external_history_fully_audited": False,
            "independent_confirmation": False}

    def close(self):
        if not self.closed:
            if self.native is not None:
                with _ENGINE_LOCK:
                    self.native.close()
            self.closed = True
