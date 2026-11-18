"""Portable regression checks against the frozen cap2000 scientific condition.

tests/data/parity_reference.json contains reference hashes/fixtures, so the original repository and
historical outputs are not needed on the receiving machine. No model is called.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from blade import environment, policy

PACKAGE = Path(__file__).resolve().parents[1]
BUNDLE = PACKAGE.parent


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return sha256_bytes(data)


def canonical_ast_dump(node):
    """Preserve AST content across Python 3.9/3.12 schema extensions.

    Python 3.12 adds type_params=[] to nongeneric functions/classes. An empty
    new field carries no source semantics; nonempty type parameters and every
    other AST field remain part of the identity.
    """
    node = deepcopy(node)
    for child in ast.walk(node):
        if getattr(child, "type_params", None) == []:
            child._fields = tuple(field for field in child._fields
                                  if field != "type_params")
    return ast.dump(node, include_attributes=False)


def selected_ast_sha256(path, symbol):
    """Hash exact selected code, ignoring source position/formatting only."""
    tree = ast.parse(Path(path).read_text())
    parts = symbol.split(".")
    node = None
    for part in parts:
        node = next(item for item in tree.body if isinstance(
            item, (ast.FunctionDef, ast.ClassDef)) and item.name == part)
        tree = node
    node = deepcopy(node)
    if symbol == "Game.__init__":
        # The only allowed factory extension is an early ALFWorld dispatch,
        # which is outside this three-benchmark cohort. All remaining code,
        # including supported menus, native caps and dispatch, must match.
        alfworld = ast.dump(ast.parse('name == "alfworld"', mode="eval").body,
                            include_attributes=False)
        node.body = [statement for statement in node.body if not (
            isinstance(statement, ast.If)
            and ast.dump(statement.test, include_attributes=False) == alfworld)]
    return sha256_bytes(canonical_ast_dump(node).encode())


class ShortExperimentParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((PACKAGE / "tests/data/parity_reference.json").read_text())

    def test_tests_import_this_bundle_not_another_checkout(self):
        self.assertEqual(Path(policy.__file__).resolve().parent, PACKAGE)
        self.assertEqual(Path(environment.__file__).resolve().parent, PACKAGE)

    def test_frozen_prompts_native_lock_and_d3_readout_are_byte_identical(self):
        for relative, entry in self.reference["byte_identical_files"].items():
            with self.subTest(file=relative):
                self.assertEqual(sha256_bytes((PACKAGE / relative).read_bytes()),
                                 entry["reference_sha256"])
        lock = json.loads((PACKAGE / "upstream.lock.json").read_text())
        self.assertEqual(lock["balrog_files_sha256"],
                         self.reference["official_wrapper_source_sha256"])

    def test_native_short_environment_and_existing_policy_code_are_preserved(self):
        for filename, symbols in self.reference["selected_ast"].items():
            for symbol, entry in symbols.items():
                with self.subTest(file=filename, symbol=symbol):
                    self.assertEqual(selected_ast_sha256(PACKAGE / filename, symbol),
                                     entry["reference_sha256"])

    def test_complete_native_case_bank_keeps_tasks_seeds_groups_and_caps(self):
        for name, entry in self.reference["short_case_banks"].items():
            with self.subTest(environment=name):
                cases = environment.make_cases(name, profile="paper", seed_start=20271014)
                self.assertEqual(len(cases), entry["cases"])
                self.assertEqual(canonical_sha256(cases), entry["canonical_sha256"])
                self.assertEqual(sorted({case["horizon"] for case in cases}),
                                 entry["native_horizons"])

    def test_existing_distribution_inputs_match_frozen_pre_ablation_payloads(self):
        fixture = self.reference["legacy_payload_fixture"]
        for role, expected in fixture["requests"].items():
            with self.subTest(role=role):
                actual = policy.payload(role, fixture["public"], fixture["actions"],
                                        fixture["history"], fixture["goal"],
                                        fixture["time"], model="d3")
                self.assertEqual(actual, expected)
                self.assertNotIn("hidden-marker", json.dumps(actual))
                self.assertNotIn("private-score-marker", json.dumps(actual))
        history_request = deepcopy(fixture["requests"]["ph"])
        joint = policy.payload("joint", fixture["public"], fixture["actions"],
                               fixture["history"], fixture["goal"], fixture["time"],
                               model="d3")
        history_request["state"]["desired_state"] = fixture["goal"]["goal"]
        self.assertEqual(joint, history_request)
        self.assertEqual(len(joint["state"]["previous_interactions"]), 4)
        self.assertEqual(list(joint["questions"]["action"]["criteria"]),
                         fixture["actions"])

    def test_target_schedule_and_warmup_do_not_depend_on_outcomes(self):
        for horizon, expected in self.reference["target_schedules"].items():
            self.assertEqual(policy.goal_slots(int(horizon), 4), expected)
        for actor in ("jev", "d3"):
            for method in (actor + "-joint-h4",):
                for step in range(8):
                    self.assertEqual(policy.roles_for(method, step), ("ph",))
            self.assertEqual(policy.roles_for(actor + "-joint-h4", 8), ("joint",))
            self.assertEqual(policy.roles_for(actor + "-history-h4", 99), ("ph",))
            self.assertFalse(policy.needs_goal(actor + "-history-h4"))



if __name__ == "__main__":
    unittest.main()
