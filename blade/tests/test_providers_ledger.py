"""Offline contracts: no keys, network, model weights, or paid calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from blade.ledger import BudgetHalt, Halt, Ledger
from blade.providers import (D3_FORMAT, D3_MODEL_SHA256, D3_REPO,
    D3_REVISION, GOAL_SCHEMA, GPT_ENDPOINT, GPT_MODEL, JEV_ENDPOINT, JEV_MODEL,
    ProviderError, Providers, Receipt, extract_goal, parse_distribution,
    settings_for, strict_json, usage_and_cost)


def config():
    return {"protocol_hash": hashlib.sha256(b"offline-test-only").hexdigest(),
            "providers": {
              "jev": {"endpoint": JEV_ENDPOINT, "model": JEV_MODEL,
                      "timeout_seconds": 30, "max_input_tokens": 32768,
                      "prices_per_million": {"input": 0.042, "output": 0}},
              "gpt": {"endpoint": GPT_ENDPOINT, "model": GPT_MODEL,
                      "reasoning_effort": "medium", "service_tier": "default",
                      "timeout_seconds": 30, "max_prompt_tokens": 32768,
                      "max_output_tokens": 4096,
                      "prices_per_million": {"input": 2.5, "cached_input": 0.2, "output": 10}},
              "d3": {"endpoint": "http://127.0.0.1:8902/v1/systemone", "model": "d3",
                     "repo_id": D3_REPO, "revision": D3_REVISION,
                     "model_sha256": D3_MODEL_SHA256, "timeout_seconds": 30,
                     "max_input_tokens": 32768}},
            "budgets": {"jev_usd": 5, "gpt_usd": 5, "total_usd": None}}


def action_payload(model=JEV_MODEL, observation="public observation", order=("left", "right")):
    return {"model": model, "state": {"observation": observation},
            "questions": {"action": {"type": "choice", "instructions": "Choose one action.",
                                      "criteria": {action: action for action in order}}}}


def action_response(payload):
    menu = list(payload["questions"]["action"]["criteria"])
    return {"model": payload["model"], "answers": {"action": {"type": "choice",
            "choice": menu[0], "probabilities": {menu[0]: 0.6, menu[1]: 0.4}}},
            "usage": {"input_tokens": 100, "output_tokens": 0}}


def goal_payload():
    return {"model": GPT_MODEL, "input": '{"observation":"public"}',
            "instructions": "Return one desired state.", "reasoning": {"effort": "medium"},
            "service_tier": "default", "max_output_tokens": 4096, "store": False,
            "truncation": "disabled", "text": {"format": {"type": "json_schema",
            "name": "desired_state", "strict": True, "schema": deepcopy(GOAL_SCHEMA)}}}


def goal_response():
    return {"model": GPT_MODEL, "status": "completed", "service_tier": "default",
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": '{"goal":"reach the door"}'}]}],
            "usage": {"input_tokens": 100, "output_tokens": 200,
                      "input_tokens_details": {"cached_tokens": 20},
                      "output_tokens_details": {"reasoning_tokens": 180}}}


def replica_process(out, cfg, endpoint, queue, started, finish, dispatch_file):
    """Spawn-compatible actor stub; no HTTP, credentials, or paid requests."""
    cfg = deepcopy(cfg)
    cfg["providers"]["d3"]["endpoint"] = endpoint
    def provider(payload, settings):
        with Path(dispatch_file).open("a") as handle:
            handle.write(settings["endpoint"] + "\n")
        started.set()
        if not finish.wait(5):
            raise RuntimeError("offline process fixture timed out")
        return action_response(payload)
    ledger = None
    try:
        ledger = Ledger(out, cfg, providers={"d3": provider})
        queue.put(("ready", endpoint))
        rid, raw = ledger.call("d3", action_payload("d3"), "case-" + endpoint, 0, "pi0")
        queue.put(("complete", endpoint, rid, raw))
    except Exception as exc:
        queue.put(("error", endpoint, type(exc).__name__ + ": " + str(exc)))
        raise
    finally:
        if ledger is not None:
            ledger.close()


class ProviderContracts(unittest.TestCase):
    def test_full_native_menu_preserved_and_bad_response_is_rejected(self):
        payload = action_payload()
        raw = action_response(payload)
        self.assertEqual(parse_distribution(raw, payload), {"left": 0.6, "right": 0.4})
        incomplete = deepcopy(raw)
        del incomplete["answers"]["action"]["probabilities"]["right"]
        with self.assertRaises(ProviderError):
            parse_distribution(incomplete, payload)
        self.assertEqual(raw["answers"]["action"]["probabilities"], {"left": 0.6, "right": 0.4})

    def test_invalid_mass_nonfinite_are_not_repaired(self):
        payload = action_payload()
        for values, choice in (({"left": 0.1, "right": 0.1}, "left"),
                               ({"left": float("nan"), "right": 0.4}, "left")):
            raw = action_response(payload)
            raw["answers"]["action"].update(probabilities=values, choice=choice)
            with self.assertRaises(ProviderError):
                parse_distribution(raw, payload)
        # A non-maximal provider choice is accepted; only the distribution is used.
        raw = action_response(payload)
        raw["answers"]["action"].update(probabilities={"left": 0.6, "right": 0.4}, choice="right")
        self.assertEqual(parse_distribution(raw, payload), {"left": 0.6, "right": 0.4})

    def test_gpt_refusal_extra_fields_and_duplicate_fields_are_rejected(self):
        self.assertEqual(extract_goal(goal_response()), {"goal": "reach the door"})
        for content in ({"type": "refusal", "refusal": "no"},
                        {"type": "output_text", "text": '{"goal":"x","action":"left"}'},
                        {"type": "output_text", "text": '{"goal":"x","goal":"y"}'}):
            raw = goal_response()
            raw["output"][1]["content"] = [content]
            with self.assertRaises(ProviderError):
                extract_goal(raw)

    def test_billing_includes_reasoning_once_and_missing_usage_is_unknown(self):
        settings = settings_for(config(), "gpt")
        usage = usage_and_cost("gpt", goal_response(), settings)
        self.assertAlmostEqual(usage["cost_usd"], (80 * 2.5 + 20 * 0.2 + 200 * 10) / 1e6)
        self.assertEqual(usage["reasoning_tokens"], 180)
        raw = goal_response()
        del raw["usage"]
        self.assertIsNone(usage_and_cost("gpt", raw, settings)["cost_usd"])
        self.assertFalse(usage_and_cost("gpt", raw, settings)["billing_known"])
        self.assertIsNone(usage_and_cost("d3", action_response(action_payload("d3")),
                                        settings_for(config(), "d3"))["cost_usd"])

    def test_endpoint_credentials_queries_and_unpinned_d3_are_rejected(self):
        for endpoint in ("http://example.org/v1/systemone", "https://key@example.org/v1/systemone",
                         "https://example.org/v1/systemone?api_key=secret"):
            cfg = config()
            cfg["providers"]["d3"]["endpoint"] = endpoint
            with self.assertRaises(ProviderError):
                settings_for(cfg, "d3")
        cfg = config()
        cfg["providers"]["d3"]["revision"] = "main"
        with self.assertRaises(ProviderError):
            settings_for(cfg, "d3")
        cfg = config()
        cfg["providers"]["jev"]["api_key"] = "must-not-enter-cache-identity"
        with self.assertRaises(ProviderError):
            settings_for(cfg, "jev")

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(value)

    def test_d3_health_and_complete_native_menu_http_contract(self):
        cfg = config()
        seen = []
        def transport(kind, endpoint, payload):
            seen.append((kind, endpoint, deepcopy(payload)))
            if payload is None:
                return {"status": "ready", "model": "d3", "max_input_tokens": 32768,
                        "runtime_identity": {"repo_id": D3_REPO, "revision": D3_REVISION,
                            "model_sha256": D3_MODEL_SHA256, "format_id": D3_FORMAT}}
            return action_response(payload)
        provider = Providers(cfg, transport=transport)
        payload = action_payload("d3")
        receipt = provider.call("d3", payload)
        self.assertEqual(parse_distribution(receipt.raw, payload), {"left": 0.6, "right": 0.4})
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1][2], payload)

    def test_every_declared_replica_must_pass_the_same_scientific_health_contract(self):
        cfg = config()
        pool = [cfg["providers"]["d3"]["endpoint"], "http://127.0.0.1:8903/v1/systemone"]
        cfg["providers"]["d3"]["endpoints"] = pool
        cfg["providers"]["d3"]["runtime_identity"] = {"attention_mode": "noncausal_full_attention"}
        seen = []
        def transport(kind, endpoint, payload):
            seen.append(endpoint)
            return {"status": "ready", "model": "d3", "max_input_tokens": 32768,
                    "runtime_identity": {"repo_id": D3_REPO, "revision": D3_REVISION,
                        "model_sha256": D3_MODEL_SHA256, "format_id": D3_FORMAT,
                        "attention_mode": "noncausal_full_attention" if "8902" in endpoint else "causal"}}
        providers = Providers(cfg, transport=transport)
        with self.assertRaises(ProviderError):
            providers.call("d3", action_payload("d3"))
        self.assertEqual(seen, ["http://127.0.0.1:8902/health", "http://127.0.0.1:8903/health"])

    def test_worker_endpoint_outside_declared_pool_is_rejected(self):
        cfg = config()
        cfg["providers"]["d3"]["endpoints"] = [cfg["providers"]["d3"]["endpoint"]]
        cfg["providers"]["d3"]["endpoint"] = "http://127.0.0.1:8903/v1/systemone"
        with self.assertRaises(ProviderError):
            settings_for(cfg, "d3")


class LedgerContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = config()
        self.proxy = patch("blade.providers.token_proxy", return_value=100)
        self.proxy.start()
        self.addCleanup(self.proxy.stop)

    def ledger(self, providers, cfg=None, suffix="run"):
        ledger = Ledger(Path(self.temp.name) / suffix, cfg or self.cfg, providers=providers)
        self.addCleanup(ledger.close)
        return ledger

    def test_exact_cache_counts_full_cost_for_each_method(self):
        calls = []
        def provider(payload, settings):
            calls.append(deepcopy(payload))
            return action_response(payload)
        ledger = self.ledger({"jev": provider})
        payload = action_payload()
        first = ledger.call("jev", payload, "jev-h0-case1", 0, "pi0")
        second = ledger.call("jev", payload, "jev-w2-case1", 0, "pi0")
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        stats = ledger.statistics()
        self.assertEqual(stats["physical"]["jev"]["requests"], 1)
        for episode in ("jev-h0-case1", "jev-w2-case1"):
            self.assertEqual(stats["logical_by_episode"][episode]["jev"]["requests"], 1)
            self.assertAlmostEqual(stats["logical_by_episode"][episode]["jev"]["cost_usd"], 100 * .042 / 1e6)

    def test_statistics_uses_numeric_sql_aggregates_without_loading_receipt_bodies(self):
        ledger = self.ledger({"jev": lambda payload, settings: action_response(payload)})
        for step in range(5):
            for episode in ("case-a", "case-b"):
                ledger.call("jev", action_payload(observation=str(step)), episode, step, "pi0")
        queries = []
        ledger.db.set_trace_callback(queries.append)
        with patch.object(ledger, "report_rows", side_effect=AssertionError("must not load raw receipt bodies")):
            stats = ledger.statistics()
            spend = ledger.spend("jev")
        ledger.db.set_trace_callback(None)
        self.assertEqual(stats["physical"]["jev"]["requests"], 5)
        self.assertEqual(stats["logical_by_episode"]["case-a"]["jev"]["requests"], 5)
        self.assertEqual(stats["logical_by_episode"]["case-b"]["jev"]["requests"], 5)
        self.assertEqual(spend["physical_requests"], 5)
        self.assertAlmostEqual(spend["known_cost_usd"], stats["physical"]["jev"]["cost_usd"])
        self.assertFalse(any("payload" in query or "raw_body" in query or "r.raw" in query for query in queries))
        self.assertFalse(any("SELECT * FROM requests" in query for query in queries))

    def test_budget_null_or_too_small_never_dispatches(self):
        calls = []
        for cap in (None, 0.0, 0.0001):
            cfg = deepcopy(self.cfg)
            cfg["budgets"]["gpt_usd"] = cap
            ledger = self.ledger({"gpt": lambda *args: calls.append(args)}, cfg, str(cap))
            with self.assertRaises(BudgetHalt):
                ledger.call("gpt", goal_payload(), "case", 8, "goal")
            self.assertEqual(ledger.statistics()["physical"]["gpt"]["requests"], 0)
        self.assertEqual(calls, [])

    def test_known_billed_format_failure_retained_without_retry(self):
        calls = []
        def provider(payload, settings):
            calls.append(1)
            raw = action_response(payload)
            raw["answers"]["action"]["probabilities"] = {"left": 1.0}
            return raw
        ledger = self.ledger({"jev": provider})
        for _ in range(2):
            with self.assertRaises(Halt):
                ledger.call("jev", action_payload(), "case", 0, "pi0")
        self.assertEqual(len(calls), 1)
        stats = ledger.statistics()["physical"]["jev"]
        self.assertEqual(stats["failed"], 1)
        self.assertAlmostEqual(stats["cost_usd"], 100 * .042 / 1e6)

    def test_unknown_billing_retains_reservation_without_blocking_new_paid_requests(self):
        calls = []
        def provider(payload, settings):
            calls.append(1)
            raw = goal_response()
            del raw["usage"]
            return raw
        ledger = self.ledger({"gpt": provider, "jev": lambda *args: calls.append(2)})
        with self.assertRaises(Halt):
            ledger.call("gpt", goal_payload(), "case", 8, "goal")
        spend = ledger.spend("gpt")
        self.assertEqual(spend["unknown_billing_requests"], 1)
        self.assertGreater(spend["unknown_reserved_usd"], 0)
        with self.assertRaises(Halt):
            ledger.call("jev", action_payload(), "case", 8, "pi0")
        self.assertEqual(calls, [1, 2])
        with self.assertRaises(Halt):
            ledger.call("gpt", goal_payload(), "case", 8, "goal")
        self.assertEqual(calls, [1, 2])

    def test_restart_recovers_orphaned_inflight_without_redispatch(self):
        ledger = self.ledger({"gpt": lambda *args: self.fail("must never dispatch")})
        payload = goal_payload()
        rid = ledger.request_id("gpt", payload)
        settings = settings_for(self.cfg, "gpt")
        with ledger.transaction() as db:
            db.execute("INSERT INTO requests(id,kind,status,payload,identity,endpoint,started,reservation_usd,billing_known) "
                       "VALUES(?,?,?,?,?,?,?,?,?)", (rid, "gpt", "pending", json.dumps(payload),
                       json.dumps(ledger._identity("gpt"), ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                       settings["endpoint"], time.time(), ledger._reservation("gpt", settings), 0))
        resumed = self.ledger({"gpt": lambda *args: self.fail("must never redispatch")})
        self.assertEqual(resumed.report_rows()["requests"][0]["status"], "uncertain")
        with self.assertRaises(Halt):
            resumed.call("gpt", payload, "case", 8, "goal")

    def test_duplicate_threads_dispatch_once(self):
        calls = []
        def provider(payload, settings):
            calls.append(1)
            time.sleep(.03)
            return action_response(payload)
        ledger = self.ledger({"jev": provider})
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda episode: ledger.call("jev", action_payload(), episode, 0, "pi0"),
                                        ("case-a", "case-b")))
        self.assertEqual(results[0], results[1])
        self.assertEqual(calls, [1])

    def test_two_processes_share_success_across_declared_replicas(self):
        cfg = deepcopy(self.cfg)
        pool = [cfg["providers"]["d3"]["endpoint"], "http://127.0.0.1:8903/v1/systemone"]
        cfg["providers"]["d3"]["endpoints"] = pool
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        started, finish = context.Event(), context.Event()
        output = Path(self.temp.name) / "multiprocess"
        dispatch_file = Path(self.temp.name) / "dispatches.txt"
        processes = [context.Process(target=replica_process,
            args=(str(output), cfg, endpoint, queue, started, finish, str(dispatch_file))) for endpoint in pool]
        def cleanup():
            finish.set()
            for process in processes:
                if process.pid is not None:
                    process.join(2)
                    if process.is_alive():
                        process.terminate()
                        process.join(2)
            queue.close()
        self.addCleanup(cleanup)
        processes[0].start()
        self.assertEqual(queue.get(timeout=5), ("ready", pool[0]))
        self.assertTrue(started.wait(5))
        # Opening a second ledger must not misclassify the live first request
        # as orphaned, nor reject its different selected replica address.
        processes[1].start()
        self.assertEqual(queue.get(timeout=5), ("ready", pool[1]))
        finish.set()
        results = [queue.get(timeout=5), queue.get(timeout=5)]
        self.assertEqual([item[0] for item in results], ["complete", "complete"])
        self.assertEqual(results[0][2], results[1][2])
        for process in processes:
            process.join(5)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(dispatch_file.read_text().splitlines(), [pool[0]])
        ledger = self.ledger({}, cfg, "multiprocess")
        rows = ledger.report_rows()
        self.assertEqual(len(rows["requests"]), 1)
        self.assertEqual(rows["requests"][0]["endpoint"], pool[0])
        self.assertEqual(len(rows["logical_uses"]), 2)
        self.assertEqual(rows["requests"][0]["status"], "complete")

    def test_parallel_distinct_requests_cannot_overreserve(self):
        cfg = deepcopy(self.cfg)
        reserve = (32768 * 2.5 + 4096 * 10) / 1e6
        cfg["budgets"]["gpt_usd"] = reserve + .00001
        started = threading.Event()
        finish = threading.Event()
        calls = []
        def provider(payload, settings):
            calls.append(1)
            started.set()
            finish.wait(1)
            return goal_response()
        ledger = self.ledger({"gpt": provider}, cfg)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(ledger.call, "gpt", goal_payload(), "case-a", 8, "goal")
            self.assertTrue(started.wait(1))
            payload = goal_payload()
            payload["input"] = '{"observation":"other"}'
            with self.assertRaises(BudgetHalt):
                ledger.call("gpt", payload, "case-b", 8, "goal")
            finish.set()
            future.result()
        self.assertEqual(calls, [1])

    def test_native_menu_order_changes_identity(self):
        ledger = self.ledger({"jev": lambda payload, settings: action_response(payload)})
        self.assertNotEqual(ledger.request_id("jev", action_payload()),
                            ledger.request_id("jev", action_payload(order=("right", "left"))))

    def test_scientific_changes_rejected_but_budget_can_change(self):
        ledger = self.ledger({"jev": lambda payload, settings: action_response(payload)})
        cfg = deepcopy(self.cfg)
        cfg["budgets"]["jev_usd"] = 10
        self.ledger({}, cfg)
        cfg["protocol_hash"] = hashlib.sha256(b"another-protocol").hexdigest()
        with self.assertRaises(Halt):
            Ledger(Path(self.temp.name) / "run", cfg, providers={})

    def test_credential_errors_are_redacted_and_not_logged(self):
        def provider(payload, settings):
            raise RuntimeError("unexpected echo " + os.environ["OPENAI_API_KEY"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-credential-never-usable"}):
            ledger = self.ledger({"gpt": provider})
            with self.assertRaises(Halt):
                ledger.call("gpt", goal_payload(), "case", 8, "goal")
            text = json.dumps(ledger.report_rows())
            self.assertNotIn("test-only-credential-never-usable", text)
            self.assertIn("[REDACTED]", text)

    def test_d3_cost_remains_null_and_needs_no_cloud_budget(self):
        cfg = deepcopy(self.cfg)
        cfg["budgets"] = {"jev_usd": None, "gpt_usd": None}
        ledger = self.ledger({"d3": lambda payload, settings: action_response(payload)}, cfg)
        ledger.call("d3", action_payload("d3"), "case", 0, "pi0")
        stats = ledger.statistics()
        self.assertIsNone(stats["physical"]["d3"]["cost_usd"])
        self.assertIsNone(stats["spend"]["d3"]["accounted_usd"])


if __name__ == "__main__":
    unittest.main()
