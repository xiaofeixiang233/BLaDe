"""Accounting counterexamples, using synthetic receipts and no model calls."""
from copy import deepcopy
import json
import unittest

from blade.efficiency import summarize_efficiency


def episode(identity, method="jev-flat-h0", status="complete", won=False,
            progression=0, decisions=10, wall=20):
    return {"id": identity, "environment": "crafter", "case": {"id": "case1"},
            "method": method, "status": status, "metrics": {"won": won, "progression": progression},
            "decisions": decisions, "episode_wall_seconds": wall}


def request(request_id, kind="jev", cost=.1, owner=None, latency=2, **extra):
    return {"id": request_id, "kind": kind, "status": "complete", "cost_usd": cost,
            "billing_known": 1, "owner_episode": owner, "latency": latency,
            "model": "test-model", "credential_slot": kind + "-primary",
            "input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 15,
            "cached_input_tokens": 40, **extra}


def use(identity, receipt, role="pi0", step=0, kind="jev"):
    return {"episode": identity, "request": receipt, "role": role, "step": step, "kind": kind}


class EfficiencyAccounting(unittest.TestCase):
    def test_provider_model_and_slot_latency_are_separate_for_consuming_task(self):
        owner = episode("owner")
        consumer = episode("consumer", "jev-w2")
        consumer["case"]["task_type"] = "same-task"
        requests = [request("actor", owner="owner", latency=1, model="actor-model"),
                    request("planner", kind="gpt", owner="consumer", latency=9,
                            model="planner-model", credential_slot="gpt-key-02")]
        result = summarize_efficiency([owner, consumer], requests,
                    [use("owner", "actor"), use("consumer", "actor"),
                     use("consumer", "planner", role="goal", kind="gpt")])
        logical = result["logical"]
        self.assertEqual(logical["by_environment_method_provider"]["crafter|jev-w2|jev"]["latency_seconds"]["p95"], 1)
        self.assertEqual(logical["by_environment_method_provider"]["crafter|jev-w2|gpt"]["latency_seconds"]["p95"], 9)
        self.assertEqual(logical["by_task_method_model"]["crafter|same-task|jev-w2|gpt|planner-model"]["requests"], 1)
        self.assertEqual(logical["by_task_method_credential_slot"]["crafter|same-task|jev-w2|gpt|gpt-key-02"]["requests"], 1)
        physical = result["physical"]["by_environment_method_provider"]
        self.assertNotIn("crafter|jev-w2|jev", physical)
        self.assertEqual(physical["crafter|jev-flat-h0|jev"]["requests"], 1)

    def test_empty_context_breakdowns_have_stable_schema(self):
        result = summarize_efficiency([], [], [])
        for accounting in ("physical", "logical"):
            for context in ("environment_method", "task_method"):
                for identity in ("provider", "model", "credential_slot"):
                    self.assertEqual(result[accounting]["by_" + context + "_" + identity], {})

    def test_anonymous_key_slots_separate_per_task_usage_and_shared_logical_cost(self):
        one=episode('one');one['case']['task_type']='native-task'
        two=episode('two','jev-w2');two['case']['task_type']='native-task'
        requests=[request('a',owner='one',credential_slot='jev-key-01'),
                  request('b',owner='two',credential_slot='jev-key-02',cost=.2)]
        result=summarize_efficiency([one,two],requests,[use('one','a'),use('two','a'),use('two','b',step=1)])
        physical=result['physical']['by_task_method_credential_slot']
        logical=result['logical']['by_task_method_credential_slot']
        self.assertEqual(physical['crafter|native-task|jev-flat-h0|jev|jev-key-01']['requests'],1)
        self.assertNotIn('crafter|native-task|jev-w2|jev|jev-key-01',physical)
        self.assertEqual(logical['crafter|native-task|jev-w2|jev|jev-key-01']['requests'],1)
        self.assertAlmostEqual(physical['crafter|native-task|jev-w2|jev|jev-key-02']['cost']['total_api_cost_usd'],.2)

    def test_shared_response_costs_full_logically_but_only_explicit_first_owner_physically(self):
        episodes = [episode("one"), episode("two", "jev-w2")]
        requests = [request("r", owner="two")]
        result = summarize_efficiency(episodes, requests, [use("one", "r"), use("two", "r")])
        self.assertEqual(result["physical"]["total"]["requests"], 1)
        self.assertAlmostEqual(result["physical"]["total"]["cost"]["total_api_cost_usd"], .1)
        self.assertEqual(result["logical"]["total"]["requests"], 2)
        self.assertAlmostEqual(result["logical"]["total"]["cost"]["total_api_cost_usd"], .2)
        owners = result["physical"]["first_owner_attribution"]
        self.assertEqual(list(owners["by_episode"]), ["two"])
        self.assertEqual(list(owners["by_method"]), ["jev-w2"])
        self.assertEqual(owners["unallocated"]["requests"], 0)
        self.assertEqual(result["logical"]["by_episode"]["one"]["cost"]["total_api_cost_usd"], .1)

    def test_historical_receipt_ownership_never_inferred_from_use_order(self):
        requests = [request("r"), request("s", owner="not-in-cohort")]
        result = summarize_efficiency([episode("one")], requests, [use("one", "r"), use("one", "s", step=1)])
        attributed = result["physical"]["first_owner_attribution"]
        self.assertEqual(attributed["by_episode"], {})
        self.assertEqual(attributed["unallocated"]["requests"], 2)
        self.assertAlmostEqual(attributed["unallocated"]["cost"]["total_api_cost_usd"], .2)
        self.assertEqual(attributed["unallocated_reasons"],
                         {"owner_not_recorded": 1, "owner_episode_not_in_cohort": 1})

    def test_failed_unknown_bill_and_pending_episodes_are_preserved(self):
        episodes = [episode("done", won=True, progression=.5), episode("failed", status="failed"),
                    episode("pending", status="pending", decisions=None, wall=None)]
        requests = [request("ok", owner="done"), request("bad", status="failed", cost_usd=None,
                     billing_known=0, input_tokens=None, output_tokens=None, latency=4)]
        result = summarize_efficiency(episodes, requests, [use("done", "ok"), use("failed", "bad")])
        total = result["logical"]["total"]
        self.assertEqual(total["episodes"], 3)
        self.assertEqual(total["episode_status_counts"], {"complete": 1, "failed": 1, "pending": 1})
        self.assertEqual(total["successes_over_frozen_denominator"], 1/3)
        self.assertFalse(total["complete_cohort"])
        self.assertFalse(total["efficiency_is_final"])
        self.assertAlmostEqual(total["cost"]["known_cost_subtotal_usd"], .1)
        self.assertIsNone(total["cost"]["total_api_cost_usd"])
        self.assertIsNone(total["successes_per_api_usd"])
        self.assertIsNone(total["summed_progression_per_api_usd"])
        self.assertIsNone(total["tokens_per_decision"])
        self.assertEqual(total["latency_seconds"]["known_requests"], 2)
        self.assertEqual(total["cost"]["unknown_api_billing_requests"], 1)

    def test_reasoning_and_cached_tokens_not_double_counted_and_throughput_is_matched(self):
        requests = [request("a", latency=1), request("b", latency=3),
                    request("unknown-time", latency=None), request("unknown-output", output_tokens=None, latency=100)]
        uses = [use("one", row["id"], step=i) for i, row in enumerate(requests)]
        result = summarize_efficiency([episode("one", won=True, progression=.5)], requests, uses)
        summary = result["physical"]["total"]
        self.assertEqual(summary["tokens"]["input_tokens"]["known_subtotal"], 400)
        self.assertEqual(summary["tokens"]["output_tokens"]["known_subtotal"], 60)
        self.assertEqual(summary["tokens"]["reasoning_tokens"]["known_subtotal"], 60)
        self.assertEqual(summary["total_input_plus_output_known_subtotal"], 460)
        self.assertEqual(summary["throughput_matched_requests"], 2)
        self.assertEqual(summary["tokens_per_receipt_second"], 240/4)
        self.assertIsNone(result["logical"]["total"]["tokens_per_decision"])
        self.assertEqual(summary["latency_seconds"]["unknown_requests"], 1)
        self.assertEqual(summary["latency_seconds"]["p50"], 3)
        self.assertAlmostEqual(summary["latency_seconds"]["p95"], 90.3)

    def test_valid_efficiency_uses_all_incurred_costs_and_steps_success_is_conditional(self):
        episodes = [episode("win", won=True, progression=1, decisions=4),
                    episode("lose", won=False, progression=.5, decisions=20)]
        requests = [request("r", cost=.5), request("s", cost=.5)]
        result = summarize_efficiency(episodes, requests, [use("win", "r"), use("lose", "s")])
        total = result["logical"]["total"]
        self.assertEqual(total["successes_per_api_usd"], 1)
        self.assertEqual(total["summed_progression_per_api_usd"], 1.5)
        self.assertEqual(total["tokens_per_decision"], 240/24)
        self.assertEqual(total["logical_receipt_seconds_per_decision"], 4/24)
        self.assertEqual(total["mean_decisions_successful_completed_known"], 4)
        self.assertEqual(total["mean_decisions_completed_known"], 12)
        self.assertEqual(total["decisions_per_episode_wall_second"], 24/40)
        self.assertTrue(total["efficiency_is_final"])

    def test_zero_or_unknown_costs_yield_null_and_local_gpu_is_not_free(self):
        for kind in ("d3", "kev"):
            requests = [request("local", kind=kind, cost=None, billing_known=0)]
            result = summarize_efficiency([episode("one", method=kind+"-flat-h0", won=True)],
                                          requests, [use("one", "local", kind=kind)])
            total = result["logical"]["total"]
            self.assertIsNone(total["cost"]["total_cost_usd"])
            self.assertEqual(total["cost"]["unknown_local_gpu_cost_requests"], 1)
            self.assertEqual(total["cost"]["total_api_cost_usd"], 0)
            self.assertIsNone(total["successes_per_api_usd"])
            self.assertIsNone(total["summed_progression_per_api_usd"])
            self.assertIn(kind+"|local", result["physical"]["by_credential_slot"])

    def test_anonymous_slots_and_old_model_fallback_do_not_return_private_fields(self):
        row = request("r", model=None, credential_slot="private-value-must-not-appear",
                      identity='{"provider":{"model":"legacy-model"},"secret":"private-secret"}',
                      raw="private-raw")
        original = deepcopy(row)
        result = summarize_efficiency([episode("one")], [row], [use("one", "r")])
        self.assertIn("jev|legacy-model", result["physical"]["by_model"])
        self.assertIn("jev|unrecorded", result["physical"]["by_credential_slot"])
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("private-", serialized)
        self.assertEqual(row, original)

    def test_missing_requests_and_orphan_uses_are_explicit_unknown_dependencies(self):
        result = summarize_efficiency([episode("one")], [],
                                     [use("one", "missing"), use("not-in-cohort", "other")])
        self.assertEqual(result["logical"]["total"]["missing_request_rows"], 2)
        self.assertEqual(result["logical"]["uses_without_episode_in_cohort"]["requests"], 1)
        self.assertIsNone(result["logical"]["by_episode"]["one"]["cost"]["total_api_cost_usd"])
        self.assertEqual(result["physical"]["total"]["requests"], 0)

    def test_duplicate_identical_use_not_double_counted_but_conflicts_rejected(self):
        single = use("one", "r")
        result = summarize_efficiency([episode("one")], [request("r")], [single, single])
        self.assertEqual(result["logical"]["total"]["requests"], 1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            summarize_efficiency([episode("one")], [request("r")], [single, use("one", "other")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_efficiency([episode("one")], [request("r"), request("r")], [])

    def test_task_grouping_and_unknown_missing_metrics_are_explicit(self):
        done = episode("done", won=False, progression=.25)
        done["case"]["task_type"] = "default"
        pending = episode("pending", status="pending", decisions=None, wall=None)
        pending["case"]["task_type"] = "default"
        pending["metrics"] = None
        result = summarize_efficiency([done, pending], [request("r")], [use("done", "r")])
        task = result["logical"]["by_task_method"]["crafter|default|jev-flat-h0"]
        self.assertEqual(task["episodes"], 2)
        self.assertEqual(task["progression_mean_completed_known"], .25)
        self.assertFalse(task["complete_cohort"])
        self.assertEqual(result["logical"]["by_episode"]["pending"]["requests"], 0)
        self.assertIsNone(result["logical"]["by_episode"]["pending"]["tokens_per_decision"])


if __name__ == "__main__":
    unittest.main()
