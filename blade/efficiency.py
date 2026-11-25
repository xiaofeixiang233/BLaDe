"""Pure reporting of physical receipts and full logical experiment costs.

No credential is read, fingerprinted or retained. Only explicit anonymous slot
labels are accepted. A cache hit is a full independent-run logical dependency,
but a physical receipt belongs solely to its explicitly recorded first owner.
"""
from collections import Counter, defaultdict
import json
import math
import re
from statistics import mean


TOKEN_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")
LOCAL_PROVIDERS = frozenset(("d3", "kev"))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower, upper = math.floor(location), math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def _model(row):
    if isinstance(row.get("model"), str) and row["model"]:
        return row["model"]
    # Compatibility with ledgers predating an explicit model column. Access
    # only the named model field, and never return the complete payload/identity.
    for field in ("identity", "payload"):
        value = row.get(field)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                continue
        if isinstance(value, dict):
            value = value.get("provider", value) if field == "identity" else value
            if isinstance(value, dict) and isinstance(value.get("model"), str) and value["model"]:
                return value["model"]
    return "unrecorded"


def _slot(row):
    kind, value = row.get("kind", "unrecorded"), row.get("credential_slot")
    if kind in LOCAL_PROVIDERS:
        return "local"
    # These are operational aliases, not hashes or substrings of a credential.
    # Historical rows without a label stay unrecorded rather than guessing a key.
    if kind in ("jev", "gpt") and isinstance(value,str) and (
            value == kind + "-primary" or re.fullmatch(kind + r"-key-[0-9]{2}",value)):
        return value
    return "unrecorded"


def _cost(rows):
    known, cloud_known = [], []
    unknown, cloud_unknown, local_unknown = 0, 0, 0
    for row in rows:
        local = row.get("kind") in LOCAL_PROVIDERS
        valid = bool(row.get("billing_known")) and _number(row.get("cost_usd"))
        if valid:
            known.append(row["cost_usd"])
            if not local:
                cloud_known.append(row["cost_usd"])
        else:
            unknown += 1
            cloud_unknown += not local
            local_unknown += local
    return {
        "known_cost_subtotal_usd": math.fsum(known),
        "unknown_cost_requests": unknown,
        "total_cost_usd": math.fsum(known) if not unknown else None,
        "known_api_cost_subtotal_usd": math.fsum(cloud_known),
        "unknown_api_billing_requests": cloud_unknown,
        "total_api_cost_usd": math.fsum(cloud_known) if not cloud_unknown else None,
        "unknown_local_gpu_cost_requests": local_unknown,
    }


def _requests(rows):
    rows = list(rows)
    latency = [row["latency"] for row in rows if _number(row.get("latency"))]
    tokens = {field: {
        "known_subtotal": sum(row[field] for row in rows if _number(row.get(field))),
        "unknown_requests": sum(not _number(row.get(field)) for row in rows),
    } for field in TOKEN_FIELDS}
    complete_usage = [row for row in rows
                      if _number(row.get("input_tokens")) and _number(row.get("output_tokens"))]
    throughput = [row for row in complete_usage
                  if _number(row.get("latency")) and row["latency"] > 0]
    matched_tokens = sum(row["input_tokens"] + row["output_tokens"] for row in throughput)
    matched_seconds = math.fsum(row["latency"] for row in throughput)
    result = {
        "requests": len(rows),
        "status_counts": dict(sorted(Counter(row.get("status", "unrecorded") for row in rows).items())),
        "missing_request_rows": sum(row.get("status") == "missing_request" for row in rows),
        "latency_seconds": {
            "known_requests": len(latency), "unknown_requests": len(rows) - len(latency),
            "sum": math.fsum(latency), "mean": mean(latency) if latency else None,
            "p50": _percentile(latency, .5), "p95": _percentile(latency, .95),
        },
        "tokens": tokens,
        "total_input_plus_output_known_subtotal": (
            tokens["input_tokens"]["known_subtotal"] + tokens["output_tokens"]["known_subtotal"]),
        "complete_input_output_usage_requests": len(complete_usage),
        "tokens_per_receipt_second": matched_tokens / matched_seconds if matched_seconds > 0 else None,
        "throughput_matched_requests": len(throughput),
        "throughput_matched_tokens": matched_tokens,
        "throughput_matched_receipt_seconds": matched_seconds,
        "cost": _cost(rows),
    }
    return result


def _group_requests(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: _requests(values) for name, values in sorted(groups.items())}


def _context_request_groups(pairs):
    """Keep actor and planner receipts separate within each evaluation cohort.

    Physical pairs carry the explicit first owner; logical pairs carry the
    consuming episode. No owner is inferred for historical or orphaned rows.
    These are request summaries, not episode-success ratios for a provider.
    """
    grouped = {"by_" + context + "_" + identity: defaultdict(list)
               for context in ("environment_method", "task_method")
               for identity in ("provider", "model", "credential_slot")}
    for episode, row in pairs:
        environment = episode.get("environment", "unallocated")
        method = episode.get("method", "unallocated")
        case = episode.get("case") or {}
        task = str(case.get("task_type", case.get("task", "unrecorded")))
        kind = row.get("kind", "unrecorded")
        contexts = {"environment_method": (environment, method),
                    "task_method": (environment, task, method)}
        identities = {"provider": (kind,), "model": (kind, _model(row)),
                      "credential_slot": (kind, _slot(row))}
        for context, prefix in contexts.items():
            for identity, suffix in identities.items():
                grouped["by_" + context + "_" + identity]["|".join(prefix + suffix)].append(row)
    return {name: {label: _requests(rows) for label, rows in sorted(values.items())}
            for name, values in sorted(grouped.items())}


def _episode_group(episodes, rows):
    summary = _requests(rows)
    complete = [episode for episode in episodes if episode.get("status") == "complete"]
    successes = [episode for episode in complete if (episode.get("metrics") or {}).get("won") is True]
    progress = [(episode.get("metrics") or {}).get("progression") for episode in complete]
    known_progress = [value for value in progress if _number(value)]
    decisions = [episode.get("decisions") for episode in episodes]
    complete_decisions = [episode.get("decisions") for episode in complete]
    success_decisions = [episode.get("decisions") for episode in successes]
    known_decisions = [value for value in decisions if _number(value)]
    matched_wall = [episode for episode in episodes
                    if _number(episode.get("episode_wall_seconds"))
                    and episode["episode_wall_seconds"] > 0 and _number(episode.get("decisions"))]
    matched_seconds = math.fsum(episode["episode_wall_seconds"] for episode in matched_wall)
    cost = summary["cost"]["total_api_cost_usd"]
    positive_cost = cost is not None and cost > 0
    all_decisions_known = len(known_decisions) == len(episodes)
    all_tokens_known = bool(rows) and summary["complete_input_output_usage_requests"] == summary["requests"]
    decision_total = sum(known_decisions)
    statuses = Counter(episode.get("status", "unrecorded") for episode in episodes)
    summary.update({
        "episodes": len(episodes), "episode_status_counts": dict(sorted(statuses.items())),
        "complete_episodes": len(complete), "successful_complete_episodes": len(successes),
        "complete_cohort": len(complete) == len(episodes),
        "successes_over_frozen_denominator": len(successes) / len(episodes) if episodes else None,
        "progression_sum_completed_known": math.fsum(known_progress),
        "progression_unknown_completed_episodes": len(progress) - len(known_progress),
        "progression_mean_completed_known": mean(known_progress) if known_progress else None,
        "decisions_known_subtotal": decision_total,
        "unknown_decision_episodes": len(episodes) - len(known_decisions),
        "mean_decisions_completed_known": mean([d for d in complete_decisions if _number(d)])
            if any(_number(d) for d in complete_decisions) else None,
        "mean_decisions_successful_completed_known": mean([d for d in success_decisions if _number(d)])
            if any(_number(d) for d in success_decisions) else None,
        "tokens_per_decision": summary["total_input_plus_output_known_subtotal"] / decision_total
            if all_tokens_known and all_decisions_known and decision_total > 0 else None,
        "logical_receipt_seconds_per_decision": summary["latency_seconds"]["sum"] / decision_total
            if all_decisions_known and decision_total > 0 and not summary["latency_seconds"]["unknown_requests"] else None,
        "episode_wall_seconds_known_subtotal": math.fsum(
            episode["episode_wall_seconds"] for episode in episodes if _number(episode.get("episode_wall_seconds"))),
        "episode_wall_seconds_unknown_episodes": sum(
            not _number(episode.get("episode_wall_seconds")) for episode in episodes),
        "decisions_per_episode_wall_second": sum(episode["decisions"] for episode in matched_wall) / matched_seconds
            if matched_seconds > 0 else None,
        "wall_throughput_matched_episodes": len(matched_wall),
        "successes_per_api_usd": len(successes) / cost if positive_cost else None,
        "summed_progression_per_api_usd": math.fsum(known_progress) / cost
            if positive_cost and len(progress) == len(known_progress) else None,
        "efficiency_is_final": len(complete) == len(episodes),
    })
    return summary


def summarize_efficiency(episodes, requests, logical_uses):
    """Return JSON-safe efficiency records without reading keys or other files.

    Rows use the ledger schema. New ledgers optionally record owner_episode,
    model and credential_slot. Unlabelled historical receipts are unallocated;
    ownership is never guessed from the order of logical uses. Missing receipt
    references become explicit unknown-usage/unknown-cost dependencies.
    """
    episodes = [dict(row) for row in episodes]
    requests = [dict(row) for row in requests]
    uses = [dict(row) for row in logical_uses]
    episode_map = {episode["id"]: episode for episode in episodes}
    request_map = {row["id"]: row for row in requests}
    if len(episode_map) != len(episodes) or len(request_map) != len(requests):
        raise ValueError("Duplicate episode or physical request identity")
    dependencies, seen = defaultdict(list), {}
    for use in uses:
        key = (use["episode"], use["step"], use["role"])
        identity = (use["kind"], use["request"])
        if key in seen:
            if seen[key] != identity:
                raise ValueError("Conflicting logical dependency identity")
            continue
        seen[key] = identity
        row = request_map.get(use["request"])
        if row is None:
            row = {"id": use["request"], "kind": use["kind"], "status": "missing_request"}
        elif row["kind"] != use["kind"]:
            raise ValueError("Logical dependency provider differs from physical request")
        dependencies[use["episode"]].append(row)

    by_episode = {name: _episode_group([episode], dependencies.get(name, []))
                  for name, episode in sorted(episode_map.items())}
    orphan_uses = [row for name, values in dependencies.items() if name not in episode_map for row in values]
    all_logical = [row for values in dependencies.values() for row in values]
    env_method, methods, tasks = defaultdict(list), defaultdict(list), defaultdict(list)
    for episode in episodes:
        env_method[(episode.get("environment", "unrecorded"), episode.get("method", "unrecorded"))].append(episode)
        methods[episode.get("method", "unrecorded")].append(episode)
        case = episode.get("case") or {}
        tasks[(episode.get("environment", "unrecorded"),
               str(case.get("task_type", case.get("task", "unrecorded"))),
               episode.get("method", "unrecorded"))].append(episode)

    def groups(cohorts):
        result = {}
        for name, cohort in sorted(cohorts.items()):
            label = "|".join(name) if isinstance(name, tuple) else name
            rows = [row for episode in cohort for row in dependencies.get(episode["id"], [])]
            result[label] = _episode_group(cohort, rows)
        return result

    owned, unallocated, reasons = defaultdict(list), [], Counter()
    for row in requests:
        owner = row.get("owner_episode")
        if owner in episode_map:
            owned[owner].append(row)
        else:
            unallocated.append(row)
            reasons["owner_not_recorded" if not owner else "owner_episode_not_in_cohort"] += 1
    owned_methods, owned_env_method = defaultdict(list), defaultdict(list)
    for owner, rows in owned.items():
        episode = episode_map[owner]
        method, environment = episode.get("method", "unrecorded"), episode.get("environment", "unrecorded")
        owned_methods[method].extend(rows)
        owned_env_method[environment + "|" + method].extend(rows)
    physical_contexts = _context_request_groups(
        (episode_map.get(row.get("owner_episode"), {}), row) for row in requests)
    logical_contexts = _context_request_groups(
        (episode_map.get(eid, {}), row) for eid, values in dependencies.items() for row in values)
    logical_total = _episode_group(episodes, all_logical)
    if orphan_uses:
        logical_total["tokens_per_decision"] = None
        logical_total["logical_receipt_seconds_per_decision"] = None
    return {
        "physical": {
            "total": _requests(requests),
            "by_provider": _group_requests(requests, lambda row: row.get("kind", "unrecorded")),
            "by_model": _group_requests(requests, lambda row: row.get("kind", "unrecorded") + "|" + _model(row)),
            "by_credential_slot": _group_requests(requests, lambda row: row.get("kind", "unrecorded") + "|" + _slot(row)),
            **physical_contexts,
            "first_owner_attribution": {
                "by_episode": {name: _requests(rows) for name, rows in sorted(owned.items())},
                "by_method": {name: _requests(rows) for name, rows in sorted(owned_methods.items())},
                "by_environment_method": {name: _requests(rows) for name, rows in sorted(owned_env_method.items())},
                "unallocated": _requests(unallocated), "unallocated_reasons": dict(sorted(reasons.items())),
            },
        },
        "logical": {
            "total": logical_total,
            "by_episode": by_episode, "by_method": groups(methods),
            "by_environment_method": groups(env_method), "by_task_method": groups(tasks),
            "by_provider": _group_requests(all_logical, lambda row: row.get("kind", "unrecorded")),
            "by_model": _group_requests(all_logical, lambda row: row.get("kind", "unrecorded") + "|" + _model(row)),
            "by_credential_slot": _group_requests(all_logical, lambda row: row.get("kind", "unrecorded") + "|" + _slot(row)),
            **logical_contexts,
            "uses_without_episode_in_cohort": _requests(orphan_uses),
        },
        "definitions": {
            "physical": "Each actual persisted request counted once, including failed, uncertain and pending requests.",
            "logical": "Every unique episode/step/role dependency carries the entire receipt at independent-run usage, cost and latency; sharing does not reduce logical cost.",
            "ownership": "Only explicit owner_episode attributes physical requests. Missing ownership stays unallocated; no proportional allocation or inference from use order.",
            "context_breakdowns": "Environment-method and task-method provider/model/slot entries are request-only summaries. Physical entries use first ownership; logical entries use the consuming episode. GPT and actor percentiles remain separate.",
            "credential_slots": "Anonymous operational aliases jev-primary/gpt-primary, jev-key-NN/gpt-key-NN or local; missing/unsupported labels are unrecorded. Operator maintains the mapping outside the package. No key reading, hashes or fingerprints.",
            "latency": "Receipt wall latency in seconds; p50/p95 use linear interpolation of known nonnegative observations. Includes failures when measured. Logical reused receipt latency is not cache-hit elapsed time.",
            "tokens": "Input plus output is the token total. Cached input is a subset of input; reasoning is a subset of output. Neither subset is added again. Missing counts remain explicit.",
            "tokens_per_receipt_second": "Sum(input+output)/sum(receipt latency) over requests with both token counts and positive measured latency; not end-to-end throughput or generation-only speed.",
            "tokens_per_decision": "Full logical dependency input+output tokens per model action decision, only when all required token and decision counts are known.",
            "cost": "Known subtotals do not turn unknown costs into zero. API cost excludes local GPU cost, which remains unknown unless explicitly accounted. Zero or unknown denominators yield null efficiency.",
            "efficiency": "Successes or sum of completed progression divided by known positive total API cost. Failed and unfinished episodes stay in cohort counts and their incurred costs stay charged. Non-complete cohorts are explicitly provisional; no future usage is projected.",
            "success_steps": "Mean decisions among successful completed episodes is conditional, not an unconditional efficiency estimate.",
            "progress": "Progression is the benchmark metric. Won is a separate metric and must not replace progression.",
        },
    }
