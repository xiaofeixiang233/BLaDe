"""Observed action timing only; never used by inference or action selection.

Receipt latencies and independent logical costs remain in efficiency.py. This
module measures persisted action-loop times, including actual cache lookup and
waiting, without inventing cache-hit labels or a missing interrupted duration.
"""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from .efficiency import _number, _percentile
from .support import digest, write_json


METRICS = ("decision_wall_seconds", "native_step_seconds", "policy_wall_seconds")
CATEGORIES = ("all", "goal_absent", "goal_update", "goal_reuse", "goal_update_unknown")


def _records(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid timing source JSONL: {path.name}:{line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Non-object timing source row: {path.name}:{line_number}")
            rows.append(row)
    return rows


def _metric(rows, field):
    values = [row[field] for row in rows if _number(row.get(field))]
    return {"known_actions": len(values), "unknown_actions": len(rows) - len(values),
            "known_sum": math.fsum(values),
            "mean": math.fsum(values) / len(values) if values else None,
            "p50": _percentile(values, .5), "p95": _percentile(values, .95)}


def _summarize(episodes, actions):
    result = {
        "episodes": len(episodes),
        "episode_status_counts": dict(sorted(Counter(row.get("status", "unrecorded") for row in episodes).items())),
        "complete_cohort": bool(episodes) and all(row.get("status") == "complete" for row in episodes),
        "trajectory_files_missing": sum(not row["trajectory_file_present"] for row in episodes),
        "retained_actions": len(actions),
        "summary_decision_count_mismatch_episodes": sum(row["summary_decision_count_mismatch"] for row in episodes),
        "negative_policy_duration_actions": sum(row["negative_policy_duration"] for row in actions),
        "interrupted_episodes": sum(row.get("status") == "failed" for row in episodes),
        "interrupted_decision_latency_seconds": None,
        "saved_goal_updates": sum(row["saved_goal_updates"] for row in episodes),
        "goal_updates_without_retained_action": sum(row["goal_updates_without_retained_action"] for row in episodes),
        "by_action_category": {},
    }
    for category in CATEGORIES:
        selected = actions if category == "all" else [row for row in actions if row["category"] == category]
        result["by_action_category"][category] = {
            "actions": len(selected), **{field: _metric(selected, field) for field in METRICS}}
    return result


def summarize_timing(out, episodes):
    """Read persisted trajectories for the frozen cohort, including failures.

    A failed request before native execution has no completed-action timing row.
    Its request latency and cost are still in the ledger, not imputed here.
    Malformed or duplicate trajectory rows stop reporting instead of silently
    discarding an inconvenient timing sample.
    """
    out = Path(out)
    episode_rows, action_rows = [], defaultdict(list)
    seen = set()
    for source in episodes:
        eid = source["id"]
        if eid in seen:
            raise ValueError("Duplicate episode identity in timing cohort")
        seen.add(eid)
        stem = digest(eid)
        trajectory = out / "episodes" / (stem + ".jsonl")
        goal_path = out / "goals" / (stem + ".jsonl")
        records, goals = _records(trajectory), _records(goal_path)
        if any(type(row.get("step")) is not int or row["step"] != index
               for index, row in enumerate(records)):
            raise ValueError("Timing trajectory sequence differs: " + stem)
        goal_steps = [row.get("step") for row in goals]
        if (any(type(step) is not int or step < 0 for step in goal_steps)
                or len(set(goal_steps)) != len(goal_steps)):
            raise ValueError("Timing goal update identity differs: " + stem)
        episode = {"id": eid, "environment": source.get("environment", "unrecorded"),
                   "method": source.get("method", "unrecorded"), "status": source.get("status", "unrecorded"),
                   "task": str((source.get("case") or {}).get("task_type", (source.get("case") or {}).get("task", "unrecorded"))),
                   "trajectory_file_present": trajectory.exists(),
                   "retained_actions": len(records), "saved_goal_updates": len(goals),
                   "goal_updates_without_retained_action": sum(step >= len(records) for step in goal_steps),
                   "summary_decision_count_mismatch": (
                       _number(source.get("decisions")) and source["decisions"] != len(records))}
        episode_rows.append(episode)
        for record in records:
            decision, native = record.get("decision_wall_seconds"), record.get("native_step_seconds")
            matched = _number(decision) and _number(native)
            negative = matched and native > decision
            present = record.get("goal") is not None
            category = ("goal_absent" if not present else "goal_update_unknown" if not goal_path.exists()
                        else "goal_update" if record["step"] in goal_steps else "goal_reuse")
            action_rows[eid].append({
                "decision_wall_seconds": decision if _number(decision) else None,
                "native_step_seconds": native if _number(native) else None,
                "policy_wall_seconds": record['policy_wall_seconds'] if _number(record.get('policy_wall_seconds'))
                    else decision - native if matched and not negative else None,
                "negative_policy_duration": bool(negative), "category": category})

    def grouped(key):
        cohorts = defaultdict(list)
        for episode in episode_rows:
            cohorts[key(episode)].append(episode)
        return {label: _summarize(cohort, [row for episode in cohort for row in action_rows[episode["id"]]])
                for label, cohort in sorted(cohorts.items())}

    return {
        "total": _summarize(episode_rows, [row for values in action_rows.values() for row in values]),
        "by_episode": grouped(lambda row: row["id"]),
        "by_environment_method": grouped(lambda row: row["environment"] + "|" + row["method"]),
        "by_task_method": grouped(lambda row: row["environment"] + "|" + row["task"] + "|" + row["method"]),
        "definitions": {
            "decision_wall_seconds": "Observed saved decision-loop time: GPT update if scheduled, actor calls/cache lookup/waiting, scoring, and native action. Excludes reset, replay, reporting writes and pauses between attempts.",
            "native_step_seconds": "Observed game.step plus public observation/record preparation, as measured by the runner.",
            "policy_wall_seconds": "Measured action selection time before game.step; for older rows, decision_wall_seconds minus native_step_seconds. Includes orchestration and cache lookup, not pure model inference. Negative/missing fallback pairs remain unknown.",
            "goal_categories": "goal_update uses saved GPT goal step; goal_reuse uses a saved prior goal. Missing goal log makes the update category unknown. No semantic completion checker is used.",
            "coverage": "Completed action timing rows are retained even in failed/paused episodes. A request interrupted before a saved action has no action timing; ledger request cost/latency remains charged. Missing timing is never zero.",
            "percentiles": "p50 and p95 use linear interpolation over known nonnegative recorded action durations. They are observed timings, not full logical independent-run receipt sums.",
            "cache": "Observed action durations include actual cache behavior. Cache hit counts and uncached counterfactual action latency are not inferred from these records.",
        },
    }


def write_timing_report(out, episodes):
    """Write timing.json, timing.csv and timing-by-episode.csv; return summary."""
    out = Path(out)
    result = summarize_timing(out, episodes)
    write_json(out / "timing.json", result)
    fields = ("scope", "group", "category", "metric", "episodes", "complete_cohort",
              "retained_actions", "known_actions", "unknown_actions", "known_sum", "mean", "p50", "p95")

    def export(path, scopes):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for scope, groups in scopes:
                for label, summary in groups.items():
                    for category, action in summary["by_action_category"].items():
                        for metric in METRICS:
                            writer.writerow({"scope": scope, "group": label, "category": category,
                                             "metric": metric, "episodes": summary["episodes"],
                                             "complete_cohort": summary["complete_cohort"],
                                             "retained_actions": action["actions"], **action[metric]})
    export(out / "timing.csv", [("total", {"all": result["total"]}),
                               ("environment_method", result["by_environment_method"]),
                               ("task_method", result["by_task_method"])])
    export(out / "timing-by-episode.csv", [("episode", result["by_episode"])])
    return result
