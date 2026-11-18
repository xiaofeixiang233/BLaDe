"""Domain-independent probability guidance. No game semantics live here."""
from copy import deepcopy
import json
import math
from collections.abc import Mapping
from . import prompts

ACTORS = ("jev", "d3", "kev")
MAIN_METHODS = tuple(actor + suffix for actor in ACTORS for suffix in ("-flat-h0", "-w2", "-w05"))
ABLATION_METHODS = tuple(actor + suffix for actor in ("jev", "d3") for suffix in ("-history-h4", "-joint-h4"))
METHODS = MAIN_METHODS + ABLATION_METHODS
SHORT_BENCHMARKS = ("babyai", "minihack", "babaisai")
WEIGHTS = {actor + suffix: weight for actor in ACTORS for suffix, weight in (("-w2", 2.), ("-w05", .5))}
ROLES = ("pi0","ph","q","joint")
HISTORY_LIMIT = 4
WARMUP_ACTIONS = 8
EPSILON = 1e-6
TIME_FIELDS = ("remaining_episode_actions","remaining_goal_horizon","actions_taken")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def roles_for(method, step):
    if method not in METHODS:
        raise ValueError("unknown experiment arm")
    _integer(step, "step")
    if is_baseline(method):
        return ("pi0",)
    if method.endswith("-history-h4"):
        return ("ph",)
    if method.endswith("-joint-h4"):
        return ("ph",) if step < WARMUP_ACTIONS else ("joint",)
    return ("ph",) if step < WARMUP_ACTIONS else ("pi0", "ph", "q")


def is_baseline(method):
    return method.endswith("flat-h0")


def needs_goal(method):
    if method not in METHODS:
        raise ValueError("unknown method")
    return method in WEIGHTS or method.endswith("-joint-h4")


def methods_for_environment(cfg, name):
    if 'execution_scope' in cfg:
        return list(cfg['execution_scope'].get(name, []))
    result = list(MAIN_METHODS)
    if name in SHORT_BENCHMARKS:
        result.extend(method for method in ABLATION_METHODS
                      if actor_kind(method) in cfg.get("ablation_actors", ACTORS))
    return result


def actor_kind(method):
    if method not in METHODS:
        raise ValueError("unknown method")
    return method.split("-", 1)[0]


def effective_decision_cap(cfg, case):
    """Outer model decisions; the case's native engine horizon stays intact."""
    horizon = _integer(case["horizon"], "native horizon", 1)
    configured = cfg.get("decision_caps", {}).get(case["environment"], horizon)
    return min(horizon, _integer(configured, "configured decision cap", 1))


def goal_slots(decision_cap, max_goals):
    _integer(decision_cap, "decision_cap", 1)
    _integer(max_goals, "max_goals", 1)
    if decision_cap <= WARMUP_ACTIONS:
        return []
    return sorted({WARMUP_ACTIONS+j*(decision_cap-WARMUP_ACTIONS)//max_goals
                   for j in range(max_goals)})


def goal_time(decision_cap, step, generated_at=None, max_goals=4):
    _integer(decision_cap, "decision_cap", 1)
    _integer(step, "step")
    if step >= decision_cap:
        raise ValueError("no decision after effective decision cap")
    remaining = 0
    if generated_at is not None:
        slots = goal_slots(decision_cap, max_goals)
        if generated_at not in slots:
            raise ValueError("invalid fixed goal slot")
        end = next((x for x in slots if x > generated_at), decision_cap)
        if not generated_at <= step < end:
            raise ValueError("goal must be replaced at its next fixed slot")
        remaining = end-step
    return {"remaining_episode_actions": decision_cap-step,
            "remaining_goal_horizon": remaining, "actions_taken": step}


def _observation(value):
    # These are the native text adapters' public projections, not game-state
    # objects. A map with extra private keys is rejected, never copied through.
    if isinstance(value, str):
        return value
    if (isinstance(value, Mapping)
            and set(value) == {"long_term_context", "short_term_context"}
            and all(isinstance(v, str) for v in value.values())):
        return deepcopy(dict(value))
    raise ValueError("complete native text observation required")


def public_view(public):
    """Allow only native public task and observation, never experiment metadata."""
    if not isinstance(public, Mapping):
        raise ValueError("public must be a mapping")
    for field in ("environment", "mission"):
        if not isinstance(public.get(field), str):
            raise ValueError("native " + field + " string required")
    result = {"environment": public["environment"], "mission": public["mission"],
              "observation": _observation(public.get("observation"))}
    if "native_instruction" in public:
        if not isinstance(public["native_instruction"], str):
            raise ValueError("native instruction must be a string")
        result["native_instruction"] = public["native_instruction"]
    return result


def history_window(history):
    """Last four actual actions with their native public post-action text only."""
    if not isinstance(history, (list, tuple)):
        raise ValueError("history must be an ordered real-interaction sequence")
    result = []
    for record in history[-HISTORY_LIMIT:]:
        if not isinstance(record, Mapping) or not isinstance(record.get("action"), str):
            raise ValueError("executed native action required")
        after = record.get("after")
        if not isinstance(after, Mapping) or not isinstance(after.get("mission"), str):
            raise ValueError("native public post-action task required")
        result.append({"action": record["action"], "mission": after["mission"],
                       "observation": _observation(after.get("observation"))})
    return result


def _actions(actions):
    if (not isinstance(actions, (list, tuple)) or not 1 <= len(actions) <= 255
            or any(not isinstance(action, str) or not action for action in actions)
            or len(set(actions)) != len(actions)):
        raise ValueError("complete distinct native menu of 1..255 actions required")
    return tuple(actions)


def _criteria(public, actions):
    menu = _actions(actions)
    descriptions = public.get("action_descriptions")
    if (not isinstance(descriptions, Mapping) or set(descriptions) != set(menu)
            or not all(isinstance(value, str) for value in descriptions.values())):
        raise ValueError("complete native action descriptions required; no mask or truncation")
    return {action: descriptions[action] for action in menu}


def _time(time):
    if not isinstance(time, Mapping) or any(key not in time for key in TIME_FIELDS):
        raise ValueError("common episode and goal countdown fields required")
    result = {key: _integer(time[key], key) for key in TIME_FIELDS}
    if result["remaining_goal_horizon"] > result["remaining_episode_actions"]:
        raise ValueError("goal horizon exceeds remaining episode actions")
    return result


def _goal(goal):
    if isinstance(goal, str):
        return goal
    if (isinstance(goal, Mapping) and set(goal) == {"goal"}
            and isinstance(goal["goal"], str)):
        return goal["goal"]
    raise ValueError("goal must be a string or the strict single-string goal object")


def payload(role, public, actions, history, goal, time, model="d3"):
    """D3Client request with one action question and the full native menu.

    pi0: current public state; ph: plus H4; q: plus one goal. All roles share exactly the same instructions and time fields.
    Arguments withheld from a role are never inspected or serialized.
    """
    if role not in ROLES:
        raise ValueError("unknown D3 distribution role")
    state = public_view(public)
    state.update(_time(time))
    if role in ("ph", "joint"):
        state["previous_interactions"] = history_window(history)
    if role in ("q", "joint"):
        state["desired_state"] = _goal(goal)
    return {"model": model, "state": state, "questions": {"action": {
        "type": "choice", "instructions": prompts.ACTOR_INSTRUCTIONS,
        "criteria": _criteria(public, actions)}}}


def normalize_distribution(actions, distribution):
    menu = _actions(actions)
    if not isinstance(distribution, Mapping) or set(distribution) != set(menu):
        raise ValueError("distribution must cover the complete native menu")
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0
           for p in distribution.values()):
        raise ValueError("finite nonnegative action mass required")
    # Divide by the maximum first so finite but large masses cannot overflow.
    maximum = max(distribution.values())
    if maximum <= 0:
        raise ValueError("distribution has no positive mass")
    scaled = {action: distribution[action] / maximum for action in menu}
    total = math.fsum(scaled.values())
    return {action: scaled[action] / total for action in menu}


def smooth_distribution(actions, distribution):
    """Normalize, add epsilon/nActions to every action, then renormalize."""
    normalized = normalize_distribution(actions, distribution)
    floor = EPSILON / len(normalized)
    return {action: (mass + floor) / (1 + EPSILON)
            for action, mass in normalized.items()}


def log_action_scores(method, actions, *, ph, q, p0):
    if method not in WEIGHTS:
        raise ValueError("weighted method required")
    h=smooth_distribution(actions,ph); p=smooth_distribution(actions,q)
    b=smooth_distribution(actions,p0); w=WEIGHTS[method]
    return {a: math.log(h[a])+w*(math.log(p[a])-math.log(b[a])) for a in h}


def combine_distributions(method, actions, *, ph, q, p0):
    scores=log_action_scores(method,actions,ph=ph,q=q,p0=p0)
    maximum=max(scores.values())
    return normalize_distribution(actions,{a:math.exp(s-maximum) for a,s in scores.items()})


def choose_action(actions, distribution):
    normalized=normalize_distribution(actions,distribution)
    return max(actions,key=lambda a:normalized[a])


def goal_payload(public, actions, history, time, gpt_cfg):
    data=public_view(public)
    data["native_action_descriptions"]=_criteria(public,actions)
    data["previous_interactions"]=history_window(history)
    data.update(_time(time))
    return {"model":gpt_cfg["model"],"reasoning":{"effort":gpt_cfg["reasoning_effort"]},
            "max_output_tokens":gpt_cfg["max_output_tokens"],"service_tier":"default",
            "store":False,"truncation":"disabled","instructions":prompts.GOAL_INSTRUCTIONS,
            "input":json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(",",":")),
            "text":{"format":{"type":"json_schema","name":"native_desired_state_v1",
                     "strict":True,"schema":deepcopy(prompts.GOAL_SCHEMA)}}}
