"""One-attempt HTTP transports and strict response contracts.

This module does not inspect a game, score a goal, filter a menu, or select an
action. Importing it does not read credentials, contact a provider, or load D3.
Credentials are read from process environment variables only at dispatch time.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


GPT_ENDPOINT = "https://api.openai.com/v1/responses"
GPT_MODEL = "gpt-6-sol"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"
D3_REPO = "vllm-sr/d3"
D3_REVISION = "cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75"
D3_MODEL_SHA256 = "0e439bebbbbf80d9e1b601c56c55a6fc3a6502b4e396cbbb5f3c3f3acc1d607b"
D3_FORMAT = "d3-code-readout-v1"
DEPLOYMENT_IDENTITY_FIELDS = {"physical_devices", "replica", "replica_id", "replicas",
                              "device", "device_map", "endpoint", "cuda_visible_devices"}
MAX_BODY_BYTES = 32 * 1024 * 1024
MASS_TOLERANCE = 0.025
GOAL_SCHEMA = {"type": "object", "properties": {"goal": {"type": "string"}},
               "required": ["goal"], "additionalProperties": False}


class ProviderError(RuntimeError):
    """A fixed, redacted error code and an optional retained provider receipt."""
    def __init__(self, message, *, raw=None, raw_body=None, dispatched=True):
        super().__init__(message)
        self.raw = raw
        self.raw_body = raw_body
        self.dispatched = dispatched


@dataclass
class Receipt:
    raw: object
    raw_body: str | None = None
    latency_seconds: float | None = None


def dumps(value, *, canonical=False):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=canonical)


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def constant(_):
        raise ValueError("nonfinite JSON value")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def redact(value, secrets=()):
    """Redact credential-shaped strings without adding credentials to logs."""
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)bearer\s+[^\s\"',;}]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{10,}|apikey_[A-Za-z0-9_-]{10,})\b",
                  "[REDACTED]", text)
    return text


def redact_runtime(value):
    """Use runtime-only credentials to redact exceptional receipts as well."""
    return redact(value, tuple(os.environ.get(name, "") for name in
                               ("OPENAI_API_KEY", "JEV_KEYS", "D3_API_KEY", "KEV_API_KEY")))


def _positive_number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ProviderError(field + " must be finite and positive", dispatched=False)


def _positive_integer(value, field):
    if type(value) is not int or value <= 0:
        raise ProviderError(field + " must be a positive integer", dispatched=False)


def _d3_endpoint(endpoint):
    parsed = urllib.parse.urlsplit(str(endpoint))
    if (not isinstance(endpoint, str) or parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.path != "/v1/systemone" or parsed.query or parsed.fragment
            or parsed.username or parsed.password):
        raise ProviderError("explicit D3 systemone endpoint required", dispatched=False)
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ProviderError("non-loopback D3 requires HTTPS", dispatched=False)
    return endpoint


def d3_endpoint_pool(settings):
    """A declared, ordered deployment pool; arbitrary worker URLs are forbidden."""
    values = settings.get("endpoints", [settings.get("endpoint")])
    if (not isinstance(values, list) or not values or any(not isinstance(item, str) for item in values)
            or len(set(values)) != len(values)):
        raise ProviderError("D3 endpoints must be a distinct nonempty declared list", dispatched=False)
    for endpoint in values:
        _d3_endpoint(endpoint)
    if settings.get("endpoint") not in values:
        raise ProviderError("effective D3 endpoint is outside the declared pool", dispatched=False)
    return deepcopy(values)


def settings_for(cfg, kind):
    """Validate explicit endpoints; never guess a hosted model or provider."""
    if kind not in ("jev", "d3", "kev", "gpt"):
        raise ProviderError("unknown provider kind", dispatched=False)
    value = deepcopy(cfg.get("providers", {}).get(kind))
    if not isinstance(value, dict):
        raise ProviderError("explicit " + kind + " provider configuration required", dispatched=False)
    def credential_field(item):
        if isinstance(item, dict):
            return any(str(key).lower() in {"key", "keys", "api_key", "api_keys", "token",
                       "access_token", "secret", "authorization", "password", "credentials"}
                       or credential_field(child) for key, child in item.items())
        return isinstance(item, list) and any(credential_field(child) for child in item)
    if credential_field(value):
        raise ProviderError("credentials must stay in runtime environment, not provider configuration", dispatched=False)
    dumps(value)  # Reject nonfinite or nonserializable frozen identity values.
    _positive_number(value.get("timeout_seconds"), "timeout_seconds")
    if kind == "kev":
        from .kev_client import settings_for_kev
        return settings_for_kev(value)
    if kind == "jev":
        if value.get("model") != JEV_MODEL or value.get("endpoint") != JEV_ENDPOINT:
            raise ProviderError("frozen official Jev endpoint/model changed", dispatched=False)
        _positive_integer(value.get("max_input_tokens"), "max_input_tokens")
    elif kind == "gpt":
        if (value.get("model") != GPT_MODEL or value.get("endpoint") != GPT_ENDPOINT
                or value.get("reasoning_effort", "medium") != "medium"
                or value.get("service_tier", "default") != "default"):
            raise ProviderError("frozen official Sol medium endpoint/settings changed", dispatched=False)
        _positive_integer(value.get("max_prompt_tokens"), "max_prompt_tokens")
        _positive_integer(value.get("max_output_tokens"), "max_output_tokens")
        if value["max_output_tokens"] != 4096:
            raise ProviderError("frozen GPT output limit changed", dispatched=False)
    else:
        d3_endpoint_pool(value)
        required = {"model": "d3", "repo_id": D3_REPO, "revision": D3_REVISION,
                    "model_sha256": D3_MODEL_SHA256, "max_input_tokens": 32768}
        if any(value.get(key) != item for key, item in required.items()):
            raise ProviderError("frozen D3 model identity changed", dispatched=False)
        extra = value.get("runtime_identity", {})
        pinned = {"repo_id": D3_REPO, "revision": D3_REVISION,
                  "model_sha256": D3_MODEL_SHA256, "format_id": D3_FORMAT}
        if (not isinstance(extra, dict)
                or any(key in extra and extra[key] != item for key, item in pinned.items())):
            raise ProviderError("D3 runtime identity cannot override pinned fields", dispatched=False)
    if kind in ("jev", "gpt"):
        prices = value.get("prices_per_million")
        expected = {"input", "output"} if kind == "jev" else {"input", "cached_input", "output"}
        if (not isinstance(prices, dict) or set(prices) != expected
                or any(type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0
                       for rate in prices.values())):
            raise ProviderError("explicit finite provider token prices required", dispatched=False)
    return value


def validate_choice_payload(payload, expected_model=None):
    if (not isinstance(payload, dict) or set(payload) != {"model", "state", "questions"}
            or not isinstance(payload.get("state"), dict)
            or not isinstance(payload.get("model"), str)
            or expected_model is not None and payload["model"] != expected_model):
        raise ProviderError("complete choice request fields/model required", dispatched=False)
    questions = payload["questions"]
    if not isinstance(questions, dict) or len(questions) != 1:
        raise ProviderError("one complete native action question required", dispatched=False)
    name, question = next(iter(questions.items()))
    if (not isinstance(name, str) or not name or not isinstance(question, dict)
            or set(question) != {"type", "instructions", "criteria"}
            or question.get("type") != "choice"
            or not isinstance(question.get("instructions"), str) or not question["instructions"]
            or not isinstance(question.get("criteria"), dict)):
        raise ProviderError("choice question schema invalid", dispatched=False)
    options = question["criteria"]
    if (not 1 <= len(options) <= 255
            or any(not isinstance(key, str) or not key or not isinstance(text, str) or not text
                   for key, text in options.items())):
        raise ProviderError("complete native menu outside choice capacity", dispatched=False)
    dumps(payload)
    return list(options)


def parse_distribution(raw, payload):
    """Validate a full answer without repairing, clipping, or adding actions.

    The returned values are exactly the provider probabilities. The policy may
    perform its separately documented numeric normalization and guidance.
    """
    options = validate_choice_payload(payload)
    name = next(iter(payload["questions"]))
    if (not isinstance(raw, dict) or raw.get("model") != payload["model"]
            or raw.get("error") is not None):
        raise ProviderError("choice response model/error mismatch", raw=raw)
    answers = raw.get("answers")
    if not isinstance(answers, dict) or list(answers) != [name]:
        raise ProviderError("choice response question mismatch", raw=raw)
    answer = answers[name]
    if (not isinstance(answer, dict) or answer.get("type") != "choice"
            or "error" in answer or answer.get("choice") not in options):
        raise ProviderError("choice response invalid answer", raw=raw)
    values = answer.get("probabilities")
    if (not isinstance(values, dict) or set(values) != set(options)
            or any(type(probability) not in (int, float) or not math.isfinite(probability)
                   or not 0 <= probability <= 1 for probability in values.values())):
        raise ProviderError("choice response full distribution invalid", raw=raw)
    mass = math.fsum(values.values())
    if mass <= 0 or abs(mass - 1.0) > MASS_TOLERANCE + 1e-12:
        raise ProviderError("choice response probability mass invalid", raw=raw)
    # Operator-approved: Jev's `choice` may disagree with its own maximum. Policies use
    # only the distribution; the raw response (with choice) stays in the ledger.
    if "confidence" in answer and (type(answer["confidence"]) not in (int, float)
            or not math.isfinite(answer["confidence"]) or not 0 <= answer["confidence"] <= 1):
        raise ProviderError("choice response confidence invalid", raw=raw)
    return {action: values[action] for action in options}


def extract_goal(raw, expected_model=GPT_MODEL):
    if (not isinstance(raw, dict) or raw.get("status") != "completed"
            or raw.get("model") != expected_model or raw.get("error") is not None
            or raw.get("incomplete_details") is not None):
        raise ProviderError("GPT incomplete or returned model changed", raw=raw)
    if not isinstance(raw.get("output"), list):
        raise ProviderError("GPT official output missing", raw=raw)
    texts = []
    for item in raw["output"]:
        if not isinstance(item, dict):
            raise ProviderError("GPT malformed output item", raw=raw)
        if item.get("type") == "reasoning":
            continue
        if (item.get("type") != "message" or item.get("role") != "assistant"
                or item.get("status") != "completed" or not isinstance(item.get("content"), list)):
            raise ProviderError("GPT unexpected output item", raw=raw)
        for content in item["content"]:
            if (not isinstance(content, dict) or content.get("type") != "output_text"
                    or not isinstance(content.get("text"), str)):
                raise ProviderError("GPT refusal or malformed goal output", raw=raw)
            texts.append(content["text"])
    if len(texts) != 1:
        raise ProviderError("GPT must produce one goal object", raw=raw)
    try:
        goal = strict_json(texts[0])
    except (ValueError, TypeError):
        raise ProviderError("GPT goal is not strict JSON", raw=raw) from None
    if (not isinstance(goal, dict) or set(goal) != {"goal"}
            or not isinstance(goal["goal"], str)):
        raise ProviderError("GPT goal schema mismatch", raw=raw)
    return goal


def validate_goal_payload(payload, settings):
    expected = {"model", "input", "instructions", "reasoning", "service_tier",
                "max_output_tokens", "store", "text", "truncation"}
    if (not isinstance(payload, dict) or set(payload) != expected
            or payload.get("model") != settings["model"]
            or not isinstance(payload.get("input"), str)
            or not isinstance(payload.get("instructions"), str) or not payload["instructions"]
            or payload.get("reasoning") != {"effort": "medium"}
            or payload.get("service_tier") != "default"
            or payload.get("max_output_tokens") != settings["max_output_tokens"]
            or payload.get("store") is not False or payload.get("truncation") != "disabled"):
        raise ProviderError("frozen Responses goal request fields changed", dispatched=False)
    fmt = payload.get("text", {}).get("format") if isinstance(payload.get("text"), dict) else None
    if (not isinstance(fmt, dict) or set(fmt) != {"type", "name", "strict", "schema"}
            or fmt.get("type") != "json_schema" or fmt.get("strict") is not True
            or not isinstance(fmt.get("name"), str) or not fmt["name"]
            or fmt.get("schema") != GOAL_SCHEMA or set(payload["text"]) != {"format"}):
        raise ProviderError("GPT must use strict one-string goal schema", dispatched=False)
    dumps(payload)


def usage_and_cost(kind, raw, settings):
    """Known token-price estimates; missing usage/cost remains None, never zero."""
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return {"input_tokens": None, "output_tokens": None, "cached_input_tokens": None,
                "reasoning_tokens": None, "cost_usd": None, "billing_known": False}
    def token(value):
        return value if type(value) is int and value >= 0 else None
    inp, output = token(usage.get("input_tokens")), token(usage.get("output_tokens"))
    details = usage.get("input_tokens_details", {})
    cached = token(details.get("cached_tokens", 0)) if isinstance(details, dict) else None
    details = usage.get("output_tokens_details", {})
    reasoning = token(details.get("reasoning_tokens", 0)) if isinstance(details, dict) else None
    result = {"input_tokens": inp, "output_tokens": output, "cached_input_tokens": cached,
              "reasoning_tokens": reasoning, "cost_usd": None, "billing_known": False}
    if kind in ("d3", "kev"):
        # Local GPU time is unpriced. Do not represent it as a zero-dollar model.
        return result
    if not isinstance(raw, dict) or raw.get("model") != settings["model"]:
        return result
    prices = settings["prices_per_million"]
    if kind == "jev":
        if inp is None or (output is None and prices["output"] != 0):
            return result
        result.update(cost_usd=(inp * prices["input"] + (output or 0) * prices["output"]) / 1e6,
                      billing_known=True)
        return result
    if (raw.get("service_tier") not in (None, "default")
            or any(value is None for value in (inp, output, cached, reasoning))
            or cached > inp or reasoning > output):
        return result
    result.update(cost_usd=((inp - cached) * prices["input"] + cached * prices["cached_input"]
                           + output * prices["output"]) / 1e6, billing_known=True)
    return result


def token_proxy(payload):
    """Conservative proxy, not a provider usage receipt or tokenizer claim."""
    try:
        import tiktoken
    except ImportError:
        raise ProviderError("install tiktoken for offline prompt-bound checks", dispatched=False) from None
    return math.ceil(len(tiktoken.get_encoding("o200k_base").encode(
        dumps(payload), disallowed_special=())) * 1.4) + 512


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Providers:
    """Explicit HTTP endpoints, no automatic retries or authentication fallback."""
    def __init__(self, cfg, transport=None):
        self.cfg = deepcopy(cfg)
        self.transport = transport
        self._pool_checked = False
        self._pool_mutex = threading.Lock()
        self._kev_client = None

    def kev_client(self):
        with self._pool_mutex:
            if self._kev_client is None:
                from .kev_client import KevClient
                self._kev_client = KevClient(settings_for(self.cfg, 'kev'), transport=self.transport)
            return self._kev_client

    def kev_health_pool(self):
        return self.kev_client().health_pool()

    def preflight(self, kind, payload):
        settings = settings_for(self.cfg, kind)
        if kind == "gpt":
            validate_goal_payload(payload, settings)
            if token_proxy(payload) > settings["max_prompt_tokens"]:
                raise ProviderError("GPT prompt bound exceeded; no truncation or dispatch", dispatched=False)
        else:
            validate_choice_payload(payload, settings["model"])
            # D3's pinned server performs the exact tokenizer/context check.
            if kind == "jev" and token_proxy(payload) > settings["max_input_tokens"]:
                raise ProviderError("Jev prompt bound exceeded; no truncation or dispatch", dispatched=False)
        return settings

    def _credential(self, kind):
        field = {"gpt": "OPENAI_API_KEY", "jev": "JEV_KEYS", "d3": "D3_API_KEY", "kev": "KEV_API_KEY"}[kind]
        key = os.environ.get(field, "")
        if kind in ("d3", "kev") and not key:
            return None
        if not key or len(key) > 16384 or any(character.isspace() for character in key):
            raise ProviderError(field + " must contain only one credential", dispatched=False)
        return key

    def _http(self, kind, endpoint, payload, settings):
        if self.transport is not None:
            start = time.monotonic()
            result = self.transport(kind, endpoint, deepcopy(payload))
            return result if isinstance(result, Receipt) else Receipt(result, dumps(result), time.monotonic() - start)
        key = self._credential(kind)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        data = dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(endpoint, data=data, headers=headers,
                                         method="POST" if payload is not None else "GET")
        parsed = urllib.parse.urlsplit(endpoint)
        handlers = [_NoRedirect()]
        if parsed.hostname in ("127.0.0.1", "localhost", "::1"):
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        start = time.monotonic()
        body = None
        try:
            with opener.open(request, timeout=settings["timeout_seconds"]) as response:
                content = response.read(MAX_BODY_BYTES + 1)
            body = redact(content.decode("utf-8", errors="replace"), (key,))
            if len(content) > MAX_BODY_BYTES:
                raise ProviderError("provider response body limit exceeded", raw_body=body)
            return Receipt(strict_json(body), body, time.monotonic() - start)
        except urllib.error.HTTPError as exc:
            body = redact(exc.read(MAX_BODY_BYTES + 1).decode("utf-8", errors="replace"), (key,))
            try:
                raw = strict_json(body)
            except (ValueError, TypeError):
                raw = None
            raise ProviderError(kind + " HTTP " + str(exc.code), raw=raw, raw_body=body) from None
        except (ValueError, UnicodeError):
            raise ProviderError(kind + " response is not strict JSON", raw_body=body) from None
        except OSError:
            raise ProviderError(kind + " transport failure", raw_body=body) from None

    def health(self, endpoint=None):
        settings = settings_for(self.cfg, "d3")
        if endpoint is not None:
            if endpoint not in d3_endpoint_pool(settings):
                raise ProviderError("D3 health endpoint outside the declared pool", dispatched=False)
            settings["endpoint"] = endpoint
        endpoint = urllib.parse.urlsplit(settings["endpoint"])
        url = urllib.parse.urlunsplit((endpoint.scheme, endpoint.netloc, "/health", "", ""))
        receipt = self._http("d3", url, None, settings)
        raw = receipt.raw
        identity = raw.get("runtime_identity") if isinstance(raw, dict) else None
        required = {"repo_id": D3_REPO, "revision": D3_REVISION,
                    "model_sha256": D3_MODEL_SHA256, "format_id": D3_FORMAT}
        required.update({key: item for key, item in settings.get("runtime_identity", {}).items()
                         if key not in DEPLOYMENT_IDENTITY_FIELDS})
        if (not isinstance(raw, dict) or raw.get("status") != "ready" or raw.get("model") != "d3"
                or raw.get("max_input_tokens") != settings["max_input_tokens"]
                or not isinstance(identity, dict)
                or any(identity.get(key) != item for key, item in required.items())):
            raise ProviderError("D3 health pinned runtime identity mismatch", raw=raw,
                                raw_body=receipt.raw_body, dispatched=False)
        return deepcopy(raw)

    def health_pool(self):
        """Check every declared replica against the same pinned computation."""
        settings = settings_for(self.cfg, "d3")
        return {endpoint: self.health(endpoint) for endpoint in d3_endpoint_pool(settings)}

    def call(self, kind, payload):
        settings = self.preflight(kind, payload)
        if kind == "kev":
            return self.kev_client().call(payload)
        if kind == "d3":
            with self._pool_mutex:
                if not self._pool_checked:
                    self.health_pool()
                    self._pool_checked = True
                else:
                    self.health()
        return self._http(kind, settings["endpoint"], payload, settings)
