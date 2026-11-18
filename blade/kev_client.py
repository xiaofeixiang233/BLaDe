"""One-attempt native Kev System One client with strict model provenance.

No prompts, menus, probability values, or actions are transformed here. Imports
are inert. Credentials are read only at explicit network dispatch; HTTP GET
health requests perform no model inference and never count as action requests.
"""
from copy import deepcopy
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .kev_model import (REPO, REVISION, SOURCE_REVISION, HEAD_SHA256, MODEL_NAME,
                        FORMAT_ID, TEMPERATURE, MAX_STATE_TOKENS, MAX_ROW_TOKENS)


def _error(message, **kwargs):
    # providers delegates Kev validation here: keep this dependency lazy.
    from .providers import ProviderError
    return ProviderError(message, **kwargs)


def _endpoint(value):
    parsed = urllib.parse.urlsplit(str(value))
    if (not isinstance(value, str) or parsed.scheme not in ("http", "https")
            or not parsed.hostname or parsed.path != "/v1/systemone"
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise _error("explicit Kev System One endpoint required", dispatched=False)
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise _error("non-loopback Kev requires HTTPS", dispatched=False)
    return value


def kev_endpoint_pool(settings):
    pool = settings.get("endpoints", [settings.get("endpoint")])
    if (not isinstance(pool, list) or not pool or any(not isinstance(x, str) for x in pool)
            or len(pool) != len(set(pool))):
        raise _error("Kev endpoints must be a distinct nonempty declared list", dispatched=False)
    for endpoint in pool:
        _endpoint(endpoint)
    if settings.get("endpoint") not in pool:
        raise _error("effective Kev endpoint outside declared pool", dispatched=False)
    return deepcopy(pool)


def _credentials_in(value):
    if isinstance(value, dict):
        return any(str(k).lower() in {"key", "keys", "api_key", "api_keys", "secret", "token",
                   "access_token", "authorization", "password", "credentials"} or _credentials_in(v)
                   for k, v in value.items())
    return isinstance(value, list) and any(_credentials_in(x) for x in value)


def _fixed_identity():
    return {"repo_id": REPO, "revision": REVISION, "source_revision": SOURCE_REVISION,
            "head_sha256": HEAD_SHA256, "format_id": FORMAT_ID,
            "dtype": "bfloat16", "backend": "torch", "temperature_published": TEMPERATURE,
            "date_facts": False, "truncate_states": False, "cuda_graphs": False,
            "fused": False, "weights": "full", "verification": "all_manifest_files_verified",
            "max_options": 255, "max_state_tokens": MAX_STATE_TOKENS,
            "max_row_tokens": MAX_ROW_TOKENS}


def settings_for_kev(value):
    """Validate an explicit Kev provider object; never substitute D3 or hosted Jev."""
    from .providers import dumps
    if not isinstance(value, dict) or _credentials_in(value):
        raise _error("explicit Kev settings without embedded credentials required", dispatched=False)
    value = deepcopy(value)
    try:
        dumps(value)
    except (TypeError, ValueError):
        raise _error("Kev settings must be finite JSON", dispatched=False) from None
    fixed = {"model": MODEL_NAME, "repo_id": REPO, "revision": REVISION,
             "source_revision": SOURCE_REVISION, "head_sha256": HEAD_SHA256,
             "max_input_tokens": MAX_STATE_TOKENS}
    if any(value.get(k) != v for k, v in fixed.items()):
        raise _error("frozen Kev checkpoint identity changed", dispatched=False)
    timeout = value.get("timeout_seconds")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise _error("Kev timeout must be finite and positive", dispatched=False)
    kev_endpoint_pool(value)
    extra = value.get("runtime_identity", {})
    if not isinstance(extra, dict) or any(k in extra and extra[k] != v for k, v in _fixed_identity().items()):
        raise _error("Kev runtime identity cannot override released inference", dispatched=False)
    return value


def extract_kev_identity(models_response):
    """Extract a native card plus verification metadata added by our serving wrapper."""
    if not isinstance(models_response, dict) or models_response.get("error") is not None:
        raise _error("Kev model-card response invalid", raw=models_response, dispatched=False)
    cards = models_response.get("models")
    if not isinstance(cards, list) or any(not isinstance(x, dict) for x in cards):
        raise _error("Kev native /v1/models list missing", raw=models_response, dispatched=False)
    matches = [x for x in cards if x.get("name") == MODEL_NAME]
    if len(matches) != 1:
        raise _error("one native kev-latest model card required", raw=models_response, dispatched=False)
    identity = models_response.get("runtime_identity")
    if not isinstance(identity, dict):
        raise _error("Kev verification identity missing; use kev_model serve wrapper", raw=models_response,
                     dispatched=False)
    return {"model_card": deepcopy(matches[0]), "runtime_identity": deepcopy(identity)}


def validate_kev_identity(identity, settings):
    """Check the code/head/checkpoint and no-truncation contract without inference."""
    settings = settings_for_kev(settings)
    if not isinstance(identity, dict):
        raise _error("Kev identity object missing", dispatched=False)
    card, runtime = identity.get("model_card"), identity.get("runtime_identity")
    if not isinstance(card, dict) or not isinstance(runtime, dict):
        raise _error("Kev card/verification identity missing", raw=identity, dispatched=False)
    required = _fixed_identity()
    required.update(settings.get("runtime_identity", {}))
    if any(runtime.get(k) != v or type(v) is bool and type(runtime.get(k)) is not bool
           for k, v in required.items()):
        raise _error("Kev runtime identity mismatch", raw=identity, dispatched=False)
    actual = card.get("temperature")
    # Head SHA pins the exact fit; 1.32 is the two-decimal figure in its model card.
    if (type(actual) not in (int, float) or not math.isfinite(actual) or round(actual, 2) != TEMPERATURE
            or runtime.get("temperature") != actual):
        raise _error("Kev native head calibration changed", raw=identity, dispatched=False)
    if (card.get("name") != MODEL_NAME or card.get("base") != "Qwen/Qwen3.8-27B"
            or card.get("lora") != 0 or card.get("dtype") != "bfloat16"
            or card.get("backend") != "torch" or card.get("device") != "cuda"
            or card.get("truncate_states") is not False
            or card.get("max_state_tokens") != MAX_STATE_TOKENS
            or not isinstance(card.get("run"), str) or not card["run"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("checkpoint_manifest_sha256", "")))):
        raise _error("Kev native card does not match verified BF16 v2 server", raw=identity, dispatched=False)
    versions = runtime.get("versions")
    if (not isinstance(versions, dict)
            or any(not isinstance(versions.get(k), str) or not versions[k]
                   for k in ("torch", "transformers", "accelerate", "safetensors", "huggingface-hub"))):
        raise _error("Kev inference package versions missing", raw=identity, dispatched=False)
    return deepcopy(identity)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class KevClient:
    """Native request shape, optional transport injection, no automatic retries."""
    def __init__(self, settings, transport=None):
        self.settings = settings_for_kev(settings)
        self.transport = transport
        self._mutex = threading.Lock()
        self._checked = False
        self._identity_by_endpoint = {}

    def preflight(self, payload):
        from .providers import validate_choice_payload
        validate_choice_payload(payload, MODEL_NAME)
        return deepcopy(self.settings)

    def _http(self, endpoint, payload):
        from .providers import Receipt, dumps, strict_json, redact, MAX_BODY_BYTES
        if self.transport is not None:
            start = time.monotonic()
            result = self.transport("kev", endpoint, deepcopy(payload))
            return result if isinstance(result, Receipt) else Receipt(result, dumps(result), time.monotonic() - start)
        key = os.environ.get("KEV_API_KEY", "")
        if len(key) > 16384 or any(c.isspace() for c in key):
            raise _error("KEV_API_KEY must contain only one runtime credential", dispatched=False)
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
        start, body = time.monotonic(), None
        try:
            with opener.open(request, timeout=self.settings["timeout_seconds"]) as response:
                content = response.read(MAX_BODY_BYTES + 1)
            body = redact(content.decode("utf-8", errors="replace"), (key,))
            if len(content) > MAX_BODY_BYTES:
                raise _error("Kev response body limit exceeded", raw_body=body)
            return Receipt(strict_json(body), body, time.monotonic() - start)
        except urllib.error.HTTPError as exc:
            body = redact(exc.read(MAX_BODY_BYTES + 1).decode("utf-8", errors="replace"), (key,))
            try:
                raw = strict_json(body)
            except (ValueError, TypeError):
                raw = None
            raise _error("Kev HTTP " + str(exc.code), raw=raw, raw_body=body) from None
        except (ValueError, UnicodeError):
            raise _error("Kev response is not strict JSON", raw_body=body) from None
        except OSError:
            raise _error("Kev transport failure", raw_body=body) from None

    def health(self, endpoint=None):
        endpoint = endpoint or self.settings["endpoint"]
        if endpoint not in kev_endpoint_pool(self.settings):
            raise _error("Kev health endpoint outside declared pool", dispatched=False)
        parsed = urllib.parse.urlsplit(endpoint)
        url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/v1/models", "", ""))
        from .providers import ProviderError
        try:
            receipt = self._http(url, None)
        except ProviderError as exc:
            exc.dispatched = False  # A failed model-card GET dispatched no inference.
            raise
        identity = validate_kev_identity(extract_kev_identity(receipt.raw), self.settings)
        # A running endpoint cannot silently change its checkpoint or kernels on
        # resume; deployment counters in the model card do not affect identity.
        pinned = self._identity_by_endpoint.get(endpoint)
        if pinned is not None and pinned != identity["runtime_identity"]:
            raise _error("Kev runtime changed after first health check", raw=receipt.raw,
                         raw_body=receipt.raw_body, dispatched=False)
        self._identity_by_endpoint[endpoint] = deepcopy(identity["runtime_identity"])
        return identity

    def health_pool(self):
        result = {endpoint: self.health(endpoint) for endpoint in kev_endpoint_pool(self.settings)}
        identities = [x["runtime_identity"] for x in result.values()]
        if identities and any(x != identities[0] for x in identities[1:]):
            raise _error("Kev replicas use different checkpoints or inference settings", raw=result, dispatched=False)
        return result

    def call(self, payload):
        self.preflight(payload)
        with self._mutex:
            if not self._checked:
                self.health_pool()
                self._checked = True
            else:
                self.health()
        receipt = self._http(self.settings["endpoint"], payload)
        raw = receipt.raw
        if isinstance(raw, dict) and "truncated" in raw:
            raise _error("Kev truncating server response retained; no dispatch retry",
                         raw=raw, raw_body=receipt.raw_body)
        return receipt
