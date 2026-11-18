"""Offline identity/transport tests: no weights, CUDA, secrets, or model requests."""
from copy import deepcopy
import tempfile
import hashlib
import json
import unittest
from unittest.mock import patch
from pathlib import Path

from blade import kev_client, kev_model
from blade.providers import ProviderError, Receipt, parse_distribution


def settings():
    return {"model": kev_model.MODEL_NAME, "repo_id": kev_model.REPO,
            "revision": kev_model.REVISION, "source_revision": kev_model.SOURCE_REVISION,
            "head_sha256": kev_model.HEAD_SHA256,
            "endpoint": "http://127.0.0.1:8906/v1/systemone",
            "endpoints": ["http://127.0.0.1:8906/v1/systemone"],
            "timeout_seconds": 300, "max_input_tokens": 65536,
            "runtime_identity": {"format_id": kev_model.FORMAT_ID}}


def card_response():
    runtime = kev_client._fixed_identity()
    runtime.update(temperature=1.32, checkpoint_manifest_sha256="a" * 64,
                   versions={k: "test-version" for k in
                             ("torch", "transformers", "accelerate", "safetensors", "huggingface-hub")})
    return {"models": [{"name": "kev-latest", "run": "/operator/local/kev-27b",
                         "base": "Qwen/Qwen3.8-27B", "lora": 0,
                         "device": "cuda", "backend": "torch", "dtype": "bfloat16",
                         "temperature": 1.32, "max_state_tokens": 65536,
                         "truncate_states": False}], "runtime_identity": runtime}


def payload():
    return {"model": "kev-latest", "state": {"observation": "test public observation"},
            "questions": {"action": {"type": "choice", "instructions": "Choose one action.",
                                       "criteria": {"a": "Action a", "b": "Action b"}}}}


def answer():
    return {"model": "kev-latest", "answers": {"action": {"type": "choice", "choice": "a",
                     "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.4}},
            "usage": {"input_tokens": 123, "output_tokens": 48}, "latency_ms": 1.5}


class KevClientTests(unittest.TestCase):
    def test_exact_release_identity_and_no_embedded_keys(self):
        self.assertEqual(kev_client.settings_for_kev(settings()), settings())
        for key, value in (("repo_id", "vllm-sr/d3"), ("revision", "main"),
                           ("source_revision", "main"), ("head_sha256", "0" * 64),
                           ("model", "d3"), ("timeout_seconds", float("nan"))):
            cfg = settings(); cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ProviderError):
                kev_client.settings_for_kev(cfg)
        cfg = settings(); cfg["api_key"] = "test-only"
        with self.assertRaises(ProviderError):
            kev_client.settings_for_kev(cfg)
        cfg = settings(); cfg["runtime_identity"]["date_facts"] = True
        with self.assertRaises(ProviderError):
            kev_client.settings_for_kev(cfg)

    def test_safe_declared_endpoint_pool(self):
        for endpoint in ("http://external.example/v1/systemone", "http://127.0.0.1:8906/health",
                         "http://name:password@127.0.0.1:8906/v1/systemone",
                         "http://127.0.0.1:8906/v1/systemone?token=forbidden"):
            cfg = settings(); cfg["endpoint"] = endpoint; cfg["endpoints"] = [endpoint]
            with self.subTest(endpoint=endpoint), self.assertRaises(ProviderError):
                kev_client.settings_for_kev(cfg)
        cfg = settings(); cfg["endpoints"] *= 2
        with self.assertRaises(ProviderError):
            kev_client.settings_for_kev(cfg)

    def test_native_models_health_has_no_inference(self):
        calls = []
        def transport(kind, endpoint, value):
            calls.append((kind, endpoint, value)); return card_response()
        client = kev_client.KevClient(settings(), transport=transport)
        identity = client.health()
        self.assertEqual(identity["runtime_identity"]["head_sha256"], kev_model.HEAD_SHA256)
        self.assertEqual(calls, [("kev", "http://127.0.0.1:8906/v1/models", None)])

    def test_no_bare_name_or_d3_substitution(self):
        variants = []
        x = card_response(); x.pop("runtime_identity"); variants.append(x)
        x = card_response(); x["models"][0]["name"] = "d3"; variants.append(x)
        x = card_response(); x["models"].append(deepcopy(x["models"][0])); variants.append(x)
        x = card_response(); x["runtime_identity"]["repo_id"] = "vllm-sr/d3"; variants.append(x)
        x = card_response(); x["runtime_identity"]["source_revision"] = "main"; variants.append(x)
        x = card_response(); x["runtime_identity"]["head_sha256"] = "0" * 64; variants.append(x)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ProviderError):
                kev_client.validate_kev_identity(kev_client.extract_kev_identity(value), settings())

    def test_calibration_dtype_context_and_preprocessing_frozen(self):
        cases = [("model_card", "temperature", 1.0), ("model_card", "dtype", "float16"),
                 ("model_card", "device", "cpu"), ("model_card", "truncate_states", True),
                 ("model_card", "max_state_tokens", 8192),
                 ("runtime_identity", "date_facts", True), ("runtime_identity", "fused", True),
                 ("runtime_identity", "versions", {}), ("runtime_identity", "temperature", 1.0)]
        base = kev_client.extract_kev_identity(card_response())
        for part, key, value in cases:
            identity = deepcopy(base); identity[part][key] = value
            with self.subTest(part=part, key=key), self.assertRaises(ProviderError):
                kev_client.validate_kev_identity(identity, settings())
        # The fitted value in head.pt is preserved, not reset to a rounded value.
        base["model_card"]["temperature"] = base["runtime_identity"]["temperature"] = 1.321
        self.assertEqual(kev_client.validate_kev_identity(base, settings())["runtime_identity"]["temperature"], 1.321)

    def test_request_and_native_distribution_are_not_rewritten(self):
        calls = []
        def transport(kind, endpoint, value):
            calls.append((kind, endpoint, value))
            return card_response() if value is None else answer()
        client = kev_client.KevClient(settings(), transport=transport)
        original = payload(); result = client.call(original)
        self.assertIsInstance(result, Receipt)
        self.assertEqual([c for c in calls if c[2] is not None], [("kev", settings()["endpoint"], original)])
        self.assertEqual(result.raw, answer())
        self.assertEqual(parse_distribution(result.raw, original), {"a": 0.7, "b": 0.3})
        self.assertEqual(original, payload())

    def test_context_rejection_or_interface_failure_not_retried(self):
        posts = []
        def transport(kind, endpoint, value):
            if value is None: return card_response()
            posts.append(value)
            raise ProviderError("Kev HTTP 422", raw={"detail": "state too long"})
        with self.assertRaises(ProviderError):
            kev_client.KevClient(settings(), transport=transport).call(payload())
        self.assertEqual(len(posts), 1)

    def test_truncation_response_retained_and_no_second_post(self):
        posts = []
        def transport(kind, endpoint, value):
            if value is None: return card_response()
            posts.append(value); raw = answer(); raw["truncated"] = False; return raw
        with self.assertRaises(ProviderError) as caught:
            kev_client.KevClient(settings(), transport=transport).call(payload())
        self.assertFalse(caught.exception.raw["truncated"])
        self.assertEqual(len(posts), 1)

    def test_runtime_changes_rejected_before_action_dispatch(self):
        calls = []
        def transport(kind, endpoint, value):
            calls.append(value)
            if value is None:
                response = card_response()
                if len(calls) > 2: response["runtime_identity"]["versions"]["torch"] = "different-kernels"
                return response
            return answer()
        client = kev_client.KevClient(settings(), transport=transport)
        client.call(payload())
        with self.assertRaises(ProviderError): client.call(payload())
        self.assertEqual(sum(x is not None for x in calls), 1)

    def test_256_choices_and_wrong_model_fail_before_transport(self):
        calls = []
        client = kev_client.KevClient(settings(), transport=lambda *args: calls.append(args))
        x = payload(); x["model"] = "d3"
        with self.assertRaises(ProviderError): client.call(x)
        x = payload(); x["questions"]["action"]["criteria"] = {str(n): str(n) for n in range(256)}
        with self.assertRaises(ProviderError): client.call(x)
        self.assertEqual(calls, [])

    def test_prepare_requires_explicit_download_and_verify_is_local_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "requires --download"):
                kev_model.prepare(tmp)
            with self.assertRaisesRegex(ValueError, "missing"):
                kev_model.verify(tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_health_failure_is_not_an_inference_attempt(self):
        def transport(*args):
            raise ProviderError("health unavailable", dispatched=True)
        with self.assertRaises(ProviderError) as caught:
            kev_client.KevClient(settings(), transport=transport).health()
        self.assertFalse(caught.exception.dispatched)

    def test_full_snapshot_verifier_checks_every_shard_and_head(self):
        # Tiny synthetic files exercise verification only; they are never loaded
        # as model weights or reported as real model output.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in kev_model._REQUIRED_SMALL | kev_model._SHARD_NAMES:
                (root / name).write_bytes(("synthetic " + name).encode())
            (root / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {str(i): name for i, name in enumerate(sorted(kev_model._SHARD_NAMES))}}))
            expected_head = hashlib.sha256((root / "head.pt").read_bytes()).hexdigest()
            files = {p.name: {"bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                             "git_blob_sha1": None} for p in root.iterdir()}
            manifest = {"repo_id": kev_model.REPO, "revision": kev_model.REVISION,
                        "source_revision": kev_model.SOURCE_REVISION, "head_sha256": expected_head,
                        "provenance": "immutable_public_huggingface_file_metadata", "files": files}
            (root / kev_model.MANIFEST_NAME).write_text(json.dumps(manifest))
            with patch.object(kev_model, "HEAD_SHA256", expected_head):
                self.assertEqual(kev_model.verify(root)["verification"], "all_manifest_files_verified")
                name = sorted(kev_model._SHARD_NAMES)[-1]
                content = (root / name).read_bytes()
                (root / name).write_bytes(b"x" + content[1:])
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    kev_model.verify(root)

    def test_dirty_or_wrong_upstream_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(kev_model.subprocess, "check_output", return_value="wrong-commit\n"):
                with self.assertRaisesRegex(ValueError, "Exact frozen"):
                    kev_model.verify_source(tmp)
            with patch.object(kev_model.subprocess, "check_output", side_effect=[kev_model.SOURCE_REVISION + "\n", " M kev/serve.py\n"]):
                with self.assertRaisesRegex(ValueError, "clean checkout"):
                    kev_model.verify_source(tmp)


if __name__ == "__main__":
    unittest.main()
