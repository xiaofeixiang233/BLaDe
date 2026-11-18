"""Durable one-attempt requests, exact caches, and independent logical costs.

SQLite transactions reserve budgets across threads/processes. File locks cover
the entire physical request so a restart cannot silently redispatch an inflight
request. Successful shared receipts retain their full cost for every method.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time

from .providers import (ProviderError, Providers, Receipt, d3_endpoint_pool, dumps, extract_goal,
                        parse_distribution, redact_runtime, settings_for, strict_json,
                        usage_and_cost)


class Halt(RuntimeError):
    def __init__(self, message, request_id=None):
        super().__init__(message)
        self.request_id = request_id


class BudgetHalt(Halt):
    """No dispatch happened; the same episode can resume with a larger budget."""


class ProviderFailure(Halt):
    """A retained, never-retried provider failure that ends only its own episode."""


class Ledger:
    def __init__(self, out, cfg, providers=None):
        self.out = Path(out).expanduser().resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "locks").mkdir(exist_ok=True)
        self.cfg = deepcopy(cfg)
        # A dict of offline callables is supported for deterministic tests only.
        self._custom = providers if isinstance(providers, dict) else {}
        self.providers = Providers(self.cfg) if providers is None or isinstance(providers, dict) else providers
        self._default = Providers(self.cfg)
        self._mutex = threading.RLock()
        self.db = sqlite3.connect(self.out / "requests.sqlite3", timeout=120,
                                  check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS requests(
            id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,
            payload TEXT NOT NULL,identity TEXT NOT NULL,endpoint TEXT NOT NULL,
            started REAL NOT NULL,finished REAL,raw TEXT,raw_body TEXT,error TEXT,
            latency REAL,input_tokens INTEGER,cached_input_tokens INTEGER,
            output_tokens INTEGER,reasoning_tokens INTEGER,cost_usd REAL,
            reservation_usd REAL,billing_known INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS logical_uses(
            episode TEXT NOT NULL,step INTEGER NOT NULL,role TEXT NOT NULL,
            kind TEXT NOT NULL,request TEXT NOT NULL,
            PRIMARY KEY(episode,step,role));
          CREATE INDEX IF NOT EXISTS uses_request ON logical_uses(request);
          CREATE INDEX IF NOT EXISTS requests_kind ON requests(kind);
          CREATE INDEX IF NOT EXISTS unknown_cloud_request ON requests(kind,status)
            WHERE billing_known=0 AND kind IN ('jev','gpt');
          CREATE TABLE IF NOT EXISTS request_totals(
            kind TEXT PRIMARY KEY,physical_requests INTEGER NOT NULL,
            known_cost_usd REAL NOT NULL,unknown_reserved_usd REAL NOT NULL,
            unknown_billing_requests INTEGER NOT NULL);
          CREATE TRIGGER IF NOT EXISTS totals_after_insert AFTER INSERT ON requests BEGIN
            INSERT OR IGNORE INTO request_totals VALUES(NEW.kind,0,0,0,0);
            UPDATE request_totals SET
              physical_requests=physical_requests+1,
              known_cost_usd=known_cost_usd+CASE WHEN NEW.billing_known=1
                THEN coalesce(NEW.cost_usd,0) ELSE 0 END,
              unknown_reserved_usd=unknown_reserved_usd+CASE WHEN NEW.kind IN ('jev','gpt')
                AND NEW.billing_known=0 THEN coalesce(NEW.reservation_usd,0) ELSE 0 END,
              unknown_billing_requests=unknown_billing_requests+CASE WHEN NEW.kind IN ('jev','gpt')
                AND NEW.billing_known=0 THEN 1 ELSE 0 END
              WHERE kind=NEW.kind;
          END;
          CREATE TRIGGER IF NOT EXISTS request_kind_is_immutable BEFORE UPDATE OF kind ON requests
            WHEN OLD.kind!=NEW.kind BEGIN SELECT RAISE(ABORT,'request kind is immutable'); END;
          CREATE TRIGGER IF NOT EXISTS totals_after_update AFTER UPDATE ON requests BEGIN
            UPDATE request_totals SET
              known_cost_usd=known_cost_usd
                -CASE WHEN OLD.billing_known=1 THEN coalesce(OLD.cost_usd,0) ELSE 0 END
                +CASE WHEN NEW.billing_known=1 THEN coalesce(NEW.cost_usd,0) ELSE 0 END,
              unknown_reserved_usd=unknown_reserved_usd
                -CASE WHEN OLD.kind IN ('jev','gpt') AND OLD.billing_known=0 THEN coalesce(OLD.reservation_usd,0) ELSE 0 END
                +CASE WHEN NEW.kind IN ('jev','gpt') AND NEW.billing_known=0 THEN coalesce(NEW.reservation_usd,0) ELSE 0 END,
              unknown_billing_requests=unknown_billing_requests
                -CASE WHEN OLD.kind IN ('jev','gpt') AND OLD.billing_known=0 THEN 1 ELSE 0 END
                +CASE WHEN NEW.kind IN ('jev','gpt') AND NEW.billing_known=0 THEN 1 ELSE 0 END
              WHERE kind=NEW.kind;
          END;
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(requests)")}
        for field in ("owner_episode", "credential_slot", "model"):
            if field not in columns:
                self.db.execute("ALTER TABLE requests ADD COLUMN " + field + " TEXT")
        # One-time migration from the first portable ledger format. Thereafter
        # SQLite triggers maintain O(1) reservation totals, including inserts
        # made by another live process or a retained orphaned request.
        with self.transaction() as db:
            if not db.execute("SELECT 1 FROM request_totals LIMIT 1").fetchone():
                db.execute("INSERT INTO request_totals SELECT kind,count(*),"
                           "sum(CASE WHEN billing_known=1 THEN coalesce(cost_usd,0) ELSE 0 END),"
                           "sum(CASE WHEN kind IN ('jev','gpt') AND billing_known=0 THEN coalesce(reservation_usd,0) ELSE 0 END),"
                           "sum(CASE WHEN kind IN ('jev','gpt') AND billing_known=0 THEN 1 ELSE 0 END) "
                           "FROM requests GROUP BY kind")
        protocol = self.cfg.get("protocol_hash")
        if (not isinstance(protocol, str) or len(protocol) != 64
                or any(character not in "0123456789abcdef" for character in protocol)):
            self.close()
            raise Halt("a frozen 64-character protocol_hash is required")
        identity = dumps({"protocol_hash": protocol,
                          "providers": {kind: self._identity(kind)
                                        for kind in ("jev", "d3", "kev", "gpt")
                                        if kind in self.cfg.get("providers", {})}}, canonical=True)
        with self.transaction() as db:
            old = db.execute("SELECT value FROM metadata WHERE key='scientific_identity'").fetchone()
            if old and old["value"] != identity:
                raise Halt("frozen protocol/provider identity changed; use a new output directory")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('scientific_identity',?)", (identity,))
        self._recover_orphans()

    def _identity(self, kind):
        # Budgets, deadlines, credential environment, and GPU billing rates are
        # operational settings. Endpoint/model/token limits/prices stay frozen.
        settings = settings_for(self.cfg, kind)
        if kind == "d3":
            # Worker selection changes only the physical serving replica. The
            # declared pool and every pinned computation field remain frozen.
            settings["endpoint"] = d3_endpoint_pool(settings)[0]
        if kind == "kev":
            from .kev_client import kev_endpoint_pool
            settings["endpoint"] = kev_endpoint_pool(settings)[0]
        return {"protocol_hash": self.cfg["protocol_hash"],
                "provider": {key: value for key, value in settings.items()
                             if key not in {"timeout_seconds", "reserve_usd_per_call"}}}

    @contextmanager
    def transaction(self):
        with self._mutex:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    @contextmanager
    def lock(self, request_id):
        # Caller values are hashed again, never interpreted as path fragments.
        name = hashlib.sha256(request_id.encode()).hexdigest()
        with (self.out / "locks" / name).open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _recover_orphans(self):
        with self._mutex:
            pending = [row["id"] for row in self.db.execute("SELECT id FROM requests WHERE status='pending'")]
        for request_id in pending:
            name = hashlib.sha256(request_id.encode()).hexdigest()
            with (self.out / "locks" / name).open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # Another live process still owns this request/reservation.
                    continue
                try:
                    with self.transaction() as db:
                        db.execute("UPDATE requests SET status='uncertain',error=? WHERE id=? AND status='pending'",
                                   ("process ended with inflight request; dispatch/billing unknown; no retry", request_id))
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def request_id(self, kind, payload):
        settings = settings_for(self.cfg, kind)
        if kind in self._custom:
            self._default.preflight(kind, payload)
        else:
            self.providers.preflight(kind, payload)
        order = None
        if kind != "gpt":
            order = [list(question["criteria"]) for question in payload["questions"].values()]
        value = {"kind": kind, "identity": self._identity(kind),
                 "payload": payload, "native_option_order": order}
        return hashlib.sha256(dumps(value, canonical=True).encode()).hexdigest()

    def _use(self, db, episode, step, role, kind, request_id):
        if (not isinstance(episode, str) or not episode or type(step) is not int or step < 0
                or not isinstance(role, str) or not role):
            raise Halt("episode, nonnegative step and logical role are required", request_id)
        old = db.execute("SELECT kind,request FROM logical_uses WHERE episode=? AND step=? AND role=?",
                         (episode, step, role)).fetchone()
        if old and (old["kind"] != kind or old["request"] != request_id):
            raise Halt("logical dependency changed during resume", request_id)
        db.execute("INSERT OR IGNORE INTO logical_uses VALUES(?,?,?,?,?)",
                   (episode, step, role, kind, request_id))

    def _reservation(self, kind, settings):
        if kind in ("d3", "kev"):
            return None
        cap = self.cfg.get("budgets", {}).get(kind + "_usd")
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap < 0:
            raise BudgetHalt("explicit finite " + kind + "_usd budget required before a paid request")
        rates = settings["prices_per_million"]
        if kind == "gpt":
            bound = (settings["max_prompt_tokens"] * max(rates["input"], rates["cached_input"])
                     + settings["max_output_tokens"] * rates["output"]) / 1e6
        else:
            output_limit = settings.get("max_output_tokens", 0)
            if rates["output"] and (type(output_limit) is not int or output_limit <= 0):
                raise BudgetHalt("a Jev output bound is required for nonfree output")
            bound = (settings["max_input_tokens"] * rates["input"]
                     + output_limit * rates["output"]) / 1e6
        configured = settings.get("reserve_usd_per_call", bound)
        if (type(configured) not in (int, float) or not math.isfinite(configured)
                or configured < bound - 1e-12 or configured < 0):
            raise BudgetHalt("per-call reservation does not cover frozen token bounds")
        return float(configured)

    def _spend(self, db, kind):
        row = db.execute("SELECT * FROM request_totals WHERE kind=?", (kind,)).fetchone()
        physical_requests = row["physical_requests"] if row else 0
        if kind in ("d3", "kev"):
            return {"physical_requests": physical_requests, "accounted_usd": None,
                    "known_cost_usd": None, "unknown_reserved_usd": None,
                    "unknown_billing_requests": 0, "local_gpu_cost_usd": None}
        known = max(0.0, row["known_cost_usd"]) if row else 0.0
        held = max(0.0, row["unknown_reserved_usd"]) if row else 0.0
        return {"physical_requests": physical_requests, "accounted_usd": known + held,
                "known_cost_usd": known, "unknown_reserved_usd": held,
                "unknown_billing_requests": row["unknown_billing_requests"] if row else 0}

    def spend(self, kind):
        with self._mutex:
            return self._spend(self.db, kind)

    def _gpt_rate_wait(self, limit=45, window=60.0):
        # Stay below a 60 requests/minute provider limit; share one window across processes.
        path = self.out / "locks" / "gpt-rate.json"
        while True:
            with open(path, "a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.seek(0)
                text = handle.read()
                now = time.time()
                stamps = [t for t in (json.loads(text) if text else []) if now - t < window]
                if len(stamps) < limit:
                    stamps.append(now)
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps(stamps))
                    return
                pause = stamps[0] + window - now
            time.sleep(max(pause, 0.05))

    def call(self, kind, payload, episode, step, role):
        payload = deepcopy(payload)
        request_id = self.request_id(kind, payload)
        settings = settings_for(self.cfg, kind)
        with self.lock(request_id):
            with self.transaction() as db:
                self._use(db, episode, step, role, kind, request_id)
                row = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
                if row:
                    if (row["kind"] != kind or strict_json(row["payload"]) != payload
                            or row["identity"] != dumps(self._identity(kind), canonical=True)):
                        raise Halt("cached request identity/body mismatch", request_id)
                    if row["status"] != "complete" or kind in ("jev", "gpt") and not row["billing_known"]:
                        raise ProviderFailure("original failed/inflight/unknown-billing request retained; no retry", request_id)
                    raw = strict_json(row["raw"])
                    extract_goal(raw) if kind == "gpt" else parse_distribution(raw, payload)
                    return request_id, raw
                reservation = self._reservation(kind, settings)
                if kind in ("jev", "gpt"):
                    # Unknown-billing failures keep their full reservation inside accounted spend.
                    spend = self._spend(db, kind)
                    if spend["accounted_usd"] + reservation > self.cfg["budgets"][kind + "_usd"] + 1e-12:
                        raise BudgetHalt(kind + " budget reservation would be exceeded; no dispatch", request_id)
                    total_cap = self.cfg.get("budgets", {}).get("total_usd")
                    if total_cap is not None:
                        if type(total_cap) not in (int, float) or not math.isfinite(total_cap) or total_cap < 0:
                            raise BudgetHalt("total_usd must be finite and nonnegative")
                        total = sum(self._spend(db, provider)["accounted_usd"] for provider in ("jev", "gpt"))
                        if total + reservation > total_cap + 1e-12:
                            raise BudgetHalt("total API budget reservation would be exceeded; no dispatch", request_id)
                # Validate missing/invalid credentials before persisting a request
                # that could not have been sent. Tests inject providers, not keys.
                if (kind not in self._custom and isinstance(self.providers, Providers)
                        and self.providers.transport is None):
                    self.providers._credential(kind)
                slot = self.cfg.get("credential_slots", {}).get(kind, kind + ("-primary" if kind in ("jev", "gpt") else "-local"))
                if not isinstance(slot, str) or not __import__('re').fullmatch(r"[A-Za-z0-9_-]{1,48}", slot):
                    raise Halt("Use a short anonymous credential slot label, never a key")
                if kind in ('jev','gpt') and slot!=kind+'-primary' and not __import__('re').fullmatch(kind+r'-key-[0-9]{2}',slot):
                    raise Halt('Use anonymous labels such as '+kind+'-key-01; never a credential value or fragment')
                db.execute("INSERT INTO requests(id,kind,status,payload,identity,endpoint,started,reservation_usd,billing_known,owner_episode,credential_slot,model) "
                           "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                           (request_id, kind, "pending", dumps(payload), dumps(self._identity(kind), canonical=True),
                            settings["endpoint"], time.time(), reservation, 0, episode, slot, payload["model"]))
            if kind == "gpt" and kind not in self._custom:
                self._gpt_rate_wait()
            start = time.monotonic()
            raw = body = error = None
            usage = usage_and_cost(kind, None, settings)
            try:
                if kind in self._custom:
                    receipt = self._custom[kind](deepcopy(payload), deepcopy(settings))
                    if not isinstance(receipt, Receipt):
                        receipt = Receipt(receipt, dumps(receipt))
                else:
                    receipt = self.providers.call(kind, payload)
                raw, body = receipt.raw, receipt.raw_body
                usage = usage_and_cost(kind, raw, settings)
                if kind in ("jev", "gpt") and not usage["billing_known"]:
                    raise ProviderError("paid provider returned unknown usage/billing", raw=raw, raw_body=body)
                if kind in ("jev", "gpt") and usage["cost_usd"] > reservation + 1e-12:
                    raise ProviderError("actual token cost exceeded reservation", raw=raw, raw_body=body)
                if kind == "gpt":
                    if (usage["input_tokens"] > settings["max_prompt_tokens"]
                            or usage["output_tokens"] > settings["max_output_tokens"]):
                        raise ProviderError("GPT actual usage exceeded frozen token bounds", raw=raw, raw_body=body)
                    extract_goal(raw)
                else:
                    parse_distribution(raw, payload)
                    if (kind == "d3" and (usage["input_tokens"] is None or usage["output_tokens"] != 0)
                            or kind == "jev" and usage["input_tokens"] > settings["max_input_tokens"]):
                        raise ProviderError("actor actual usage missing/exceeded token bound", raw=raw, raw_body=body)
            except Exception as exc:
                if raw is None:
                    raw = getattr(exc, "raw", None)
                if body is None:
                    body = getattr(exc, "raw_body", None)
                usage = usage_and_cost(kind, raw, settings)
                # HTTP errors use fixed codes. Arbitrary mocked errors are also
                # credential-redacted and bounded before entering SQLite.
                error = redact_runtime(type(exc).__name__ + ": " + str(exc))[:2048]
            body = redact_runtime(body) if body is not None else None
            if raw is not None:
                try:
                    raw_text = redact_runtime(dumps(raw))
                except (ValueError, TypeError):
                    raw_text = None
            else:
                raw_text = None
            with self.transaction() as db:
                db.execute("UPDATE requests SET status=?,finished=?,raw=?,raw_body=?,error=?,latency=?,"
                           "input_tokens=?,cached_input_tokens=?,output_tokens=?,reasoning_tokens=?,"
                           "cost_usd=?,billing_known=? WHERE id=?",
                           ("failed" if error else "complete", time.time(), raw_text, body, error,
                            time.monotonic() - start, usage["input_tokens"], usage["cached_input_tokens"],
                            usage["output_tokens"], usage["reasoning_tokens"], usage["cost_usd"],
                            int(usage["billing_known"]), request_id))
            if error:
                raise ProviderFailure("original " + kind + " failure retained; no retry: " + error, request_id)
            return request_id, deepcopy(raw)

    def report_rows(self):
        with self._mutex:
            return {name: [dict(row) for row in self.db.execute("SELECT * FROM " + name)]
                    for name in ("requests", "logical_uses")}

    def efficiency_rows(self):
        """Only request metadata, never materialize full prompt/response bodies."""
        with self._mutex:
            columns = ("id,kind,status,model,credential_slot,owner_episode,started,finished,"
                       "latency,input_tokens,output_tokens,cached_input_tokens,reasoning_tokens,"
                       "cost_usd,billing_known,reservation_usd")
            return {"requests": [dict(row) for row in self.db.execute("SELECT " + columns + " FROM requests")],
                    "logical_uses": [dict(row) for row in self.db.execute("SELECT * FROM logical_uses")]}

    @staticmethod
    def _summary(row, kind):
        """Convert one numeric SQL group; never materialize request bodies."""
        row = dict(row) if row is not None else {}
        cloud = kind in ("jev", "gpt")
        known_cost = row.get("known_cost", 0) or 0
        unknown = row.get("unknown_billing", 0) or 0
        numeric = {key: row.get(key, 0) or 0 for key in
                   ("requests", "complete", "failed", "pending_or_uncertain", "input_tokens",
                    "output_tokens", "cached_input_tokens", "reasoning_tokens",
                    "unknown_input_usage_requests", "unknown_output_usage_requests",
                    "receipt_latency_seconds", "unknown_latency_requests")}
        return {**numeric,
                "unknown_billing_requests": unknown if cloud else 0,
                "known_cost_subtotal_usd": known_cost if cloud else None,
                "cost_usd": known_cost if cloud and not unknown else None,
                "unknown_reserved_usd": (row.get("unknown_reserved", 0) or 0) if cloud else None,
                "local_gpu_cost_usd": None}

    def statistics(self):
        columns = """r.kind AS kind,count(*) AS requests,
          sum(r.status='complete') AS complete,sum(r.status='failed') AS failed,
          sum(r.status IN ('pending','uncertain')) AS pending_or_uncertain,
          sum(r.input_tokens) AS input_tokens,sum(r.output_tokens) AS output_tokens,
          sum(r.cached_input_tokens) AS cached_input_tokens,sum(r.reasoning_tokens) AS reasoning_tokens,
          sum(r.input_tokens IS NULL) AS unknown_input_usage_requests,
          sum(r.output_tokens IS NULL) AS unknown_output_usage_requests,
          sum(CASE WHEN r.billing_known=1 THEN coalesce(r.cost_usd,0) ELSE 0 END) AS known_cost,
          sum(r.billing_known=0) AS unknown_billing,
          sum(CASE WHEN r.billing_known=0 THEN coalesce(r.reservation_usd,0) ELSE 0 END) AS unknown_reserved,
          sum(r.latency) AS receipt_latency_seconds,sum(r.latency IS NULL) AS unknown_latency_requests"""
        with self._mutex:
            physical_rows = {row["kind"]: row for row in self.db.execute(
                "SELECT " + columns + " FROM requests r GROUP BY r.kind")}
            logical = {}
            for row in self.db.execute("SELECT u.episode AS episode," + columns +
                    " FROM logical_uses u JOIN requests r ON r.id=u.request GROUP BY u.episode,r.kind"):
                episode, kind = row["episode"], row["kind"]
                if episode not in logical:
                    logical[episode] = {provider: self._summary(None, provider) for provider in ("jev", "d3", "kev", "gpt")}
                    logical[episode]["undispatched_dependencies"] = 0
                logical[episode][kind] = self._summary(row, kind)
            for row in self.db.execute("SELECT u.episode AS episode,count(*) AS undispatched FROM logical_uses u "
                    "LEFT JOIN requests r ON r.id=u.request WHERE r.id IS NULL GROUP BY u.episode"):
                episode = row["episode"]
                if episode not in logical:
                    logical[episode] = {provider: self._summary(None, provider) for provider in ("jev", "d3", "kev", "gpt")}
                logical[episode]["undispatched_dependencies"] = row["undispatched"]
            spend = {kind: self._spend(self.db, kind) for kind in ("jev", "d3", "kev", "gpt")}
        return {"physical": {kind: self._summary(physical_rows.get(kind), kind) for kind in ("jev", "d3", "kev", "gpt")},
                "logical_by_episode": logical,
                "spend": spend,
                "cost_definition": "Token-price estimates, not invoices. Each logical use pays its full receipt cost; "
                                   "physical shared receipts are counted once. Unknown cloud fees retain reservations. "
                                   "Local D3 GPU dollar cost is unpriced and null."}

    stats = statistics

    def close(self):
        with self._mutex:
            self.db.close()
