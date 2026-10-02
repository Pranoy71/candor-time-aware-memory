"""Quota-frugal LLM client (standard library only).

Why this looks the way it does
  * Gemini's free tier is small and its model names churn, so: model auto-resolution (ListModels) with a fallback
    chain, exponential backoff that honours the server's retry hint, and an on-disk cache keyed by the full request
    so a re-run costs zero calls and returns byte-identical output.
  * One call per question at most; the caller decides. Every caller has a deterministic fallback, so a missing key,
    a quota error or a malformed reply never breaks a run.
Providers: gemini (default, REST), anthropic (Messages API), openai (any OpenAI-compatible endpoint, incl. Gemini's).
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

from .config import CACHE_DIR

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash-lite"      # the only model used unless GEMINI_MODEL overrides it


class LLMError(Exception):
    pass


class Cache:
    def __init__(self, path=None):
        self.path = (path or CACHE_DIR / "llm_cache.jsonl")
        self.mem = {}
        try:
            if self.path.exists():
                for line in open(self.path, encoding="utf-8"):
                    try:
                        d = json.loads(line)
                        self.mem[d["k"]] = d["v"]
                    except Exception:
                        pass
        except OSError:
            pass

    def get(self, k):
        return self.mem.get(k)

    def put(self, k, v):
        self.mem[k] = v
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        except OSError:
            pass


class LLM:
    def __init__(self, provider=None, cache=True, min_interval=None):
        self.provider = provider or os.environ.get("CANDOR_LLM") or self._detect()
        self.cache = Cache() if cache else None
        self.calls = 0
        self.cached = 0
        self.failed = 0
        self.last = 0.0
        self.min_interval = float(os.environ.get("CANDOR_MIN_INTERVAL", min_interval if min_interval is not None else 6.5))
        self.model = os.environ.get("GEMINI_MODEL") or None
        self.dead = False

    @staticmethod
    def _detect():
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return None

    @property
    def available(self):
        return bool(self.provider) and not self.dead

    # ------------------------------------------------------------------ public
    def complete(self, system, user, json_mode=False, max_tokens=1500, tag=""):
        """Returns text or raises LLMError. Cached by full request."""
        if not self.available:
            raise LLMError("no LLM configured")
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, user, json_mode, max_tokens],
                                        ensure_ascii=False).encode()).hexdigest()
        if self.cache and (hit := self.cache.get(key)) is not None:
            self.cached += 1
            return hit
        text = self._call(system, user, json_mode, max_tokens)
        if self.cache:
            self.cache.put(key, text)
        return text

    def complete_json(self, system, user, max_tokens=1500):
        raw = self.complete(system, user, json_mode=True, max_tokens=max_tokens)
        return parse_json(raw)

    # ------------------------------------------------------------------ transport
    def _throttle(self):
        wait = self.min_interval - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()

    def _http(self, url, body, headers, timeout=90):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _call(self, system, user, json_mode, max_tokens):
        delay = 8.0
        for attempt in range(6):
            self._throttle()
            try:
                self.calls += 1
                if self.provider == "gemini":
                    return self._gemini(system, user, json_mode, max_tokens)
                if self.provider == "anthropic":
                    return self._anthropic(system, user, max_tokens)
                return self._openai(system, user, json_mode, max_tokens)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")[:400]
                if e.code in (429, 500, 502, 503, 504):
                    m = re.search(r"retry in ([\d.]+)s", body) or re.search(r'"retryDelay":\s*"(\d+)s"', body)
                    if "PerDay" in body or "per day" in body.lower():
                        self.failed += 1
                        self.dead = True
                        raise LLMError("daily quota exhausted: " + body[:120])
                    time.sleep(min(90.0, float(m.group(1)) + 1 if m else delay))
                    delay = min(delay * 2, 60)
                    continue
                if e.code == 404 and self.provider == "gemini":
                    self.failed += 1
                    raise LLMError(f"model '{self.model}' not found for this key (set GEMINI_MODEL in .env)")
                self.failed += 1
                if e.code in (400, 401, 403):
                    self.dead = e.code in (401, 403)
                raise LLMError(f"HTTP {e.code}: {body[:200]}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                time.sleep(delay)
                delay = min(delay * 2, 60)
        self.failed += 1
        raise LLMError("gave up after retries")

    # ---- gemini
    def _gemini_key(self):
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def _resolve_gemini_model(self):
        return os.environ.get("GEMINI_MODEL") or GEMINI_MODEL_DEFAULT

    def _gemini(self, system, user, json_mode, max_tokens):
        if not self.model:
            self.model = self._resolve_gemini_model()
        self._last_model = self.model
        cfg = {"temperature": 0, "maxOutputTokens": max_tokens}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg}
        d = self._http(f"{GEMINI}/models/{self.model}:generateContent", body, {"x-goog-api-key": self._gemini_key()})
        try:
            parts = d["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError):
            raise LLMError("empty Gemini reply: " + json.dumps(d)[:200])

    # ---- anthropic
    def _anthropic(self, system, user, max_tokens):
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        d = self._http("https://api.anthropic.com/v1/messages",
                       {"model": model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                        "messages": [{"role": "user", "content": user}]},
                       {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in d.get("content", [])).strip()

    # ---- openai-compatible
    def _openai(self, system, user, json_mode, max_tokens):
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = {"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        d = self._http(base + "/chat/completions", body, {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
        return d["choices"][0]["message"]["content"].strip()

    def stats(self):
        return {"provider": self.provider, "model": self.model, "calls": self.calls, "cache_hits": self.cached,
                "failed": self.failed}


def parse_json(raw):
    """Tolerant JSON extraction (code fences, prose around the object)."""
    if raw is None:
        raise LLMError("empty")
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        pass
    m = re.search(r"\{.*\}|\[.*\]", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except ValueError:
            pass
    raise LLMError("unparseable JSON: " + t[:120])


class MockLLM:
    """Deterministic stand-in for tests: `fn(system, user) -> str`."""
    provider = "mock"
    available = True
    dead = False
    calls = cached = failed = 0
    model = "mock"

    def __init__(self, fn):
        self.fn = fn

    def complete(self, system, user, json_mode=False, max_tokens=1500, tag=""):
        self.calls += 1
        return self.fn(system, user)

    def complete_json(self, system, user, max_tokens=1500):
        return parse_json(self.complete(system, user, True, max_tokens))

    def stats(self):
        return {"provider": "mock", "calls": self.calls}
