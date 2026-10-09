"""Minimal OpenAI-compatible chat client (standard library only).

What it takes care of:
  * retries with back-off on time-outs, connection errors, HTTP 429 and 5xx;
  * optional request fields the server rejects (response_format, seed) are dropped once and
    remembered, and an over-large max_tokens is halved, instead of failing the document;
  * every call is appended to <run_dir>/calls.jsonl (request, raw response, timing; never the key);
  * successful answers are cached on disk by request content, so an interrupted run can be resumed
    and nothing already obtained is requested twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

OPTIONAL_FIELDS = ("response_format", "seed")
RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)
# the CSCS gateway answers 504 after about 60 s when the model has not finished: the same request
# will time out again, so it is retried only once
FAIL_FAST_STATUS = {504: 1}
TOO_LONG_HINTS = ("max_tokens", "max_completion_tokens", "context length", "maximum context", "too large", "too long")
MAX_ADJUSTMENTS = 8


@dataclass
class CallResult:
    ok: bool
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_s: float = 0.0
    attempts: int = 0
    cached: bool = False
    status: Optional[int] = None
    error: Optional[str] = None
    model: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    adjustments: List[str] = field(default_factory=list)   # e.g. ["response_format dropped", "max_tokens halved"]
    max_tokens_used: Optional[int] = None
    prompt_scores: Optional[Dict[str, Any]] = None          # compact prompt_logprobs: {"tokens", "logprobs", "prompt_text"}


def compact_prompt_logprobs(payload: dict) -> Optional[Dict[str, Any]]:
    """Reduce vLLM's prompt_logprobs (one dict per token) to parallel lists of decoded token and logprob."""
    lp = payload.get("prompt_logprobs")
    if not isinstance(lp, list) or not lp:
        return None
    ids = payload.get("prompt_token_ids")
    toks: List[str] = []
    probs: List[Optional[float]] = []
    for i, entry in enumerate(lp):
        if not isinstance(entry, dict) or not entry:
            toks.append("")
            probs.append(None)
            continue
        ent = entry.get(str(ids[i])) if isinstance(ids, list) and i < len(ids) else None
        if ent is None:
            ent = next(iter(entry.values()))
        toks.append(str(ent.get("decoded_token", "")))
        probs.append(ent.get("logprob"))
    return {"tokens": toks, "logprobs": probs, "prompt_text": payload.get("prompt_text")}


class ChatClient:
    def __init__(self, cfg: config.LLMConfig, run_dir: Optional[Path] = None, timeout: float = 120.0,
                 max_retries: int = 5, use_cache: bool = True, cache_dir: Optional[Path] = None,
                 backoff_base: float = 2.0):
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else config.RUNS / "_cache"
        self.backoff_base = backoff_base
        self.unsupported: set = set()
        self._lock = threading.Lock()
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ plumbing
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer " + self.cfg.api_key, "Content-Type": "application/json",
                "Accept": "application/json", "User-Agent": "uzh-diff/0.1"}

    def _request(self, method: str, path: str, body: Optional[dict]):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.cfg.base_url + path, data=data, headers=self._headers(), method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers.items()), raw

    def _log(self, record: dict) -> None:
        if not self.run_dir:
            return
        with self._lock:
            with open(self.run_dir / "calls.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _cache_path(self, key_body: dict) -> Path:
        key = json.dumps({"base_url": self.cfg.base_url, "body": key_body}, sort_keys=True, ensure_ascii=False)
        return self.cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    @staticmethod
    def _interesting_headers(headers: Dict[str, str]) -> Dict[str, str]:
        keep = {}
        for k, v in headers.items():
            lk = k.lower()
            if "ratelimit" in lk or "rate-limit" in lk or lk in ("retry-after", "x-request-id", "server"):
                keep[lk] = v
        return keep

    def _write_cache(self, cache_file: Path, payload: dict) -> None:
        """Write to a temporary file first so that a reader never sees a half-written entry."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".%d.tmp" % threading.get_ident())
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(tmp, cache_file)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _retry_wait(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(120.0, float(retry_after))
            except ValueError:
                pass
        return min(60.0, self.backoff_base * (2 ** (attempt - 1))) + random.uniform(0, 1.0)

    # ------------------------------------------------------------------ public API
    def raw_post(self, path: str, body: dict, tag: str = "") -> Dict[str, Any]:
        """One POST without retries or request rewriting; returns status, headers and the parsed body.
        Used by the capability probes, which need to see exactly what the server does with a request."""
        t0 = time.time()
        try:
            status, headers, raw = self._request("POST", path, body)
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            status, headers = e.code, dict(e.headers.items()) if e.headers else {}
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            out = {"status": None, "error": "%s: %s" % (type(e).__name__, str(e)[:300]), "latency_s": round(time.time() - t0, 3)}
            self._log({"tag": tag, "probe": True, "request": body, "result": out})
            return out
        try:
            payload: Any = json.loads(raw)
        except ValueError:
            payload = raw
        out = {"status": status, "headers": self._interesting_headers(headers), "payload": payload,
               "latency_s": round(time.time() - t0, 3)}
        self._log({"tag": tag, "probe": True, "request": body, "result": out})
        return out

    def list_models(self) -> List[str]:
        _, _, raw = self._request("GET", "/models", None)
        data = json.loads(raw)
        items = data.get("data", data) if isinstance(data, dict) else data
        return [m.get("id", str(m)) if isinstance(m, dict) else str(m) for m in items]

    def chat(self, messages: List[Dict[str, str]], tag: str = "", model: Optional[str] = None,
             **params: Any) -> CallResult:
        # the request as the caller asked for it: this is what identifies the call in the cache
        key_body: Dict[str, Any] = {"model": model or self.cfg.name, "messages": messages}
        key_body.update({k: v for k, v in params.items() if v is not None})
        cache_file = self._cache_path(key_body)
        if self.use_cache and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                result = CallResult(**cached["result"])
            except (ValueError, TypeError, KeyError, OSError):
                cached = None       # unreadable (e.g. still being written by another worker): treat as a miss
            if cached is not None:
                result.cached = True
                self._log({"tag": tag, "cached": True, "result": cached["result"]})
                return result

        # the request actually sent: without fields this server is known to reject
        body = {k: v for k, v in key_body.items() if k not in self.unsupported}
        adjustments: List[str] = []
        attempts = 0
        last_error: Optional[str] = None
        last_status: Optional[int] = None
        started = time.time()
        while True:
            attempts += 1
            wait = 0.0
            t0 = time.time()
            try:
                status, headers, raw = self._request("POST", "/chat/completions", body)
                payload = json.loads(raw)
                choice = (payload.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                usage = payload.get("usage") or {}
                result = CallResult(
                    ok=True, content=message.get("content"), finish_reason=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                    latency_s=round(time.time() - t0, 3), attempts=attempts, status=status,
                    model=payload.get("model"), headers=self._interesting_headers(headers),
                    adjustments=adjustments, max_tokens_used=body.get("max_tokens"),
                    prompt_scores=compact_prompt_logprobs(payload))
                if result.prompt_scores is not None:
                    # keep the log readable: the per-token dicts are replaced by the compact form
                    payload = dict(payload)
                    payload["prompt_logprobs"] = "(compacted into result.prompt_scores)"
                    payload.pop("prompt_token_ids", None)
                    payload.pop("prompt_text", None)
                self._log({"tag": tag, "cached": False, "request": body, "response": payload,
                           "result": result.__dict__, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
                if self.use_cache and result.content is not None:
                    self._write_cache(cache_file, {"request": key_body, "result": result.__dict__})
                return result
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                last_status, last_error = e.code, "HTTP %d: %s" % (e.code, detail[:600])
                if e.code in (400, 422) and len(adjustments) < MAX_ADJUSTMENTS:
                    low = detail.lower()
                    changed = False
                    for name in OPTIONAL_FIELDS:
                        if name in body and name in low:
                            body.pop(name)
                            self.unsupported.add(name)
                            adjustments.append(name + " dropped")
                            changed = True
                    if not changed and int(body.get("max_tokens") or 0) > 512 and any(h in low for h in TOO_LONG_HINTS):
                        body["max_tokens"] = int(body["max_tokens"]) // 2
                        adjustments.append("max_tokens halved to %d" % body["max_tokens"])
                        changed = True
                    if changed:
                        attempts -= 1          # adjusting the request is not a retry
                        continue
                    break
                if e.code not in RETRY_STATUS:
                    break                       # 401, 403, 404, a real 400, ...: retrying will not help
                if e.code in FAIL_FAST_STATUS and attempts > FAIL_FAST_STATUS[e.code]:
                    break
                wait = self._retry_wait(attempts, e.headers.get("Retry-After") if e.headers else None)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ValueError) as e:
                last_status, last_error = None, "%s: %s" % (type(e).__name__, str(e)[:400])
                wait = self._retry_wait(attempts, None)
            if attempts > self.max_retries:
                break
            time.sleep(wait)
        result = CallResult(ok=False, latency_s=round(time.time() - started, 3), attempts=attempts,
                            status=last_status, error=last_error, adjustments=adjustments,
                            max_tokens_used=body.get("max_tokens"))
        self._log({"tag": tag, "cached": False, "request": body, "result": result.__dict__,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return result
