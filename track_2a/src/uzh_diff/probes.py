"""Capability probes for the inference endpoint (vLLM behind a gateway).

Each probe sends one request and checks the *effect*, not the HTTP status: vLLM accepts unknown
fields silently, so "200 OK" alone proves nothing. Request shapes follow the vLLM request schema;
the pass criteria are what the research notes (10_Track2A_文献调研, appendix A) specify.

    T1  prompt_logprobs          per-token log-probabilities of the prompt      -> method D4 (teacher forcing)
    T2  assistant prefill        continue_final_message                         -> D4
    T2b prefill + prompt_logprobs (the exact combination D4 needs)              -> D4
    T3  n = 3                    several answers from one request               -> cheap voting (D3)
    T4  structured_outputs       constrain the answer to a choice               -> forced formats (D2)
    T6  json_schema              response_format with a schema                  -> forced formats (D2)
    T7  logprobs                 log-probabilities of generated tokens          -> confidence of yes/no answers
    T5  prefix caching           same long prompt twice, timing                 -> cost of blockwise prompting
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .client import ChatClient


def _content(payload: Any) -> Optional[str]:
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _fingerprint(payload: Any) -> Optional[str]:
    return payload.get("system_fingerprint") if isinstance(payload, dict) else None


def _err(res: dict) -> str:
    if res.get("error"):
        return res["error"]
    p = res.get("payload")
    if isinstance(p, dict) and "error" in p:
        return json.dumps(p["error"], ensure_ascii=False)[:300]
    return "HTTP %s" % res.get("status")


def probe_prompt_logprobs(client: ChatClient, model: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "Die Hauptstadt der Schweiz ist Bern."}],
            "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1, "return_token_ids": True, "return_prompt_text": True}
    res = client.raw_post("/chat/completions", body, tag="probe:T1")
    p = res.get("payload")
    ok, detail = False, _err(res)
    if res.get("status") == 200 and isinstance(p, dict):
        lp = p.get("prompt_logprobs")
        n_prompt = (p.get("usage") or {}).get("prompt_tokens")
        if isinstance(lp, list) and lp and lp[0] is None and all(isinstance(x, dict) for x in lp[1:]):
            sample = lp[-1]
            first = next(iter(sample.values())) if sample else {}
            ok = isinstance(first, dict) and "logprob" in first
            detail = "list of %d entries (usage.prompt_tokens=%s); last entry e.g. %s" % (
                len(lp), n_prompt, json.dumps(sample, ensure_ascii=False)[:160])
            if "prompt_token_ids" in p:
                detail += "; prompt_token_ids present"
        else:
            detail = "prompt_logprobs=%r (field ignored by the server or gateway)" % (lp,)
    return {"id": "T1", "name": "prompt_logprobs（prompt 里每个 token 的对数概率）", "ok": ok, "detail": detail,
            "fingerprint": _fingerprint(p), "status": res.get("status")}


def probe_prefill(client: ChatClient, model: str) -> dict:
    body = {"model": model,
            "messages": [{"role": "user", "content": "Name the capital of Switzerland."},
                         {"role": "assistant", "content": "The capital of Switzerland is"}],
            "max_tokens": 5, "temperature": 0, "add_generation_prompt": False, "continue_final_message": True,
            "return_prompt_text": True}
    res = client.raw_post("/chat/completions", body, tag="probe:T2")
    p = res.get("payload")
    ok, detail = False, _err(res)
    if res.get("status") == 200 and isinstance(p, dict):
        c = _content(p) or ""
        prompt_text = p.get("prompt_text") or ""
        continued = (not c.strip().lower().startswith("the capital")) and len(c.strip()) > 0
        ends_open = prompt_text.rstrip().endswith("The capital of Switzerland is") if prompt_text else None
        ok = continued and (ends_open is not False)
        detail = "answer=%r; prompt_text ends with the prefill: %s" % (c[:60], ends_open)
    return {"id": "T2", "name": "assistant 预填并接着写（continue_final_message）", "ok": ok, "detail": detail,
            "fingerprint": _fingerprint(p), "status": res.get("status")}


def probe_prefill_logprobs(client: ChatClient, model: str) -> dict:
    """The combination D4 needs: score a given text token by token, conditioned on the user message."""
    def one(user: str) -> dict:
        body = {"model": model,
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": "Es regnet."}],
                "max_tokens": 1, "temperature": 0, "add_generation_prompt": False, "continue_final_message": True,
                "prompt_logprobs": 0, "return_token_ids": True, "return_prompt_text": True}
        return client.raw_post("/chat/completions", body, tag="probe:T2b")
    res = one("Translate into German: It is raining.")
    p = res.get("payload")
    ok, detail = False, _err(res)
    tail_sum = None
    if res.get("status") == 200 and isinstance(p, dict):
        lp = p.get("prompt_logprobs")
        if isinstance(lp, list) and len(lp) > 3 and isinstance(lp[-1], dict):
            # the prefilled text is at the end of the prompt (possibly followed by nothing): take the last 4 entries
            tail = [next(iter(x.values()))["logprob"] for x in lp[-4:] if isinstance(x, dict) and x]
            tail_sum = sum(tail)
            ok = True
            detail = "prompt_logprobs has %d entries; last 4 logprobs (should cover 'Es regnet.'): %s" % (
                len(lp), ["%.2f" % v for v in tail])
        else:
            detail = "prompt_logprobs=%r" % (lp,)[:80]
    effect = None
    if ok:
        res2 = one("Translate into German: The meeting is on Monday.")
        p2 = res2.get("payload")
        lp2 = p2.get("prompt_logprobs") if isinstance(p2, dict) else None
        if isinstance(lp2, list) and len(lp2) > 3:
            tail2 = [next(iter(x.values()))["logprob"] for x in lp2[-4:] if isinstance(x, dict) and x]
            effect = {"matching_context": round(tail_sum, 3), "unrelated_context": round(sum(tail2), 3)}
            detail += "; summed logprob with matching source %.2f vs unrelated source %.2f (matching should be higher)" % (
                tail_sum, sum(tail2))
            ok = tail_sum > sum(tail2)
    return {"id": "T2b", "name": "预填 + prompt_logprobs（给指定文字逐 token 打分，D4 的形态）", "ok": ok, "detail": detail,
            "effect": effect, "fingerprint": _fingerprint(p), "status": res.get("status")}


def probe_n(client: ChatClient, model: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "Give one synonym of 'fast'. Answer with one word."}],
            "max_tokens": 8, "temperature": 0.8, "top_p": 0.9, "n": 3, "seed": 1}
    res = client.raw_post("/chat/completions", body, tag="probe:T3")
    p = res.get("payload")
    ok, detail = False, _err(res)
    if res.get("status") == 200 and isinstance(p, dict):
        choices = p.get("choices") or []
        answers = [(c.get("message") or {}).get("content", "") for c in choices]
        ok = len(choices) == 3
        detail = "%d choices: %s; usage=%s" % (len(choices), answers, p.get("usage"))
    return {"id": "T3", "name": "n=3（一次请求三份回答）", "ok": ok, "detail": detail,
            "fingerprint": _fingerprint(p), "status": res.get("status")}


def probe_structured_choice(client: ChatClient, model: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "What is the capital of Switzerland?"}],
            "max_tokens": 8, "temperature": 0, "structured_outputs": {"choice": ["QX7", "ZK3"]}}
    res = client.raw_post("/chat/completions", body, tag="probe:T4")
    p = res.get("payload")
    ok, detail = False, _err(res)
    if res.get("status") == 200 and isinstance(p, dict):
        c = (_content(p) or "").strip()
        ok = c in ("QX7", "ZK3")
        detail = "answer=%r" % c[:60]
    return {"id": "T4", "name": "structured_outputs.choice（强制只能回固定选项）", "ok": ok, "detail": detail,
            "fingerprint": _fingerprint(p), "status": res.get("status")}


def probe_json_schema(client: ChatClient, model: str) -> dict:
    schema = {"type": "object", "properties": {"blocks": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "status": {"type": "string", "enum": ["covered", "absent", "partly"]}},
        "required": ["id", "status"], "additionalProperties": False}}}, "required": ["blocks"], "additionalProperties": False}
    body = {"model": model,
            "messages": [{"role": "user", "content": "Tell me in two sentences why the sky is blue, then rate blocks a000 and a001 as covered or absent."}],
            "max_tokens": 120, "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": schema}}}
    res = client.raw_post("/chat/completions", body, tag="probe:T6")
    p = res.get("payload")
    ok, detail = False, _err(res)
    if res.get("status") == 200 and isinstance(p, dict):
        c = _content(p) or ""
        try:
            obj = json.loads(c)
            ok = isinstance(obj, dict) and set(obj.keys()) == {"blocks"} and all(
                set(b.keys()) == {"id", "status"} for b in obj["blocks"])
            detail = "parsed: %s" % json.dumps(obj, ensure_ascii=False)[:160]
        except ValueError:
            detail = "not JSON: %r" % c[:80]
    return {"id": "T6", "name": "response_format=json_schema（按 schema 强制输出）", "ok": ok, "detail": detail,
            "fingerprint": _fingerprint(p), "status": res.get("status")}


def probe_logprobs(client: ChatClient, model: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "Answer with one word: what is the capital of Switzerland?"}],
            "max_tokens": 5, "temperature": 0, "logprobs": True, "top_logprobs": 5}
    res = client.raw_post("/chat/completions", body, tag="probe:T7")
    p = res.get("payload")
    ok, detail = False, _err(res)
    if res.get("status") == 200 and isinstance(p, dict):
        try:
            content = p["choices"][0]["logprobs"]["content"]
            ok = isinstance(content, list) and len(content) > 0 and "logprob" in content[0]
            detail = "first token %r logprob %.3f, %d alternatives" % (
                content[0].get("token"), content[0].get("logprob"), len(content[0].get("top_logprobs") or []))
        except (KeyError, IndexError, TypeError):
            detail = "choices[0].logprobs missing or null"
    return {"id": "T7", "name": "logprobs（生成 token 的对数概率）", "ok": ok, "detail": detail,
            "fingerprint": _fingerprint(p), "status": res.get("status")}


def probe_prefix_cache(client: ChatClient, model: str, repeats: int = 2) -> dict:
    filler = " ".join("The Federal Office of Energy publishes its annual report in four languages ." for _ in range(220))
    def one(k: int) -> Optional[float]:
        body = {"model": model, "messages": [{"role": "user", "content": filler + "\n\nQuestion %d: reply with the single word OK." % k}],
                "max_tokens": 1, "temperature": 0}
        res = client.raw_post("/chat/completions", body, tag="probe:T5")
        if res.get("status") != 200:
            return None
        return res["latency_s"]
    times: List[Optional[float]] = []
    for r in range(repeats):
        times.append(one(1000 + r))   # same long prefix, different last sentence
        times.append(one(2000 + r))
    first, rest = times[0], [t for t in times[1:] if t is not None]
    ok = first is not None and bool(rest) and (min(rest) < 0.6 * first)
    detail = "latencies with the same ~4k-token prefix: %s (a clearly faster second call means prefix caching)" % (
        ["%.2fs" % t if t is not None else "fail" for t in times])
    return {"id": "T5", "name": "前缀缓存（相同长前缀第二次是否明显更快）", "ok": ok, "detail": detail, "status": 200 if first else None}


ALL_PROBES = [probe_prompt_logprobs, probe_prefill, probe_prefill_logprobs, probe_n, probe_structured_choice,
              probe_json_schema, probe_logprobs, probe_prefix_cache]


def run_probes(client: ChatClient, model: str, which: Optional[List[str]] = None) -> List[dict]:
    out = []
    for fn in ALL_PROBES:
        if which and fn.__name__.replace("probe_", "") not in which:
            continue
        try:
            out.append(fn(client, model))
        except Exception as e:   # a probe must never take the others down
            out.append({"id": fn.__name__, "name": fn.__name__, "ok": False, "detail": "internal error: %s: %s" % (type(e).__name__, e)})
    return out
