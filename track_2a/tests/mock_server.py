"""A tiny OpenAI-compatible server for offline tests. It never produces model output that is
reported anywhere: it only exists to exercise the plumbing (requests, retries, parsing, projection)."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self):
        self.counts = {}
        self.requests = []
        self.lock = threading.Lock()


def _m1_answer(user: str) -> str:
    blocks = dict(re.findall(r"^\[([ab]\d{3})\] (.*)$", user, flags=re.M))
    items = []
    if "a001" in blocks:
        words = blocks["a001"].split()
        items.append({"a": {"block_id": "a001", "quote": " ".join(words[:2])}, "b": None,
                      "category": "only_in_a", "explanation": "mock"})
    if "b000" in blocks:
        words = blocks["b000"].split()
        items.append({"a": None, "b": {"block_id": "b000", "quote": " ".join(words[-3:])},
                      "category": "only_in_b", "explanation": "mock"})
    items.append({"a": {"block_id": "a000", "quote": "THIS IS NOT IN THE TEXT"}, "b": None,
                  "category": "only_in_a", "explanation": "mock"})
    return json.dumps({"decision": "differences_found", "differences": items}, ensure_ascii=False)


def _m2_answer(user: str) -> str:
    """Mark the first block 'absent', the second 'partly' with its first two words, the rest 'covered'."""
    tail = user[user.find("BLOCKS"):]
    blocks = re.findall(r"^\[([ab]\d{3})\] (.*)$", tail, flags=re.M)
    items = []
    for k, (bid, text) in enumerate(blocks):
        if k == 0:
            items.append({"id": bid, "status": "absent", "quotes": []})
        elif k == 1:
            items.append({"id": bid, "status": "partly", "quotes": [" ".join(text.split()[:2]), "NOT IN TEXT"]})
        else:
            items.append({"id": bid, "status": "covered", "quotes": []})
    return json.dumps({"blocks": items}, ensure_ascii=False)


def _b0_answer(user: str) -> str:
    tail = user[user.rfind("## Input to Annotate"):]
    s1 = json.loads(re.search(r"^Sentence 1: (\[.*\])$", tail, flags=re.M).group(1))
    s2 = json.loads(re.search(r"^Sentence 2: (\[.*\])$", tail, flags=re.M).group(1))
    return json.dumps({"sentence1": [[t, 0 if i == 0 else 5] for i, t in enumerate(s1)],
                       "sentence2": [[t, 0 if i == 0 else 5] for i, t in enumerate(s2)]}, ensure_ascii=False)


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, obj, headers=None):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("x-ratelimit-remaining-requests", "99")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": "swiss-ai/Apertus-v1.5-8B"}, {"id": "swiss-ai/Apertus-v1.5-70B"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
            model = body.get("model", "")
            with state.lock:
                state.counts[model] = state.counts.get(model, 0) + 1
                n = state.counts[model]
                state.requests.append(body)
            if self.headers.get("Authorization") != "Bearer test-key":
                return self._send(401, {"error": "bad key"})
            if model == "mock-401":
                return self._send(401, {"error": "unauthorized"})
            if model == "mock-500":
                return self._send(500, {"error": "boom"})
            if model == "mock-504":
                return self._send(504, {"error": "upstream request timeout"})
            if model == "mock-429-once" and n == 1:
                return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
            if model == "mock-no-json-mode" and "response_format" in body:
                return self._send(400, {"error": {"message": "response_format is not supported by this server"}})
            if model == "mock-maxtok" and body.get("max_tokens", 0) > 1000:
                return self._send(400, {"error": {"message": "max_tokens is too large for this model"}})
            messages = body["messages"]
            user = messages[-1]["content"]
            system = messages[0]["content"] if messages[0]["role"] == "system" else ""
            # ---- capability probes (model "mock-full" honours the vLLM extras, "mock-plain" ignores them)
            extras = model == "mock-full"
            if messages[-1]["role"] == "assistant" and extras and body.get("prompt_logprobs") is not None and len(messages) == 3:
                # d4-style request: build a realistic templated prompt and one logprob per whitespace piece
                prefill = messages[-1]["content"]
                templated = ("<s><|system_start|>" + messages[0]["content"] + "<|system_end|><|user_start|>"
                             + messages[1]["content"] + "<|user_end|><|assistant_start|>" + prefill)
                pieces = re.findall(r"\s*\S+|\s+", templated[3:])
                conditioned = "version of a page of a Swiss government website:" in messages[1]["content"]
                lp = [None]
                for piece in pieces:
                    val = -1.0
                    if any(ch.isdigit() for ch in piece):
                        val = -5.0 if conditioned else -3.0
                    lp.append({"9": {"logprob": val, "rank": 1, "decoded_token": piece}})
                return self._send(200, {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": " x"}}],
                                        "usage": {"prompt_tokens": len(lp), "completion_tokens": 1},
                                        "prompt_logprobs": lp, "prompt_token_ids": list(range(len(lp))), "prompt_text": templated})
            if messages[-1]["role"] == "assistant":
                prefill = messages[-1]["content"]
                if extras and body.get("continue_final_message"):
                    content = " Bern." if "capital" in prefill else " Ja."
                    resp = {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                                          "message": {"role": "assistant", "content": content}}],
                            "usage": {"prompt_tokens": 12, "completion_tokens": 2}, "prompt_text": "<|assistant_start|>" + prefill}
                    if body.get("prompt_logprobs") is not None:
                        matching = "raining" in messages[0]["content"]
                        resp["prompt_logprobs"] = [None] + [{"7": {"logprob": -0.3 if matching else -2.5, "rank": 1, "decoded_token": "x"}} for _ in range(11)]
                        resp["prompt_token_ids"] = list(range(12))
                    return self._send(200, resp)
                return self._send(200, {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": "The capital of Switzerland is Bern."}}],
                                        "usage": {"prompt_tokens": 12, "completion_tokens": 8}, "prompt_text": "<|assistant_end|><|assistant_start|>"})
            if body.get("structured_outputs") and extras:
                choice = body["structured_outputs"].get("choice") or ["X"]
                return self._send(200, {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": choice[0]}}], "usage": {"prompt_tokens": 9, "completion_tokens": 1}})
            if (body.get("response_format") or {}).get("type") == "json_schema":
                content = '{"blocks": [{"id": "a000", "status": "covered"}, {"id": "a001", "status": "absent"}]}' if extras else "The sky is blue because of Rayleigh scattering."
                return self._send(200, {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 30, "completion_tokens": 20}})
            if body.get("n", 1) > 1:
                k = body["n"] if extras else 1
                return self._send(200, {"model": model, "choices": [{"index": i, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": ["quick", "rapid", "swift"][i % 3]}} for i in range(k)],
                                        "usage": {"prompt_tokens": 14, "completion_tokens": 3 * k}})
            if body.get("prompt_logprobs") is not None or body.get("logprobs"):
                resp = {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": "Bern"}}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 1}}
                if extras and body.get("prompt_logprobs") is not None:
                    resp["prompt_logprobs"] = [None] + [{"5": {"logprob": -1.0, "rank": 1, "decoded_token": "t"}} for _ in range(8)]
                    resp["prompt_token_ids"] = list(range(9))
                elif body.get("prompt_logprobs") is not None:
                    resp["prompt_logprobs"] = None
                if body.get("logprobs"):
                    resp["choices"][0]["logprobs"] = {"content": [{"token": "Bern", "logprob": -0.05, "bytes": [66], "top_logprobs": [{"token": "Bern", "logprob": -0.05}]}]} if extras else None
                return self._send(200, resp)
            if "REFERENCE (" in user and "BLOCKS (" in user:
                content = _m2_answer(user)
            elif "numbered blocks" in system:
                content = _m1_answer(user)
            elif user.startswith("Annotate a pair of"):
                content = _b0_answer(user)
            elif "PONG" in system:
                content = "PONG"
            elif '"answer"' in user:
                content = '{"answer": "ok"}'
            else:
                content = "OK"
            finish = "stop"
            if model == "mock-truncate":
                content, finish = content[: len(content) // 2], "length"
            self._send(200, {"model": model, "choices": [{"index": 0, "finish_reason": finish,
                                                           "message": {"role": "assistant", "content": content}}],
                             "usage": {"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                                       "completion_tokens": len(content) // 4}})
    return Handler


class MockServer:
    def __enter__(self):
        self.state = State()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
