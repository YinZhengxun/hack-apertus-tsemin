"""The methods compared in this project. Each takes one document pair and returns token labels.

  b0  the baseline prompting setup of the SwissGov-RSD paper: the model labels every token with a
      similarity score 0-5 (three in-context examples); parsed with the semantics of the official
      evaluation code (unlabelled token = "no difference"; unreadable answer = all zeros).
  m1  direct comparison: the model sees both documents as numbered blocks and nominates the
      differences as exact quotes; quotes are located in the text and the covered tokens get 1.
  m2  blockwise coverage: for a handful of blocks of one document at a time, with the complete other
      document as reference, the model says per block whether it is covered, absent, or partly
      covered (with short quotes of the differing words). Both directions. Short outputs, so every
      call stays far below the gateway's 60-second limit.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import config, project, schema
from .blocks import Block, make_blocks, render
from .client import ChatClient
from .data import LANG_NAMES, DocPair

OK = "ok"
API_ERROR = "api_error"            # no answer from the endpoint after retries
INVALID_ANSWER = "invalid_answer"  # an answer came back but could not be read
TRUNCATED = "truncated"            # answer cut off by the output limit (b0: scored as all zeros)
SALVAGED = "salvaged"              # answer cut off; the complete items were kept (m1)

FAILED = (API_ERROR, INVALID_ANSWER, TRUNCATED)


@dataclass
class DocResult:
    doc_id: str
    lang: str
    method: str
    prompt_version: str
    model: str
    status: str
    labels_a: List[float]
    labels_b: List[float]
    n_calls: int = 0
    n_cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    raw: List[Optional[str]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status in FAILED


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _account(result: DocResult, call) -> None:
    result.n_calls += 1
    result.n_cached += 1 if call.cached else 0
    result.prompt_tokens += call.prompt_tokens or 0
    result.completion_tokens += call.completion_tokens or 0
    result.latency_s = round(result.latency_s + (0.0 if call.cached else call.latency_s), 3)
    result.finish_reason = call.finish_reason
    result.raw.append(call.content)


def _zeros(doc: DocPair):
    return [0.0] * len(doc.tokens_a), [0.0] * len(doc.tokens_b)


# ----------------------------------------------------------------------------- M1
M1_VERSION = "m1_direct_v1"


def _m1_system(lang: str) -> str:
    template = (config.PROMPTS / (M1_VERSION + ".txt")).read_text(encoding="utf-8")
    return template.replace("{{LANGUAGE}}", LANG_NAMES[lang]).strip()


def m1_messages(doc: DocPair, blocks_a: List[Block], blocks_b: List[Block]) -> List[Dict[str, str]]:
    user = ("DOCUMENT A (English)\n%s\n\nDOCUMENT B (%s)\n%s\n\nReturn the JSON object now."
            % (render(blocks_a), LANG_NAMES[doc.lang], render(blocks_b)))
    return [{"role": "system", "content": _m1_system(doc.lang)}, {"role": "user", "content": user}]


def _resolve_block(block_id: str, by_id: Dict[str, Block], side: str) -> Optional[Block]:
    """Accept "a003", "A3", "[a003]", "3" (side known from context)."""
    if block_id in by_id:
        return by_id[block_id]
    m = re.search(r"([abAB])?\s*0*(\d+)", block_id or "")
    if not m:
        return None
    letter = (m.group(1) or side).lower()
    if letter != side:
        return None
    return by_id.get("%s%03d" % (side, int(m.group(2))))


def run_m1(doc: DocPair, client: ChatClient, max_tokens: int = 6000) -> DocResult:
    blocks = {"a": make_blocks(doc.tokens_a, "a"), "b": make_blocks(doc.tokens_b, "b")}
    by_id = {side: {b.id: b for b in bl} for side, bl in blocks.items()}
    tokens = {"a": doc.tokens_a, "b": doc.tokens_b}
    labels_a, labels_b = _zeros(doc)
    labels = {"a": labels_a, "b": labels_b}
    template = (config.PROMPTS / (M1_VERSION + ".txt")).read_text(encoding="utf-8")
    result = DocResult(doc.id, doc.lang, "m1", "%s@%s" % (M1_VERSION, prompt_hash(template)), client.cfg.name,
                       OK, labels_a, labels_b)
    call = client.chat(m1_messages(doc, blocks["a"], blocks["b"]), tag="m1:" + doc.id,
                       temperature=0, max_tokens=max_tokens, seed=0, response_format={"type": "json_object"})
    _account(result, call)
    if not call.ok:
        result.status, result.error = API_ERROR, call.error
        return result
    try:
        parsed = schema.parse_m1(call.content or "")
    except schema.InvalidAnswer as e:
        result.status, result.error = INVALID_ANSWER, str(e)
        return result
    if parsed.salvaged or call.finish_reason == "length":
        result.status = SALVAGED
    located = []
    counts: Dict[str, int] = {}
    for nom in parsed.nominations:
        block = _resolve_block(nom.block_id, by_id[nom.side], nom.side)
        span = (block.start, block.end) if block else None
        loc = project.locate(nom.quote, tokens[nom.side], span)
        project.apply(labels[nom.side], loc)
        counts[loc.status] = counts.get(loc.status, 0) + 1
        located.append({"item": nom.item, "side": nom.side, "block_id": nom.block_id, "quote": nom.quote,
                        "category": nom.category, "explanation": nom.explanation,
                        "status": loc.status, "how": loc.how, "start": loc.start, "end": loc.end,
                        "n_matches": loc.n_matches, "block_known": block is not None})
    result.details = {
        "decision": parsed.decision, "n_items": parsed.n_items, "n_quotes": len(parsed.nominations),
        "location_counts": counts, "format_problems": parsed.problems, "quotes": located,
        "n_blocks_a": len(blocks["a"]), "n_blocks_b": len(blocks["b"]),
        "adjustments": call.adjustments,
    }
    return result


# ----------------------------------------------------------------------------- B0
B0_VERSION = "b0_paper_template"


def _b0_prompt(doc: DocPair) -> str:
    template = (config.PROMPTS / "b0_paper" / ("template_%s_admin.txt" % doc.lang)).read_text(encoding="utf-8")
    return (template.replace("{{ sentence1 }}", json.dumps(doc.tokens_a, ensure_ascii=False))
                    .replace("{{ sentence2 }}", json.dumps(doc.tokens_b, ensure_ascii=False)))


def _b0_json_str(content: str) -> str:
    """Same clean-up as evaluation/predictions.py (get_json_str) in the dataset repository."""
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    if not content.startswith("{") and "{" in content:
        content = content[content.index("{"):]
    if "```\n" in content:
        content = content.split("```\n")[0]
    return content


def _b0_load(content: str) -> Any:
    text = _b0_json_str(content)
    try:
        return json.loads(text)
    except ValueError:
        try:
            return ast.literal_eval(text)     # the reference code uses eval(); literal_eval is the safe equivalent
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return {}


def b0_token_labels(tokens: List[str], predictions: Any, fallback: float = 5.0) -> List[float]:
    """Same walk as evaluation/utils.py (parse_token_labels): an emitted token is matched against the
    current gold position or the next one; anything else is skipped; unmatched positions keep the
    fallback (similarity 5 = no difference)."""
    labels = [fallback] * len(tokens)
    try:
        predictions = list(predictions)
    except TypeError:
        return labels
    i = 0
    for entry in predictions:
        if i >= len(tokens):
            break
        if isinstance(entry, str):
            token, rest = entry, []
        else:
            try:
                entry = list(entry)
            except TypeError:
                continue
            if not entry:
                continue
            token, rest = entry[0], entry[1:]
        if token == tokens[i]:
            if rest:
                try:
                    labels[i] = float(rest[0])
                except (TypeError, ValueError):
                    continue
            i += 1
        elif i + 1 < len(tokens) and token == tokens[i + 1]:
            i += 1
            if rest:
                try:
                    labels[i] = float(rest[0])
                except (TypeError, ValueError):
                    continue
            i += 1
    return labels


def _to_difference(similarity: float) -> float:
    return 1.0 - (similarity / 5.0) if similarity >= 0 else -1.0


def run_b0(doc: DocPair, client: ChatClient, max_tokens_cap: int = 32000) -> DocResult:
    prompt = _b0_prompt(doc)
    labels_a, labels_b = _zeros(doc)
    template_hash = prompt_hash((config.PROMPTS / "b0_paper" / ("template_%s_admin.txt" % doc.lang)).read_text(encoding="utf-8"))
    result = DocResult(doc.id, doc.lang, "b0", "%s@%s" % (B0_VERSION, template_hash), client.cfg.name,
                       OK, labels_a, labels_b)
    # every token is echoed with a score: budget roughly 12 output tokens per word
    max_tokens = min(max_tokens_cap, 12 * doc.n_tokens + 400)
    call = client.chat([{"role": "user", "content": prompt}], tag="b0:" + doc.id,
                       temperature=0, max_tokens=max_tokens, seed=0, response_format={"type": "json_object"})
    _account(result, call)
    if not call.ok:
        result.status, result.error = API_ERROR, call.error
        return result
    data = _b0_load(call.content or "")
    found = {"a": False, "b": False}
    if isinstance(data, dict):
        if "sentence1" in data:
            sims = b0_token_labels(doc.tokens_a, data["sentence1"])
            result.labels_a = [_to_difference(s) for s in sims]   # -1 (model says punctuation) stays -1, as in the reference code
            found["a"] = True
        if "sentence2" in data:
            sims = b0_token_labels(doc.tokens_b, data["sentence2"])
            result.labels_b = [_to_difference(s) for s in sims]
            found["b"] = True
    if call.finish_reason == "length":
        result.status = TRUNCATED
    elif not (found["a"] and found["b"]):
        result.status = INVALID_ANSWER
        result.error = "answer has no readable sentence1/sentence2"
    result.details = {
        "sides_read": found,
        "labelled_a": sum(1 for v in result.labels_a if v > 0), "labelled_b": sum(1 for v in result.labels_b if v > 0),
        "max_tokens": call.max_tokens_used, "adjustments": call.adjustments,
    }
    return result


# ----------------------------------------------------------------------------- M2
M2_VERSION = "m2_coverage_v1"
M2_STATUSES = ("covered", "absent", "partly")


def _m2_system(ref_lang: str, block_lang: str) -> str:
    template = (config.PROMPTS / (M2_VERSION + ".txt")).read_text(encoding="utf-8")
    return template.replace("{{REF_LANG}}", ref_lang).replace("{{BLOCK_LANG}}", block_lang).strip()


def reference_window(ref_tokens: List[str], ref_total: int, batch_start: int, batch_end: int, side_total: int,
                     max_words: int) -> Tuple[str, Tuple[int, int]]:
    """The reference text given to the model. Whole document when it is short enough; otherwise a window of
    max_words tokens centred on the position that corresponds proportionally to the batch, which is wide
    enough to absorb the shifts that insertions and deletions cause (returns the text and the window)."""
    n = len(ref_tokens)
    if n <= max_words:
        return " ".join(ref_tokens), (0, n)
    centre = int(round((batch_start + batch_end) / 2.0 / max(1, side_total) * n))
    start = max(0, min(n - max_words, centre - max_words // 2))
    end = start + max_words
    return " ".join(ref_tokens[start:end]), (start, end)


def m2_messages(doc: DocPair, side: str, batch: List[Block], max_ref_words: int = 1500) -> List[Dict[str, str]]:
    """side is the side the blocks come from; the other side is the reference."""
    if side == "a":
        ref_lang, block_lang, ref_tokens, side_total = LANG_NAMES[doc.lang], "English", doc.tokens_b, len(doc.tokens_a)
    else:
        ref_lang, block_lang, ref_tokens, side_total = "English", LANG_NAMES[doc.lang], doc.tokens_a, len(doc.tokens_b)
    reference, window = reference_window(ref_tokens, len(ref_tokens), batch[0].start, batch[-1].end, side_total, max_ref_words)
    scope = "complete" if window == (0, len(ref_tokens)) else "the part that corresponds to these blocks"
    user = ("REFERENCE (%s, %s):\n%s\n\nBLOCKS (%s):\n%s\n\nReturn the JSON object for these %d blocks."
            % (ref_lang, scope, reference, block_lang, render(batch), len(batch)))
    return [{"role": "system", "content": _m2_system(ref_lang, block_lang)}, {"role": "user", "content": user}]


def _parse_m2(text: str) -> Tuple[Dict[str, dict], bool, List[str]]:
    """Return ({block_id: {"status", "quotes"}}, salvaged, problems)."""
    obj, salvaged = schema.extract_json(text)
    if not isinstance(obj, dict):
        raise schema.InvalidAnswer("top level is not an object")
    items = obj.get("blocks")
    if items is None and isinstance(obj.get("id"), str):     # a single block object
        items = [obj]
    if not isinstance(items, list):
        raise schema.InvalidAnswer('"blocks" is not a list')
    out: Dict[str, dict] = {}
    problems: List[str] = []
    for k, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append("item %d is not an object" % k)
            continue
        bid = str(item.get("id") or item.get("block_id") or "")
        status = str(item.get("status") or "").strip().lower()
        if status not in M2_STATUSES:
            problems.append("block %s: unknown status %r" % (bid, status))
            status = "partly" if item.get("quotes") else "covered"
        quotes = item.get("quotes") or []
        if isinstance(quotes, str):
            quotes = [quotes]
        quotes = [q for q in quotes if isinstance(q, str) and q.strip()]
        out[bid] = {"status": status, "quotes": quotes}
    return out, salvaged, problems


TIER_ABSENT, TIER_QUOTE = 1, 2     # 1: token in a block judged "absent"; 2: token covered by a quote in a "partly" block


def run_m2(doc: DocPair, client: ChatClient, blocks_per_call: int = 3, max_tokens: int = 900,
           max_ref_words: int = 1500) -> DocResult:
    blocks = {"a": make_blocks(doc.tokens_a, "a"), "b": make_blocks(doc.tokens_b, "b")}
    tokens = {"a": doc.tokens_a, "b": doc.tokens_b}
    labels_a, labels_b = _zeros(doc)
    labels = {"a": labels_a, "b": labels_b}
    tiers = {"a": [0] * len(doc.tokens_a), "b": [0] * len(doc.tokens_b)}
    windows: List[dict] = []
    template = (config.PROMPTS / (M2_VERSION + ".txt")).read_text(encoding="utf-8")
    result = DocResult(doc.id, doc.lang, "m2", "%s@%s" % (M2_VERSION, prompt_hash(template)), client.cfg.name,
                       OK, labels_a, labels_b)
    per_block: List[dict] = []
    quotes_out: List[dict] = []
    counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    failed_calls = 0
    n_calls = 0
    salvaged = False
    for side in ("a", "b"):
        side_blocks = blocks[side]
        step = blocks_per_call if blocks_per_call > 0 else max(1, len(side_blocks))
        for start in range(0, len(side_blocks), step):
            batch = side_blocks[start:start + step]
            n_calls += 1
            msgs = m2_messages(doc, side, batch, max_ref_words=max_ref_words)
            call = client.chat(msgs, tag="m2:%s:%s:%d" % (doc.id, side, start),
                               temperature=0, max_tokens=max_tokens, seed=0, response_format={"type": "json_object"})
            _account(result, call)
            if not call.ok:
                failed_calls += 1
                for b in batch:
                    per_block.append({"id": b.id, "status": "call_failed", "error": (call.error or "")[:200]})
                continue
            try:
                parsed, was_salvaged, problems = _parse_m2(call.content or "")
            except schema.InvalidAnswer as e:
                failed_calls += 1
                for b in batch:
                    per_block.append({"id": b.id, "status": "unreadable", "error": str(e)})
                continue
            salvaged = salvaged or was_salvaged or call.finish_reason == "length"
            by_id = {b.id: b for b in batch}
            for b in batch:
                entry = parsed.get(b.id)
                if entry is None:
                    # tolerate "a3" / "A003" style ids
                    for key, val in parsed.items():
                        if _resolve_block(key, by_id, side) is b:
                            entry = val
                            break
                if entry is None:
                    per_block.append({"id": b.id, "status": "missing_in_answer"})
                    continue
                status = entry["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                rec = {"id": b.id, "status": status, "n_quotes": len(entry["quotes"])}
                if status == "absent":
                    for i in range(b.start, b.end):
                        labels[side][i] = 1.0
                        tiers[side][i] = TIER_ABSENT
                elif status == "partly":
                    for q in entry["quotes"]:
                        loc = project.locate(q, tokens[side], (b.start, b.end))
                        project.apply(labels[side], loc)
                        if loc.status in project.COUNTED and loc.start is not None:
                            for i in range(loc.start, loc.end):
                                if tiers[side][i] == 0:
                                    tiers[side][i] = TIER_QUOTE
                        counts[loc.status] = counts.get(loc.status, 0) + 1
                        quotes_out.append({"side": side, "block_id": b.id, "quote": q, "status": loc.status,
                                           "how": loc.how, "start": loc.start, "end": loc.end, "n_matches": loc.n_matches})
                per_block.append(rec)
            for pr in problems:
                per_block.append({"id": "-", "status": "format_problem", "error": pr})
    # blocks the model was asked about but did not judge (not in the answer, call failed, unreadable answer):
    # they keep the default "no difference" and must be visible as such, not as a model judgement
    unjudged = [p for p in per_block if p.get("status") in ("missing_in_answer", "call_failed", "unreadable")]
    if failed_calls == n_calls:
        result.status = API_ERROR if any(p.get("status") == "call_failed" for p in per_block) else INVALID_ANSWER
        result.error = "all %d calls failed" % n_calls
    elif failed_calls:
        result.status = SALVAGED
        result.error = "%d of %d calls failed; their blocks are scored as no difference" % (failed_calls, n_calls)
    elif unjudged:
        result.status = SALVAGED
        result.error = "%d of %d blocks not judged (missing in the answer); scored as no difference" % (
            len(unjudged), len(blocks["a"]) + len(blocks["b"]))
    elif salvaged:
        result.status = SALVAGED
    result.details = {
        "blocks_per_call": blocks_per_call, "max_ref_words": max_ref_words,
        "n_blocks_a": len(blocks["a"]), "n_blocks_b": len(blocks["b"]),
        "calls": n_calls, "failed_calls": failed_calls, "unjudged_blocks": len(unjudged),
        "block_status_counts": status_counts,
        "location_counts": counts, "blocks": per_block, "quotes": quotes_out,
        "tiers_a": tiers["a"], "tiers_b": tiers["b"],
    }
    return result


def paper_chunks(n_a: int, n_b: int, max_tokens: int = 250) -> List[Tuple[int, int, int, int]]:
    """The chunking rule of the dataset repository (scripts/create_test_data_admin.py, '_short' files):
    up to max_tokens/2 tokens of side a, then side b fills the rest, until both sides are used up.
    Returns (start_a, end_a, start_b, end_b) spans."""
    if n_a + n_b <= max_tokens:
        return [(0, n_a, 0, n_b)]
    out = []
    sa = sb = 0
    while sa < n_a or sb < n_b:
        remaining = max_tokens
        ea = min(sa + remaining // 2, n_a)
        remaining -= ea - sa
        eb = min(sb + remaining, n_b)
        out.append((sa, ea, sb, eb))
        sa, sb = ea, eb
    return out


def run_b0s(doc: DocPair, client: ChatClient, max_tokens_cap: int = 6000) -> DocResult:
    """b0 on the paper's short chunks: the same prompt and parser as b0, one call per chunk pair,
    scores written back to the full-document positions. This is how the paper ran models that could
    not take whole documents, and it keeps every call far below the gateway's time limit."""
    labels_a, labels_b = _zeros(doc)
    template_path = config.PROMPTS / "b0_paper" / ("template_%s_admin.txt" % doc.lang)
    template = template_path.read_text(encoding="utf-8")
    result = DocResult(doc.id, doc.lang, "b0s", "%s@%s" % (B0_VERSION + "_short", prompt_hash(template)), client.cfg.name,
                       OK, labels_a, labels_b)
    chunks = paper_chunks(len(doc.tokens_a), len(doc.tokens_b))
    statuses: Dict[str, int] = {}
    for k, (sa, ea, sb, eb) in enumerate(chunks):
        ta, tb = doc.tokens_a[sa:ea], doc.tokens_b[sb:eb]
        prompt = (template.replace("{{ sentence1 }}", json.dumps(ta, ensure_ascii=False))
                          .replace("{{ sentence2 }}", json.dumps(tb, ensure_ascii=False)))
        max_tokens = min(max_tokens_cap, 12 * (len(ta) + len(tb)) + 300)
        call = client.chat([{"role": "user", "content": prompt}], tag="b0s:%s:%d" % (doc.id, k),
                           temperature=0, max_tokens=max_tokens, seed=0, response_format={"type": "json_object"})
        _account(result, call)
        if not call.ok:
            statuses[API_ERROR] = statuses.get(API_ERROR, 0) + 1
            continue
        data = _b0_load(call.content or "")
        got = 0
        if isinstance(data, dict):
            if "sentence1" in data and ta:
                sims = b0_token_labels(ta, data["sentence1"])
                result.labels_a[sa:ea] = [_to_difference(x) for x in sims]
                got += 1
            if "sentence2" in data and tb:
                sims = b0_token_labels(tb, data["sentence2"])
                result.labels_b[sb:eb] = [_to_difference(x) for x in sims]
                got += 1
        if call.finish_reason == "length":
            statuses[TRUNCATED] = statuses.get(TRUNCATED, 0) + 1
        elif got < (1 if not ta or not tb else 2):
            statuses[INVALID_ANSWER] = statuses.get(INVALID_ANSWER, 0) + 1
        else:
            statuses[OK] = statuses.get(OK, 0) + 1
    n_bad = sum(v for k, v in statuses.items() if k != OK)
    if n_bad == len(chunks):
        result.status = API_ERROR if statuses.get(API_ERROR) == len(chunks) else INVALID_ANSWER
        result.error = "no chunk produced a readable answer"
    elif n_bad:
        result.status = SALVAGED
        result.error = "%d of %d chunks failed (%s); their tokens are scored as no difference" % (
            n_bad, len(chunks), ", ".join("%s=%d" % kv for kv in sorted(statuses.items())))
    result.details = {"chunks": len(chunks), "chunk_statuses": statuses,
                      "labelled_a": sum(1 for v in result.labels_a if v > 0), "labelled_b": sum(1 for v in result.labels_b if v > 0)}
    return result


METHODS = {"m1": run_m1, "b0": run_b0, "b0s": run_b0s, "m2": run_m2}


def get_method(name: str):
    """Look a method up by name; d4 lives in surprisal.py, which imports this module, so it is loaded lazily."""
    if name == "d4" and "d4" not in METHODS:
        from .surprisal import run_d4
        METHODS["d4"] = run_d4
    return METHODS[name]
