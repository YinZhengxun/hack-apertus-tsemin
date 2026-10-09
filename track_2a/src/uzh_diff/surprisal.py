"""Method d4: let Apertus *read* a document instead of writing one.

For each side X of a document pair (the other side is Y):
  * conditioned call   : user = "here is the Y version, write the X version" ; assistant = X (prefilled)
  * unconditioned call : user = "write the X version"                        ; assistant = X (prefilled)
The request asks vLLM for the log-probability of every prompt token (prompt_logprobs=0) and generates
one token only. The log-probabilities of the prefilled X tokens are read back, summed per whitespace word,
and turned into

    s_c  = residual surprisal of the word after seeing Y   (−log p(word | Y, prefix))
    s_u  = surprisal of the word without Y                   (−log p(word | prefix))
    gain = s_u − s_c                                         (how much Y helped)

Words that Y explains are easy to predict after Y (low s_c); words whose content Y does not contain stay
surprising. This is the autoregressive successor of DiffMask (Vamvas & Sennrich, 2023), evaluated here on
long document pairs in both directions. The method emits continuous scores; how they are binarised or
combined with the block judgements of m2 is decided by `analyze`.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

from .client import ChatClient
from .data import LANG_NAMES, DocPair
from .methods import API_ERROR, INVALID_ANSWER, OK, SALVAGED, DocResult, _account, _zeros

D4_VERSION = "d4_readback_v1"
ASSISTANT_START = "<|assistant_start|>"

SYSTEM = ("You are a professional translator working for the Swiss federal administration. You translate "
          "web pages between English, German, French and Italian faithfully and completely, keeping the "
          "content, order and formatting of the source.")


def _user_conditioned(ref_lang: str, tgt_lang: str, ref_text: str) -> str:
    return ("Here is the %s version of a page of a Swiss government website:\n\n%s\n\n"
            "Write the %s version of the same page. Translate faithfully and completely; keep the same content "
            "in the same order." % (ref_lang, ref_text, tgt_lang))


def _user_unconditioned(tgt_lang: str) -> str:
    return "Write the %s version of a page of a Swiss government website." % tgt_lang


def _request(text: str, user: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                {"role": "assistant", "content": text}]
    params = {"temperature": 0, "max_tokens": 1, "add_generation_prompt": False, "continue_final_message": True,
              "prompt_logprobs": 0, "return_token_ids": True, "return_prompt_text": True}
    return messages, params


def word_surprisal(text: str, toks: List[str], probs: List[Optional[float]], prompt_text: str) -> Tuple[List[float], List[int], str]:
    """Sum −logprob over the tokens of each whitespace word of `text` (the prefilled assistant message).
    Returns (per-word surprisal, tokens per word, how the alignment was obtained)."""
    words = text.split()
    n_words = len(words)
    out = [0.0] * n_words
    cnt = [0] * n_words
    # 1) where does the prefilled text sit in the templated prompt?
    pos = prompt_text.rfind(ASSISTANT_START)
    if pos == -1:
        return out, cnt, "no_assistant_start"
    start = pos + len(ASSISTANT_START)
    tail = prompt_text[start:]
    how = "exact"
    if tail != text:
        how = "fuzzy"
    # 2) character offsets of the tokens, from the decoded strings (they concatenate to the prompt text after
    #    the first, undecoded token)
    recon = "".join(toks)
    lead = len(prompt_text) - len(recon)
    if lead < 0 or prompt_text[lead:] != recon:
        how = "fuzzy"
    # character -> word index within `tail`
    word_of_tail_char: List[int] = [-1] * len(tail)
    if tail == text:
        c = 0
        for k, w in enumerate(words):
            c = tail.index(w, c)
            for j in range(c, c + len(w)):
                word_of_tail_char[j] = k
            c += len(w)
    else:
        # map tail characters onto text characters with a sequence alignment, then onto words
        word_of_text_char: List[int] = [-1] * len(text)
        c = 0
        for k, w in enumerate(words):
            c = text.index(w, c)
            for j in range(c, c + len(w)):
                word_of_text_char[j] = k
            c += len(w)
        sm = difflib.SequenceMatcher(None, tail, text, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for d in range(i2 - i1):
                    word_of_tail_char[i1 + d] = word_of_text_char[j1 + d]
    # 3) walk the tokens; a token belongs to the word that contains its first non-space character
    offset = lead
    for tok, lp in zip(toks, probs):
        if lp is None:
            offset += len(tok)
            continue
        first = offset
        while first < offset + len(tok) and tok[first - offset].isspace():
            first += 1
        if first < offset + len(tok) and first >= start:
            k = word_of_tail_char[first - start] if first - start < len(tail) else -1
            if k >= 0:
                out[k] += -float(lp)
                cnt[k] += 1
        offset += len(tok)
    return out, cnt, how


def smooth_max(values: List[float], radius: int = 1) -> List[float]:
    n = len(values)
    return [max(values[max(0, i - radius): i + radius + 1]) for i in range(n)]


def run_d4(doc: DocPair, client: ChatClient, with_unconditioned: bool = True, smooth: int = 1) -> DocResult:
    labels_a, labels_b = _zeros(doc)
    result = DocResult(doc.id, doc.lang, "d4", D4_VERSION, client.cfg.name, OK, labels_a, labels_b)
    sides = {"a": (doc.text_a, "English", doc.text_b, LANG_NAMES[doc.lang]),
             "b": (doc.text_b, LANG_NAMES[doc.lang], doc.text_a, "English")}
    details: Dict[str, Any] = {"alignment": {}, "n_tokens": {}, "failed": []}
    for side, (text, tgt_lang, ref_text, ref_lang) in sides.items():
        n_words = len(text.split())
        scores: Dict[str, List[float]] = {}
        for kind, user in (("c", _user_conditioned(ref_lang, tgt_lang, ref_text)),
                           ("u", _user_unconditioned(tgt_lang))):
            if kind == "u" and not with_unconditioned:
                continue
            messages, params = _request(text, user)
            call = client.chat(messages, tag="d4:%s:%s:%s" % (doc.id, side, kind), **params)
            _account(result, call)
            ps = call.prompt_scores if call.ok else None
            if not ps or not isinstance(ps.get("prompt_text"), str):
                details["failed"].append({"side": side, "kind": kind, "error": call.error or "no prompt_logprobs in response"})
                continue
            per_word, cnt, how = word_surprisal(text, ps["tokens"], ps["logprobs"], ps["prompt_text"])
            details["alignment"]["%s_%s" % (side, kind)] = how
            details["n_tokens"]["%s_%s" % (side, kind)] = sum(cnt)
            if how == "no_assistant_start" or sum(cnt) == 0 or sum(1 for c in cnt if c) < max(1, n_words // 2):
                # the prefilled text could not be located in the templated prompt (different chat template?) or
                # fewer than half of the words received a token: an all-zero score would look like a valid answer
                details["failed"].append({"side": side, "kind": kind,
                                          "error": "alignment failed (%s): %d of %d words mapped" % (how, sum(1 for c in cnt if c), n_words)})
                continue
            scores[kind] = per_word
        if "c" not in scores:
            continue
        s_c = scores["c"]
        s_u = scores.get("u")
        details["s_c_" + side] = [round(v, 4) for v in s_c]
        if s_u is not None:
            details["s_u_" + side] = [round(v, 4) for v in s_u]
        smoothed = smooth_max(s_c, smooth) if smooth else list(s_c)
        if side == "a":
            result.labels_a = smoothed
        else:
            result.labels_b = smoothed
    n_fail = len(details["failed"])
    if n_fail and "s_c_a" not in details and "s_c_b" not in details:
        result.status = API_ERROR if all("prompt_logprobs" not in f["error"] for f in details["failed"]) else INVALID_ANSWER
        result.error = "; ".join(f["error"][:120] for f in details["failed"])
    elif n_fail:
        result.status = SALVAGED
        result.error = "%d of %d calls failed" % (n_fail, result.n_calls)
    result.details = details
    return result
