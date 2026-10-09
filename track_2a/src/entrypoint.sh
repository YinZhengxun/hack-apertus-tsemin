#!/bin/sh
# Runs inside the container. Reproduces the submission: m2 + d4 on SPLIT, fused into the final prediction
# files, audited, and copied to output/ (mounted as ./output on the host when run through Docker).
#   SPLIT   test (default; 56 pages x 3 languages, 15-25 min at WORKERS=2) | full (all 224 pages, the organisers'
#           full-set order, 1-1.5 h) | dev | dev/train | dev/val | smoke (12 development pages, ~3 min)
#   WORKERS parallel documents (default 2; the hosted endpoint has a ~60 s per-request limit)
#   SYSTEM  name used in the prediction files (default tsemin)
# Endpoint: LLM_NAME, LLM_BASE_URL, LLM_API_KEY (the three variables of the Hack Apertus template).
#
# Every step must succeed; the script stops at the first failure and copies nothing to output/ in that case.
# Inference steps whose run ends with failed requests are repeated up to 3 times (cached requests are not re-sent).
# Step 2 is a hard gate: d4 needs prompt_logprobs on a prefilled assistant message (a vLLM extension); an endpoint
# without it ends the run with exit code 4 rather than producing m2-only predictions.
set -eu
cd "$(dirname "$0")/.."
SPLIT="${SPLIT:-test}"; WORKERS="${WORKERS:-2}"; SYSTEM="${SYSTEM:-tsemin}"
: "${LLM_API_KEY:?set LLM_API_KEY (and optionally LLM_BASE_URL, LLM_NAME)}"
mkdir -p runs output
STAMP="$(date +%Y%m%d%H%M%S)-$$"           # unique label suffix: every step below finds exactly its own run directory
LOG="output/run-${SPLIT}-${STAMP}.log"
echo "== model ${LLM_NAME:-swiss-ai/Apertus-v1.5-8B} at ${LLM_BASE_URL:-https://api.inference.cscs.ch/v1}; split ${SPLIT}; ${WORKERS} parallel documents; log ${LOG}"

step() {   # step <title> <command...>: run, keep the full output in the log, show the tail, stop on failure
    title="$1"; shift
    echo "== ${title}"
    echo "== ${title}" >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        return 0
    else
        rc=$?
        echo "-- step failed (exit ${rc}); last lines of ${LOG}:"
        tail -25 "$LOG"
        echo "== stopped: nothing copied to output/${SPLIT}/ (exit ${rc})"
        exit "$rc"
    fi
}

# inference <title> <command...>: like step, but a run that ends with failed requests (exit 5: the endpoint
# answered 5xx or timed out for some documents even after the client's retries) is repeated up to 3 times.
# Successful requests are cached, so a repetition only re-sends the failed ones. Anything else stops the run.
inference() {
    title="$1"; shift
    attempt=1
    while :; do
        echo "== ${title}$( [ "$attempt" -gt 1 ] && echo " (attempt ${attempt}: re-sending failed requests)" )"
        echo "== ${title} (attempt ${attempt})" >> "$LOG"
        "$@" >> "$LOG" 2>&1 && rc=0 || rc=$?
        if [ "$rc" -eq 0 ]; then
            return 0
        fi
        if [ "$rc" -eq 5 ] && [ "$attempt" -lt 3 ]; then
            grep "^INCOMPLETE\|^请求失败" "$LOG" | tail -2
            attempt=$((attempt + 1)); sleep 20; continue
        fi
        echo "-- step failed (exit ${rc}); last lines of ${LOG}:"
        tail -25 "$LOG"
        echo "== stopped: nothing copied to output/${SPLIT}/ (exit ${rc})"
        exit "$rc"
    done
}

step "1/6 offline self-test" python run.py selftest
tail -1 "$LOG"

step "2/6 endpoint capability gate (prompt_logprobs + assistant prefill, the form d4 needs)" \
    python run.py probe --only prompt_logprobs,prefill,prefill_logprobs --require T2b
grep -E "^\[(通过|失败)\]|^REQUIRED" "$LOG" | tail -4

ALLOW=""; case "$SPLIT" in test|full) ALLOW="--allow-test";; esac
if [ "$SPLIT" = "smoke" ]; then SEL="--manifest smoke12 --split dev/train"; else SEL="--split $SPLIT $ALLOW"; fi

inference "3/6 m2: block-level coverage judgements" \
    python run.py run --method m2 $SEL --workers "$WORKERS" --label "m2-${SPLIT}-${STAMP}"
M2="$(ls -d runs/*-m2-"${SPLIT}"-"${STAMP}" | tail -1)"      # the last attempt is the complete one
grep -v "^  m2 " "$LOG" | tail -8

inference "4/6 d4: read-back surprisal (prompt_logprobs, 4 requests per document)" \
    python run.py run --method d4 $SEL --workers "$WORKERS" --label "d4-${SPLIT}-${STAMP}"
D4="$(ls -d runs/*-d4-"${SPLIT}"-"${STAMP}" | tail -1)"
grep -v "^  d4 " "$LOG" | tail -8

step "5/6 fuse: verified absent blocks first, everything else ordered by the d4 signal" \
    python run.py fuse --m2-run "$M2" --d4-run "$D4" $ALLOW --system "$SYSTEM" --label "fuse-${SPLIT}-${STAMP}"
FU="$(ls -d runs/*-fuse-"${SPLIT}"-"${STAMP}")"
tail -12 "$LOG"

# 6/6: the audit exits 3 on a fatal problem (missing documents or d4 sides, length or alignment errors, prediction
# file not matching the gold order); declared limitations (a few m2 blocks without a judgement) are warnings only.
step "6/6 acceptance audit" python run.py audit --run "$FU" $ALLOW
sed -n '/验收摘要/,$p' "$LOG" | tail -30

mkdir -p "output/${SPLIT}"
cp "$FU"/predictions/fuse/*.jsonl.jsonl "output/${SPLIT}/"
cp "$FU"/summary_fuse.txt "$FU"/audit.txt "$FU"/manifest.json "output/${SPLIT}/"
echo "== done. prediction files:"; ls -la "output/${SPLIT}/"
