"""Offline HTML viewer: every document pair side by side, words shaded by the predicted score, gold
differences marked, per-document and pooled Spearman, optional reference system for comparison.

    python run.py viewer --pred-dir data/predictions/full --system tsemin --allow-test --out demo/viewer.html

The result is ONE self-contained HTML file (no network access, no external scripts or fonts); open it in
any browser. Nothing here calls the model or changes predictions: the viewer only reads prediction files,
the gold files of the same documents and, when present, the reference predictions in data/reference.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .data import LANGS, LANG_NAMES, DocPair, numeric_id, read_jsonl, split_ids
from .stats import pooled_spearman

Labels = Tuple[Sequence[float], Sequence[float]]

REFERENCE_SYSTEMS = {
    # display name: (dev file pattern, test file pattern); %s = lang
    "CTFAlign": ("ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl", "test/ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl"),
}


# ----------------------------------------------------------------------------- loading

def load_predictions(pred_dir: Path, system: str, lang: str) -> Dict[str, Labels]:
    path = pred_dir / ("%s_admin_%s.jsonl.jsonl" % (system, lang))
    if not path.exists():
        raise FileNotFoundError(path)
    return {r["id"]: (r["labels_a"], r["labels_b"]) for r in read_jsonl(path)}


def load_reference(lang: str, allow_test: bool) -> Dict[str, Dict[str, Labels]]:
    """Reference predictions per display name; dev and test files merged, missing files skipped."""
    out: Dict[str, Dict[str, Labels]] = {}
    for name, (dev_pat, test_pat) in REFERENCE_SYSTEMS.items():
        merged: Dict[str, Labels] = {}
        for pat, is_test in ((dev_pat, False), (test_pat, True)):
            if is_test and not allow_test:
                continue
            path = config.DATA / "reference" / (pat % lang)
            if path.exists():
                merged.update({r["id"]: (r["labels_a"], r["labels_b"]) for r in read_jsonl(path)})
        if merged:
            out[name] = merged
    return out


def page_url(name: str) -> str:
    """The dataset stores page names as file names: 'https___www.x.ch_a_b.html' -> 'https://www.x.ch/a/b.html'.
    Underscores that were part of the original path cannot be told apart from separators (rare)."""
    if not name:
        return ""
    if "___" in name:
        scheme, rest = name.split("___", 1)
        return scheme + "://" + rest.replace("_", "/")
    return name.replace("_", "/")


def page_title(name: str) -> str:
    url = page_url(name)
    last = url.rstrip("/").rsplit("/", 1)[-1]
    if last.endswith(".html") or last.endswith(".htm"):
        last = last.rsplit(".", 1)[0]
    return last.replace("-", " ")


def gold_docs(lang: str, allow_test: bool) -> Tuple[List[DocPair], Dict[str, str], Dict[str, dict]]:
    """All gold documents of a language we may look at, with split name and page metadata per id."""
    split_of: Dict[str, str] = {}
    meta: Dict[str, dict] = {}
    rows = read_jsonl(config.GOLD_DEV / ("gold_admin_%s.jsonl" % lang))
    val_ids = set(split_ids(lang, "dev/val") or [])
    for r in rows:
        split_of[r["id"]] = "dev/val" if r["id"] in val_ids else "dev/train"
    if allow_test:
        test_path = config.DATA / "gold" / "test" / ("gold_admin_%s.jsonl" % lang)
        if test_path.exists():
            test_rows = read_jsonl(test_path)
            for r in test_rows:
                split_of[r["id"]] = "test"
            rows = rows + test_rows
    for r in rows:
        meta[r["id"]] = {"page_en": page_url(r.get("page_en", "")), "page_other": page_url(r.get("page_other", "")),
                         "title": page_title(r.get("page_en", ""))}
    rows = sorted(rows, key=lambda r: numeric_id(r["id"]))
    docs = [DocPair(r["id"], lang, r["text_a"].split(), r["text_b"].split(), list(r["labels_a"]), list(r["labels_b"]))
            for r in rows]
    return docs, split_of, meta


# ----------------------------------------------------------------------------- statistics

def _nan_to_none(x: float):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 4)


def doc_rho(doc: DocPair, labels: Labels) -> Dict[str, Optional[float]]:
    """Spearman inside one document: pooled over both sides, and per side (None when undefined)."""
    pa, pb = labels
    g_all, p_all = [], []
    res = {}
    for key, gold, pred in (("a", doc.labels_a, pa), ("b", doc.labels_b, pb)):
        g = [float(x) for x, y in zip(gold, pred) if x != -1]
        p = [float(y) for x, y in zip(gold, pred) if x != -1]
        g_all += g
        p_all += p
        res["rho_" + key] = _nan_to_none(pooled_spearman(g, p)) if len(g) > 1 else None
    res["rho"] = _nan_to_none(pooled_spearman(g_all, p_all)) if len(g_all) > 1 else None
    return res


def pooled_scores(docs: Sequence[DocPair], preds: Dict[str, Labels], ids: Sequence[str]) -> Optional[float]:
    g, p = [], []
    by_id = {d.id: d for d in docs}
    for i in ids:
        d = by_id[i]
        pa, pb = preds[i]
        for gold, pred in ((d.labels_a, pa), (d.labels_b, pb)):
            for x, y in zip(gold, pred):
                if x != -1:
                    g.append(float(x))
                    p.append(float(y))
    return _nan_to_none(pooled_spearman(g, p)) if len(g) > 1 else None


# ----------------------------------------------------------------------------- payload

def _q(values: Sequence[float], scale: int = 100) -> List[int]:
    """Quantise scores to integers 0..scale (gold -1 stays -1); keeps the file small."""
    out = []
    for v in values:
        v = float(v)
        if v == -1:
            out.append(-1)
        else:
            out.append(int(round(min(max(v, 0.0), 1.0) * scale)))
    return out


def build_payload(pred_dir: Path, system: str, allow_test: bool, system_label: str = "Fuse",
                  langs: Sequence[str] = LANGS, title: str = "") -> dict:
    payload = {
        "title": title or "SwissGov-RSD · word-level semantic differences · %s predictions (team Tse-min, Hack Apertus 2026, Track 2A)" % system_label,
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "main": system_label,
        "systems": [system_label],
        "langs": {},
        "summary": {},   # lang -> split -> system -> pooled Spearman
    }
    for lang in langs:
        docs, split_of, meta = gold_docs(lang, allow_test)
        preds = load_predictions(pred_dir, system, lang)
        refs = load_reference(lang, allow_test)
        for name in refs:
            if name not in payload["systems"]:
                payload["systems"].append(name)
        docs = [d for d in docs if d.id in preds]
        if not docs:
            raise ValueError("%s: no predicted document has gold labels" % lang)
        entries = []
        for d in docs:
            pa, pb = preds[d.id]
            if len(pa) != len(d.tokens_a) or len(pb) != len(d.tokens_b):
                raise ValueError("%s: prediction length does not match the token count" % d.id)
            systems = {system_label: {"a": _q(pa), "b": _q(pb)}}
            stats = {system_label: doc_rho(d, (pa, pb))}
            for name, ref in refs.items():
                if d.id in ref:
                    ra, rb = ref[d.id]
                    if len(ra) == len(d.tokens_a) and len(rb) == len(d.tokens_b):
                        systems[name] = {"a": _q(ra), "b": _q(rb)}
                        stats[name] = doc_rho(d, (ra, rb))
            gold_scored = sum(1 for x in d.labels_a + d.labels_b if x != -1)
            gold_diff = sum(1 for x in d.labels_a + d.labels_b if x != -1 and x > 0)
            entries.append({
                "id": d.id, "split": split_of.get(d.id, "?"),
                "page_en": meta.get(d.id, {}).get("page_en", ""), "page_other": meta.get(d.id, {}).get("page_other", ""),
                "title": meta.get(d.id, {}).get("title", ""),
                "ta": d.text_a, "tb": d.text_b,
                "ga": _q(d.labels_a), "gb": _q(d.labels_b),
                "sys": systems, "st": stats,
                "n_scored": gold_scored, "n_gold_diff": gold_diff,
                "n_flagged": sum(1 for v in list(pa) + list(pb) if float(v) >= 0.5),
            })
        payload["langs"][lang] = {"name": LANG_NAMES[lang], "docs": entries}
        # pooled scores per split
        summary = {}
        groups = {"all": [d.id for d in docs]}
        for d in docs:
            sp = split_of.get(d.id, "?")
            top = "test" if sp == "test" else "dev"
            groups.setdefault(top, []).append(d.id)
        for grp, ids in groups.items():
            summary[grp] = {"n": len(ids), system_label: pooled_scores(docs, preds, ids)}
            for name, ref in refs.items():
                have = [i for i in ids if i in ref]
                summary[grp][name] = pooled_scores(docs, ref, have) if len(have) == len(ids) and have else None
        payload["summary"][lang] = summary
    return payload


# ----------------------------------------------------------------------------- html

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1f2328;--muted:#656d76;--line:#d8dee4;--accent:#0969da;
  --hit:#2da44e;--miss:#cf222e;--false:#8250df;--punct:#9aa3ad}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px}
header h1{font-size:17px;margin:0 0 4px}
header .sub{color:var(--muted);font-size:12.5px}
.summary{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px}
.summary table{border-collapse:collapse;font-size:12.5px;background:var(--panel)}
.summary th,.summary td{border:1px solid var(--line);padding:2px 7px;text-align:right}
.summary th:first-child,.summary td:first-child{text-align:left}
.summary caption{text-align:left;font-weight:600;padding:0 0 3px;font-size:12.5px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:8px 16px;background:var(--panel);
  border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.controls label{display:flex;gap:4px;align-items:center;font-size:12.5px;color:var(--muted)}
.controls select,.controls input[type=range]{font:inherit}
.controls .sep{width:1px;height:22px;background:var(--line)}
main{display:grid;grid-template-columns:300px 1fr;gap:0;min-height:calc(100vh - 150px)}
aside{border-right:1px solid var(--line);background:var(--panel);max-height:calc(100vh - 150px);overflow:auto}
aside .row{display:grid;grid-template-columns:1fr auto auto;gap:6px;padding:5px 10px;border-bottom:1px solid #eef1f4;
  cursor:pointer;font-size:12.5px;align-items:center}
aside .row:hover{background:#f0f4ff}
aside .row.active{background:#dbe7ff}
aside .row .id{font-weight:600}
aside .row .ttl{color:var(--muted);font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;grid-column:1/-1}
aside .row .tag{font-size:10.5px;padding:0 5px;border-radius:8px;background:#eaeef2;color:var(--muted)}
aside .row .tag.test{background:#ffe9c7;color:#8a4b00}
aside .row .rho{font-variant-numeric:tabular-nums;color:var(--muted)}
section.doc{padding:12px 16px;overflow:auto}
.dochead{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline;margin-bottom:8px}
.dochead h2{font-size:15px;margin:0}
.dochead a{color:var(--accent);text-decoration:none;font-size:12.5px}
.dochead a:hover{text-decoration:underline}
.stats{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--muted);margin-bottom:10px}
.stats b{color:var(--ink);font-variant-numeric:tabular-nums}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.col{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px;line-height:1.9}
.col h3{font-size:12.5px;color:var(--muted);margin:0 0 6px;font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.w{padding:1px 2px;border-radius:3px;border-bottom:3px solid transparent;cursor:default}
.w.p{color:var(--punct)}
.w:hover{outline:1px solid var(--accent)}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px;color:var(--muted);margin:10px 0 0}
.legend .sw{display:inline-block;width:14px;height:12px;vertical-align:-2px;border-radius:2px;margin-right:4px;border-bottom:3px solid transparent}
#tip{position:fixed;pointer-events:none;background:#1f2328;color:#fff;font-size:12px;padding:5px 8px;border-radius:4px;
  z-index:20;display:none;max-width:320px;white-space:pre-line}
.hint{color:var(--muted);font-size:12px;margin-top:10px}
@media (max-width:900px){main{grid-template-columns:1fr}aside{max-height:220px;border-right:0;border-bottom:1px solid var(--line)}
  .cols{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">Shading = predicted difference score (0–1); the gold word-level difference labels of the SwissGov-RSD annotators are marked underneath.
    Scores are the submitted predictions, read from the prediction files; nothing is recomputed by a model. Built __GENERATED__.</div>
  <div class="summary" id="summary"></div>
</header>
<div class="controls">
  <label>Language <select id="lang"></select></label>
  <label>Documents <select id="split"><option value="all">all</option><option value="dev">dev</option><option value="test">test</option></select></label>
  <label>Sort <select id="sort"><option value="id">by id</option><option value="rho_desc">best ρ first</option><option value="rho_asc">worst ρ first</option><option value="diff_desc">most gold differences</option></select></label>
  <span class="sep"></span>
  <label>Shading <select id="system"></select></label>
  <label>View <select id="mode">
    <option value="both">prediction + gold</option>
    <option value="pred">prediction only</option>
    <option value="gold">gold only</option>
    <option value="agree">agreement at threshold</option>
  </select></label>
  <label id="thrbox">threshold <input type="range" id="thr" min="1" max="100" value="50" style="width:110px"> <span id="thrv">0.50</span></label>
  <span class="sep"></span>
  <label><input type="checkbox" id="dimpunct" checked> dim punctuation (not scored)</label>
  <span class="hint" id="nav">← / → previous / next document</span>
</div>
<main>
  <aside id="list"></aside>
  <section class="doc" id="doc"></section>
</main>
<div id="tip"></div>
<script id="rsd-data" type="application/json">__DATA__</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('rsd-data').textContent);
const $ = id => document.getElementById(id);
const state = {lang: Object.keys(D.langs)[0], split:'all', sort:'id', system: D.main, mode:'both', thr:50, dim:true, doc:null};

function fmt(x, d){ return (x===null||x===undefined||Number.isNaN(x)) ? '—' : Number(x).toFixed(d===undefined?3:d); }
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ---- colours
function predStyle(s){           // s: 0..100
  if(s<=0) return '';
  if(s>=50){ const t=(s-50)/50; const L=88-30*t; return 'background:hsl(2,85%,'+L+'%)'+(t>0.6?';color:#fff':''); }
  const t=Math.sqrt(s/49); const L=96-32*t; return 'background:hsl(38,95%,'+L+'%)';
}
function goldBorder(g){          // g: -1,0..100
  if(g<=0) return '';
  const L=62-30*(g/100); return 'border-bottom-color:hsl(212,85%,'+L+'%)';
}
function goldBg(g){
  if(g<=0) return '';
  const L=94-32*(g/100); return 'background:hsl(212,85%,'+L+'%)';
}
function agreeStyle(s,g,thr){
  const p = s>=thr, y = g>0;
  if(p&&y) return 'background:#c7f0d2;border-bottom-color:#2da44e';
  if(p&&!y) return 'background:#e6dcff;border-bottom-color:#8250df';
  if(!p&&y) return 'background:#ffd8d6;border-bottom-color:#cf222e';
  return '';
}

// ---- summary table
function renderSummary(){
  const box=$('summary'); box.innerHTML='';
  const systems=D.systems;
  for(const lang of Object.keys(D.langs)){
    const S=D.summary[lang]; const tbl=document.createElement('table');
    let h='<caption>'+esc(D.langs[lang].name)+' (EN–'+lang.toUpperCase()+'): pooled Spearman</caption><tr><th></th>';
    for(const sp of ['dev','test','all']) if(S[sp]) h+='<th>'+sp+' (n='+S[sp].n+')</th>';
    h+='</tr>';
    for(const sys of systems){ h+='<tr><td>'+esc(sys)+'</td>';
      for(const sp of ['dev','test','all']) if(S[sp]) h+='<td>'+fmt(S[sp][sys])+'</td>';
      h+='</tr>'; }
    tbl.innerHTML=h; box.appendChild(tbl);
  }
}

// ---- document list
function docsOf(){
  let docs=D.langs[state.lang].docs.slice();
  if(state.split==='dev') docs=docs.filter(d=>d.split!=='test');
  if(state.split==='test') docs=docs.filter(d=>d.split==='test');
  const rho=d=>{const s=d.st[state.system]; return s&&s.rho!==null&&s.rho!==undefined?s.rho:-9;};
  if(state.sort==='rho_desc') docs.sort((a,b)=>rho(b)-rho(a));
  else if(state.sort==='rho_asc') docs.sort((a,b)=>rho(a)-rho(b));
  else if(state.sort==='diff_desc') docs.sort((a,b)=>(b.n_gold_diff/b.n_scored)-(a.n_gold_diff/a.n_scored));
  return docs;
}
function renderList(){
  const docs=docsOf(); const list=$('list'); list.innerHTML='';
  if(!docs.some(d=>d.id===state.doc)) state.doc=docs.length?docs[0].id:null;
  for(const d of docs){
    const row=document.createElement('div'); row.className='row'+(d.id===state.doc?' active':''); row.dataset.id=d.id;
    const s=d.st[state.system]||{};
    row.innerHTML='<span class="id">'+esc(d.id)+'</span><span class="tag '+(d.split==='test'?'test':'')+'">'+esc(d.split)+'</span>'
      +'<span class="rho" title="Spearman inside this document, '+esc(state.system)+'">ρ '+fmt(s.rho,2)+'</span>'
      +'<span class="ttl">'+esc(d.title)+' · '+d.n_gold_diff+'/'+d.n_scored+' gold diff</span>';
    row.onclick=()=>{state.doc=d.id; renderList(); renderDoc();};
    list.appendChild(row);
  }
  const act=list.querySelector('.row.active'); if(act) act.scrollIntoView({block:'nearest'});
}

// ---- one document
function wordsHtml(text, pred, gold, side){
  const words=text.split(' '); let out='';
  for(let i=0;i<words.length;i++){
    const s=pred?pred[i]:0, g=gold[i]; let style='';
    if(state.mode==='pred') style=predStyle(s);
    else if(state.mode==='gold') style=goldBg(g);
    else if(state.mode==='agree') style=(g===-1?'':agreeStyle(s,g,state.thr));
    else style=predStyle(s)+';'+goldBorder(g);
    const cls='w'+((state.dim&&g===-1)?' p':'');
    out+='<span class="'+cls+'" style="'+style+'" data-i="'+i+'" data-side="'+side+'">'+esc(words[i])+'</span> ';
  }
  return out;
}
function renderDoc(){
  const box=$('doc'); const d=D.langs[state.lang].docs.find(x=>x.id===state.doc);
  if(!d){ box.innerHTML='<p class="hint">No document.</p>'; return; }
  const sys=d.sys[state.system]; const st=d.st[state.system]||{};
  let h='<div class="dochead"><h2>'+esc(d.id)+'</h2><span class="tag">'+esc(d.split)+'</span><span style="color:var(--muted)">'+esc(d.title)+'</span>';
  if(d.page_en) h+='<a href="'+esc(d.page_en)+'" target="_blank" rel="noopener">English page ↗</a>';
  if(d.page_other) h+='<a href="'+esc(d.page_other)+'" target="_blank" rel="noopener">'+esc(D.langs[state.lang].name)+' page ↗</a>';
  h+='</div><div class="stats">';
  h+='<span>words: <b>'+(d.ta.split(' ').length+d.tb.split(' ').length)+'</b> (scored '+d.n_scored+')</span>';
  h+='<span>gold difference words: <b>'+d.n_gold_diff+'</b> ('+fmt(100*d.n_gold_diff/d.n_scored,1)+'%)</span>';
  h+='<span>'+esc(D.main)+' words ≥ 0.5 (verified absent blocks): <b>'+d.n_flagged+'</b></span>';
  for(const name of D.systems){ const s=d.st[name]; if(!s) continue;
    h+='<span>'+esc(name)+' ρ in this document: <b>'+fmt(s.rho)+'</b> (EN '+fmt(s.rho_a,2)+' · '+state.lang.toUpperCase()+' '+fmt(s.rho_b,2)+')</span>'; }
  h+='</div>';
  if(!sys) h+='<p class="hint">'+esc(state.system)+' has no predictions for this document; shading shows nothing.</p>';
  h+='<div class="cols"><div class="col"><h3>English</h3>'+wordsHtml(d.ta, sys?sys.a:null, d.ga, 'a')+'</div>';
  h+='<div class="col"><h3>'+esc(D.langs[state.lang].name)+'</h3>'+wordsHtml(d.tb, sys?sys.b:null, d.gb, 'b')+'</div></div>';
  h+=legendHtml();
  h+='<p class="hint">Page links are reconstructed from the dataset\'s page names and need an internet connection. Spearman inside one document is descriptive only; the official metric pools all words of a language. ρ is undefined (—) when gold or prediction is constant in the document.</p>';
  box.innerHTML=h;
  box.querySelectorAll('.w').forEach(el=>{ el.onmousemove=ev=>showTip(ev,d,el); el.onmouseleave=hideTip; });
}
function legendHtml(){
  let h='<div class="legend">';
  if(state.mode==='pred'||state.mode==='both'){
    h+='<span><span class="sw" style="background:hsl(38,95%,90%)"></span>low score</span><span><span class="sw" style="background:hsl(38,95%,66%)"></span>high score (&lt; 0.5, ranked by the probability signal)</span>';
    h+='<span><span class="sw" style="background:hsl(2,85%,75%)"></span>≥ 0.5: block judged absent and verified</span>';
  }
  if(state.mode==='both') h+='<span><span class="sw" style="border-bottom-color:hsl(212,85%,45%)"></span>underline: gold difference (darker = more annotators)</span>';
  if(state.mode==='gold') h+='<span><span class="sw" style="background:hsl(212,85%,78%)"></span>gold difference (darker = stronger label)</span>';
  if(state.mode==='agree'){
    h+='<span><span class="sw" style="background:#c7f0d2"></span>hit: predicted ≥ threshold and gold difference</span>';
    h+='<span><span class="sw" style="background:#e6dcff"></span>predicted ≥ threshold, no gold difference</span>';
    h+='<span><span class="sw" style="background:#ffd8d6"></span>gold difference, predicted below threshold</span>';
  }
  h+='<span><span class="sw" style="background:#fff;border:1px solid #ccc;color:#9aa3ad"></span>grey text: punctuation, not scored (gold −1)</span></div>';
  return h;
}
const tip=$('tip');
function showTip(ev,d,el){
  const i=+el.dataset.i, side=el.dataset.side; const g=(side==='a'?d.ga:d.gb)[i];
  let t=el.textContent+'\n';
  for(const name of D.systems){ const s=d.sys[name]; if(s) t+=name+': '+(s[side][i]/100).toFixed(2)+'\n'; }
  t+='gold: '+(g===-1?'not scored':(g/100).toFixed(1));
  tip.textContent=t; tip.style.display='block';
  const x=Math.min(ev.clientX+12, window.innerWidth-330), y=ev.clientY+14; tip.style.left=x+'px'; tip.style.top=y+'px';
}
function hideTip(){ tip.style.display='none'; }

// ---- controls
function fill(sel, items, value){ sel.innerHTML=''; for(const [v,l] of items){ const o=document.createElement('option'); o.value=v; o.textContent=l; sel.appendChild(o);} sel.value=value; }
fill($('lang'), Object.keys(D.langs).map(l=>[l, D.langs[l].name+' ('+l+')']), state.lang);
fill($('system'), D.systems.map(s=>[s,s]), state.system);
$('lang').onchange=e=>{state.lang=e.target.value; state.doc=null; renderList(); renderDoc();};
$('split').onchange=e=>{state.split=e.target.value; renderList(); renderDoc();};
$('sort').onchange=e=>{state.sort=e.target.value; renderList();};
$('system').onchange=e=>{state.system=e.target.value; renderList(); renderDoc();};
$('mode').onchange=e=>{state.mode=e.target.value; $('thrbox').style.visibility=state.mode==='agree'?'visible':'hidden'; renderDoc();};
$('thr').oninput=e=>{state.thr=+e.target.value; $('thrv').textContent=(state.thr/100).toFixed(2); if(state.mode==='agree') renderDoc();};
$('dimpunct').onchange=e=>{state.dim=e.target.checked; renderDoc();};
$('thrbox').style.visibility='hidden';
document.addEventListener('keydown',ev=>{
  if(ev.key!=='ArrowLeft'&&ev.key!=='ArrowRight') return;
  if(ev.target&&/INPUT|SELECT/.test(ev.target.tagName)) return;
  const docs=docsOf(); const k=docs.findIndex(d=>d.id===state.doc); if(k<0) return;
  const n=ev.key==='ArrowRight'?Math.min(k+1,docs.length-1):Math.max(k-1,0);
  state.doc=docs[n].id; renderList(); renderDoc(); ev.preventDefault();
});
renderSummary(); renderList(); renderDoc();
})();
</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return (_TEMPLATE.replace("__TITLE__", payload["title"].replace("&", "&amp;").replace("<", "&lt;"))
            .replace("__GENERATED__", payload["generated"])
            .replace("__DATA__", data))


def build_viewer(pred_dir: Path, system: str, out: Path, allow_test: bool, system_label: str = "Fuse",
                 langs: Sequence[str] = LANGS, title: str = "") -> dict:
    payload = build_payload(pred_dir, system, allow_test, system_label=system_label, langs=langs, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    return {"out": str(out), "bytes": out.stat().st_size,
            "docs": {lang: len(v["docs"]) for lang, v in payload["langs"].items()},
            "systems": payload["systems"], "summary": payload["summary"]}
