"""
STEM + novelty  (ARTEM engine only — no external systems)

Takes the ARTEM/STEM fusion-ART engine and replaces its HARD AND-vigilance retrieval
with the adaptive confidence-weighted scoring from REVIEW_CIRCUIT §4.1, reusing the
engine's own GRADED per-channel match m_k (this is what STEM already computes and what
a binary-indicator scorer lacks).

Per query, per active axis k:
    m_k(event)  = fusion-ART match in [0,1]   (graded evidence-match)
    conf_k      = extractor confidence in [0,1] (low when the cue was mis-extracted)

  BASELINE (hard / STEM):  keep event iff  m_k >= rho_k  for ALL active k.
  NOVELTY  (adaptive):     score(event) = sum_k conf_k * m_k / sum_k conf_k ;
                           keep event iff score >= theta  (soft-AND + abstain).

We corrupt cue axes with prob p (and mark them low-confidence) to simulate imperfect
query->cue extraction, then compare the two on the SAME EPBench book with the SAME
field set-F1 metric used by STEM_evaluation.

Run:  python stem_with_novelty.py --src data_root_ep200/book1
"""
import os, re, json, copy, random, argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

from fusionART import complement_code

CONF_CLEAN, CONF_NOISY = 0.9, 0.3
RHO = {"time": 1.0, "spaces": 0.99, "entities": 0.99, "content": 0.98}
SLOT_TO_FIELD = {
    "t": "time", "time": "time", "times": "time",
    "s": "spaces", "l": "spaces", "loc": "spaces", "location": "spaces", "space": "spaces", "spaces": "spaces",
    "e": "entities", "ent": "entities", "entity": "entities", "entities": "entities",
    "c": "content", "con": "content", "content": "content",
}
TYPE_FIELD = {"times": "time", "time": "time", "spaces": "spaces", "entities": "entities",
              "content": "content", "event contents": "content"}
FIELDS = ["spaces", "entities", "content"]   # embedding channels
EMB = 384


def pv_minmax(emb):
    """per-vector min-max -> [0,1] (ARTEM eq.3)."""
    lo, hi = emb.min(axis=-1, keepdims=True), emb.max(axis=-1, keepdims=True)
    rng = np.where(hi != lo, hi - lo, 1.0)
    out = (emb - lo) / rng
    return np.where(hi == lo, 0.5, out)


def _date(s):
    s = s.strip()
    # bỏ tiền tố giờ kiểu "9:47 PM, " nếu có (LLM đôi khi trích cả giờ)
    s2 = re.sub(r"^\d{1,2}:\d{2}\s*(AM|PM)?\s*,?\s*", "", s, flags=re.I).strip()
    for cand in (s2, s):
        for f in ("%B %d, %Y", "%B %d %Y"):
            try: return datetime.strptime(cand, f)
            except Exception: pass
    return datetime.min


def parse_cue(q):
    """-> dict field -> clean value, for the specified (non-*) axes."""
    slots = [s.strip().lower().strip("{}") for s in q["cue"].strip("()").split(",")]
    tokens = re.findall(r"{([^}]+)}", q["cue_completed"])
    out, ti = {}, 0
    for s in slots:
        if s == "*":
            continue
        field = SLOT_TO_FIELD.get(s)
        val = tokens[ti] if ti < len(tokens) else ""
        ti += 1
        if field:
            out[field] = val
    return out


class StemEngine:
    """Encodes events; exposes graded per-channel match m_k for any query."""
    def __init__(self, events):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.events = events
        # --- time channel (global min-max) ---
        ts = []
        for e in events:
            t = e["time"][0] if isinstance(e["time"], list) else e["time"]
            d = _date(str(t))
            try:
                ts.append(d.timestamp())
            except (ValueError, OSError, OverflowError):
                ts.append(0.0)                 # date không parse được -> sentinel, không crash
        ts = np.array(ts)
        self.tmin, self.tmax = ts.min(), ts.max()
        self.ev_time = (ts - self.tmin) / (self.tmax - self.tmin if self.tmax != self.tmin else 1)
        self.ev_date = [_date(str(e["time"][0] if isinstance(e["time"], list) else e["time"])) for e in events]
        # metric cho m_k embedding: "fuzzy" (STEM gốc) hoặc "cosine" (REVIEW §3.1)
        self.metric = "fuzzy"
        # --- embedding channels ---
        self.W = {}        # fuzzy: complement-coded per-vector-minmax (N, 768)
        self.Eraw = {}     # cosine: L2-normalized MiniLM embedding (N, 384)
        for f in FIELDS:
            texts = []
            for e in events:
                v = e.get(f, "none")
                if isinstance(v, list):
                    v = v[0] if v else "none"
                texts.append(str(v))
            raw = np.asarray(self.model.encode(texts, normalize_embeddings=False))
            emb = pv_minmax(raw)
            self.W[f] = np.concatenate([emb, 1.0 - emb], axis=1)            # (N, 768)
            self.Eraw[f] = np.asarray(self.model.encode(texts, normalize_embeddings=True))  # (N,384) unit
        self._cache = {}    # fuzzy query cache (cc)
        self._ccache = {}   # cosine query cache (unit)

    def _q_emb(self, text):
        if text not in self._cache:
            e = pv_minmax(np.asarray(self.model.encode([text], normalize_embeddings=False))[0])
            self._cache[text] = complement_code(e)                 # (768,)
        return self._cache[text]

    def _q_emb_cos(self, text):
        if text not in self._ccache:
            self._ccache[text] = np.asarray(self.model.encode([text], normalize_embeddings=True)[0])
        return self._ccache[text]

    def match_text(self, field, value):
        """graded m_k cho mọi event trên 1 kênh embedding (vectorized)."""
        if self.metric == "cosine":
            q = self._q_emb_cos(str(value))                        # (384,) unit
            cos = self.Eraw[field] @ q                             # (N,) ∈ [-1,1]
            return np.clip(cos, 0.0, 1.0)                          # sàn thấp, tách bạch
        q = self._q_emb(str(value))                                # (768,)
        return np.minimum(q[None, :], self.W[field]).sum(axis=1) / EMB   # fuzzy (N,)

    def match_time(self, value):
        d = _date(str(value))
        if d == datetime.min:                                      # cue thời gian không parse được
            return np.zeros(len(self.ev_time))                     # → không khớp gì
        try:
            ts = d.timestamp()
        except (ValueError, OSError, OverflowError):
            return np.zeros(len(self.ev_time))
        tq = (ts - self.tmin) / (self.tmax - self.tmin if self.tmax != self.tmin else 1)
        return 1.0 - np.abs(tq - self.ev_time)                     # (N,)

    def match(self, field, value):
        return self.match_time(value) if field == "time" else self.match_text(field, value)


def build_vocab(events):
    vocab = {"time": set(), "spaces": set(), "entities": set(), "content": set()}
    for e in events:
        t = e["time"][0] if isinstance(e["time"], list) else e["time"]
        vocab["time"].add(str(t)); vocab["spaces"].add(str(e["spaces"]))
        ent = e["entities"]; vocab["entities"].add(str(ent[0] if isinstance(ent, list) else ent))
        vocab["content"].add(str(e["content"]))
    return {k: sorted(v) for k, v in vocab.items()}


def perturb_cue(cue_vals, vocab, p, rng):
    """corrupt each specified axis w.p. p -> (values, confidences)."""
    vals, conf = {}, {}
    for f, v in cue_vals.items():
        if rng.random() < p:
            choices = [x for x in vocab[f] if x != v]
            vals[f] = rng.choice(choices) if choices else v
            conf[f] = CONF_NOISY
        else:
            vals[f] = v
            conf[f] = CONF_CLEAN
    return vals, conf


def field_value(eng, idx, field):
    e = eng.events[idx]
    v = e.get(field)
    if isinstance(v, list):
        v = v[0] if v else None
    return str(v).strip() if v not in (None, "", "none") else None


def evaluate(eng, qa, vocab, method, p, theta=0.9, seed=0):
    rng = random.Random(seed)
    rows = []
    for q in qa:
        cue_vals = parse_cue(q)
        vals, conf = perturb_cue(cue_vals, vocab, p, rng)
        axes = list(vals.keys())
        N = len(eng.events)
        if not axes:
            keep = []
        else:
            M = {f: eng.match(f, vals[f]) for f in axes}           # graded m_k per axis
            if method == "hard":
                ok = np.ones(N, dtype=bool)
                for f in axes:
                    ok &= (M[f] >= RHO[f])
                keep = list(np.where(ok)[0])
            else:  # adaptive confidence-weighted
                wsum = sum(conf[f] for f in axes)
                score = sum(conf[f] * M[f] for f in axes) / wsum   # weighted-avg match in [0,1]
                keep = list(np.where(score >= theta)[0])
        # get-mode
        if q.get("get") == "latest" and keep:
            keep = [max(keep, key=lambda i: eng.ev_date[i])]
        # extract answer field
        tf = TYPE_FIELD.get(q["retrieval_type"].lower())
        pred = []
        if tf:
            for i in keep:
                fv = field_value(eng, i, tf)
                if fv:
                    pred.append(fv)
        pred = list(dict.fromkeys(pred))
        gold = [str(x) for x in q.get("correct_answer", [])]
        ps, gs = set(pred), set(gold)
        if not ps and not gs: pr, rc, f1 = 1.0, 1.0, 1.0
        elif not ps:          pr, rc, f1 = 0.0, 0.0, 0.0
        elif not gs:          pr, rc, f1 = 0.0, 1.0, 0.0
        else:
            tp = len(ps & gs); pr = tp/len(ps); rc = tp/len(gs)
            f1 = 2*pr*rc/(pr+rc) if (pr+rc) else 0.0
        rows.append((q["retrieval_type"], pr, rc, f1))
    import numpy as _np
    f1s = _np.array([r[3] for r in rows]); prs = _np.array([r[1] for r in rows]); rcs = _np.array([r[2] for r in rows])
    return f1s.mean(), prs.mean(), rcs.mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_root_ep200/book1")
    ap.add_argument("--thetas", default="0.90,0.95,0.97")
    args = ap.parse_args()

    events = json.load(open(os.path.join(args.src, "extracted_features_book1.json")))
    qa = json.load(open(os.path.join(args.src, "qa_book1.json")))
    vocab = build_vocab(events)
    print(f"#events={len(events)}  #questions={len(qa)}")
    print("Encoding events into STEM engine ...")
    eng = StemEngine(events)

    levels = [0.0, 0.3, 0.5]
    thetas = [float(x) for x in args.thetas.split(",")]

    print("\n=== F1 (field set-F1) — STEM baseline vs STEM+novelty ===")
    print("%-34s | %7s %7s %7s" % ("method", "0%", "30%", "50%"))
    print("-" * 66)
    base = [evaluate(eng, qa, vocab, "hard", p)[0] for p in levels]
    print("%-34s | %7.3f %7.3f %7.3f" % ("hard vigilance (STEM baseline)", *base))
    for th in thetas:
        vals = [evaluate(eng, qa, vocab, "adaptive", p, theta=th)[0] for p in levels]
        print("%-34s | %7.3f %7.3f %7.3f" % (f"adaptive conf-weighted theta={th}", *vals))

    print("\n=== P / R / F1 detail at 30% noise ===")
    bp = evaluate(eng, qa, vocab, "hard", 0.3)
    print("  hard            : F1 %.3f | P %.3f | R %.3f" % (bp[2], bp[0]*0+bp[0], bp[1]) if False else
          "  hard            : F1 %.3f | P %.3f | R %.3f" % (bp[0], bp[1], bp[2]))
    for th in thetas:
        ap_ = evaluate(eng, qa, vocab, "adaptive", 0.3, theta=th)
        print("  adaptive th=%.2f : F1 %.3f | P %.3f | R %.3f" % (th, ap_[0], ap_[1], ap_[2]))
