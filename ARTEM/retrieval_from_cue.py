"""
retrieval_from_cue.py — từ cue (extracted/oracle) + method → chọn event giữ lại, và
(tùy chọn) ghi ra match_based_retrieval_results JSON đúng format cho LLM-as-a-Judge.

Lõi `retrieve()` dùng chung cho cả bước chọn θ (set-F1) lẫn bước LLM-judge.

method:
  hard     : giữ event nếu m_k ≥ rho_k cho MỌI trục active (baseline ARTEM)          -- AND
  gated    : bỏ trục conf<theta rồi hard-AND trên trục còn lại (CIRCUIT novelty)      -- AND
  adaptive : score = Σ conf_k·m_k / Σ conf_k ; giữ nếu score ≥ θ                      -- soft-AND (graded)
  literal  : Score = Σ_k w_base_k·conf_k·1(m_k≥rho_k) ; giữ nếu score > 0             -- Eq.3.3 GỐC
             (TỔNG có trọng số của indicator NHỊ PHÂN, không phải AND -- 1 trục khớp
             mạnh cũng đủ giữ event, giống hệt công thức draft Section 3.3)
"""
import os, re, json, argparse
import numpy as np
from stem_with_novelty import StemEngine, RHO, TYPE_FIELD, _date

AXES = ["time", "spaces", "entities", "content"]


def retrieve(eng, vals, conf, method, theta=0.95, rho=None):
    """vals: {axis:value}, conf: {axis:confidence}. -> (keep_idx list, score array|None)."""
    rho = rho or RHO
    axes = [a for a in AXES if vals.get(a)]
    N = len(eng.events)
    if not axes:
        return [], None
    M = {a: eng.match(a, vals[a]) for a in axes}
    if method == "hard":
        ok = np.ones(N, dtype=bool)
        for a in axes:
            ok &= (M[a] >= rho[a])
        return list(np.where(ok)[0]), None
    if method == "gated":
        # GATED-AND: bỏ trục conf < theta (≈ trục trích sai) rồi hard-AND trên trục còn lại
        use = [a for a in axes if conf.get(a, 1.0) >= theta]
        if not use:                      # đừng abstain oan nếu drop hết
            use = axes
        ok = np.ones(N, dtype=bool)
        for a in use:
            ok &= (M[a] >= rho[a])
        return list(np.where(ok)[0]), None
    if method == "literal":
        # Eq.3.3 GOC: Score = Sum_k w_base_k * Conf_k(Q) * 1(m_k >= rho_k), w_base_k=1.0 (uniform)
        score = np.zeros(N)
        for a in axes:
            ind = (M[a] >= rho[a]).astype(float)
            score += 1.0 * conf.get(a, 1.0) * ind
        return list(np.where(score > 0)[0]), score
    if method == "circuit":
        # CIRCUIT additive graded (như LoCoMo): Score = Sum_k Conf_k(Q) * m_k(event), m_k GRADED
        # (KHÔNG threshold cứng). Giữ TOP-K theo rank (giống LoCoMo top-mturn) -> KHÔNG BAO GIỜ
        # rỗng, bounded, không bị 0.98-vigilance giết. theta dùng làm K (số event giữ, mặc định 8).
        K = int(theta) if theta and theta >= 1 else 8
        score = np.zeros(N)
        for a in axes:
            score += conf.get(a, 1.0) * M[a]
        if score.max() <= 0:
            return [], score
        keep = list(np.argsort(-score)[:K])
        return keep, score

    wsum = sum(conf.get(a, 1.0) for a in axes) or 1.0
    score = sum(conf.get(a, 1.0) * M[a] for a in axes) / wsum
    return list(np.where(score >= theta)[0]), score


def cue_from_extracted(cue_obj):
    vals = {a: cue_obj[a]["value"] for a in AXES if cue_obj.get(a)}
    conf = {a: cue_obj[a]["confidence"] for a in AXES if cue_obj.get(a)}
    return vals, conf


def event_dict(eng, i, score):
    ev = eng.events[i]
    ent = ev["entities"]
    primary = ent[0] if isinstance(ent, list) and ent else (ent if isinstance(ent, str) else "")
    post = ent[1:] if isinstance(ent, list) else []
    s = float(score) if score is not None else 1.0
    return {
        "time": ev["time"] if isinstance(ev["time"], list) else [str(ev["time"])],
        "spaces": ev["spaces"], "entities": primary, "post_entities": post,
        "content": ev["content"], "match_score": s, "weighted_match_score": s,
        "avg_match_score": s, "individual_match_scores": [], "normalized_time": float(eng.ev_time[i]),
        "vigilance_passed": True,
    }


def build_results(eng, qa, cues_by_idx, method, theta):
    """qa: list câu (đúng thứ tự). cues_by_idx: pos(str)->cue_obj. -> dict retrieval_results."""
    out = []
    for i, q in enumerate(qa):
        cue = cues_by_idx.get(str(i), {})
        vals, conf = cue_from_extracted(cue)
        keep, score = retrieve(eng, vals, conf, method, theta)
        evs = [event_dict(eng, j, None if score is None else score[j]) for j in keep]
        evs.sort(key=lambda e: e["normalized_time"])           # time-sorted
        out.append({
            "query_id": q.get("q_idx", i), "retrieval_type": q.get("retrieval_type"),
            "results": {
                "all_vigilant_events_time_sorted": evs,
                "top_k_events_time_sorted": evs[-1:] if evs else [],   # latest = mới nhất
            },
        })
    return {"retrieval_results": out, "config": {"method": method, "theta": theta}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_root_sample")
    ap.add_argument("--book_id", type=int, default=1)
    ap.add_argument("--cues", required=True, help="extracted_cues_calibrated.json (keyed theo pos của --base)")
    ap.add_argument("--method", choices=["hard", "adaptive", "gated", "literal", "circuit"], required=True)
    ap.add_argument("--theta", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bk = f"{args.base}/book{args.book_id}"
    qa = json.load(open(f"{bk}/qa_book{args.book_id}.json"))
    events = json.load(open(f"{bk}/extracted_features_book{args.book_id}.json"))
    cues = json.load(open(args.cues))
    eng = StemEngine(events)
    res = build_results(eng, qa, cues, args.method, args.theta)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"Ghi {len(res['retrieval_results'])} kết quả ({args.method}, θ={args.theta}) -> {args.out}")
