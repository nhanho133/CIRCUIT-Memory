"""
gated_novelty_demo.py — CIRCUIT novelty (confidence-GATED intersection) end-to-end, TỰ CHỨA.

Không phụ thuộc EPBench/Modal/sentence-transformers — chỉ dùng string-match làm proxy cho
match_k(EC,Q) để chạy được ngay, không cần data/GPU nào. Mục đích: cho người khác thấy
CHÍNH XÁC cơ chế novelty trong vài chục dòng, không lẫn vào hạ tầng pipeline thật.

Ý tưởng (đúng logic thật trong ARTEM/retrieval_from_cue.py:35-43, mode "gated"):
  1. Hard-AND (baseline ARTEM/STEM): 1 trục trích SAI đủ để loại bỏ event ĐÚNG hoàn toàn.
  2. Gated-AND (CIRCUIT novelty): DÙNG confidence per-axis để loại bỏ trục khả nghi (conf < theta)
     TRƯỚC KHI hard-AND — event đúng vẫn được giữ vì trục sai đã bị loại khỏi phép AND.
  3. Nếu loại hết mọi trục (confidence tất cả đều thấp) -> fallback dùng lại toàn bộ trục
     (không abstain oan).

Chạy: python gated_novelty_demo.py
"""
from difflib import SequenceMatcher

AXES = ["time", "space", "entity", "content"]
RHO = {"time": 1.0, "space": 0.99, "entity": 0.99, "content": 0.6}   # ngưỡng match tối thiểu/trục

# ---------------------------------------------------------------- Tier-2: ExplicitCells (toy)
EVENTS = [
    {"id": "EC1", "time": "2025-03-02", "space": "Berlin",  "entity": "Dr. Vega",   "content": "cryo-pod integrity breach"},
    {"id": "EC2", "time": "2025-03-02", "space": "Berlin",  "entity": "Dr. Vega",   "content": "reactor coolant leak"},
    {"id": "EC3", "time": "2025-05-11", "space": "Odessa",  "entity": "Kai Ramos",  "content": "cryo-pod integrity breach"},
]


def match(field, cell_val, query_val):
    """Proxy đơn giản cho m_k(EC,Q): exact=1.0 cho time/space/entity, fuzzy string cho content.
    Bản thật dùng cosine similarity qua sentence-transformers (xem stem_with_novelty.py:147)."""
    if field == "content":
        return SequenceMatcher(None, cell_val.lower(), query_val.lower()).ratio()
    return 1.0 if cell_val.lower() == query_val.lower() else 0.0


def hard_and(query_vals):
    """Baseline ARTEM/STEM: giữ event nếu m_k >= rho_k cho MỌI trục có trong query."""
    kept = []
    for ec in EVENTS:
        ok = all(match(a, ec[a], v) >= RHO[a] for a, v in query_vals.items())
        if ok:
            kept.append(ec["id"])
    return kept


def gated_and(query_vals, conf, theta=0.9):
    """CIRCUIT novelty: bỏ trục có conf < theta rồi hard-AND trên trục còn lại."""
    use_axes = {a: v for a, v in query_vals.items() if conf.get(a, 1.0) >= theta}
    if not use_axes:                      # đừng abstain oan nếu drop hết mọi trục
        use_axes = query_vals
    kept = []
    for ec in EVENTS:
        ok = all(match(a, ec[a], v) >= RHO[a] for a, v in use_axes.items())
        if ok:
            kept.append(ec["id"])
    return kept, list(use_axes.keys())


if __name__ == "__main__":
    print("Events (Tier-2 ExplicitCells):")
    for ec in EVENTS:
        print(f"  {ec}")

    # Câu hỏi thật: "Cryo-pod integrity breach xảy ra ở đâu?" -> đáp án đúng = EC1 (Berlin).
    # LLM cue-extractor trích entity="Dr. Vega" ĐÚNG (conf cao) nhưng lỡ trích nhầm
    # space="Odessa" (conf THẤP — chính LLM cũng không chắc, vì câu hỏi không nêu rõ địa điểm).
    query_vals = {"time": "2025-03-02", "space": "Odessa", "entity": "Dr. Vega", "content": "cryo-pod integrity breach"}
    conf       = {"time": 0.97,          "space": 0.55,     "entity": 0.96,        "content": 0.93}

    print(f"\nQuery cue (1 trục 'space' bị trích SAI, conf thấp 0.55): {query_vals}")

    print("\n[Hard-AND, baseline ARTEM/STEM]")
    kept_hard = hard_and(query_vals)
    print(f"  giữ lại: {kept_hard or '(RỖNG — event đúng EC1 bị loại vì trục space sai!)'}")

    print("\n[Gated-AND, CIRCUIT novelty, theta=0.9]")
    kept_gated, used_axes = gated_and(query_vals, conf, theta=0.9)
    print(f"  trục dùng sau khi gate (bỏ 'space' vì conf 0.55 < 0.9): {used_axes}")
    print(f"  giữ lại: {kept_gated}")

    print("\n=> Novelty: cùng 1 trục trích sai, Hard-AND loại oan event đúng (EC1); "
          "Gated-AND dùng đúng confidence per-axis để bỏ trục khả nghi trước khi AND, "
          "phục hồi lại event đúng mà KHÔNG cần biết trước trục nào sai — chỉ cần "
          "confidence tự báo của cue-extractor.")
