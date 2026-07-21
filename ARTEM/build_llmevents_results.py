"""
build_llmevents_results.py — B4 tier: dùng events LLM-trích từ raw text (KHÔNG gold)
thay cho data_root_ep200 gold events. QA (668 câu) + cue LLM-trích/calibrated giữ
nguyên (không phụ thuộc gold events). Build match_based_retrieval_results cho
hard-AND + gated-AND(theta=0.75) trên full 668 câu, dùng engine dựng từ LLM events.
"""
import os, json
from stem_with_novelty import StemEngine
from retrieval_from_cue import build_results

BASE668 = "data_root_ep200/book1"          # qa + cues (không đổi)
LLMEV = "data_root_ep200_llmevents/book1/extracted_features_book1.json"
GATE_THETA = 0.75

qa668 = json.load(open(f"{BASE668}/qa_book1.json"))
events = json.load(open(LLMEV))
cues668 = json.load(open(f"{BASE668}/extracted_cues_calibrated.json"))
eng = StemEngine(events)

print(f"LLM-extracted events: {len(events)}, QA: {len(qa668)}")

for method, theta, tag in [("hard", 0.0, "extracted_hard_llmev668"),
                           ("gated", GATE_THETA, "extracted_gated_llmev668")]:
    out_dir = f"data_root_{tag}/book1"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(qa668, open(f"{out_dir}/qa_book1.json", "w"), ensure_ascii=False, indent=1)
    res = build_results(eng, qa668, cues668, method, theta)
    json.dump(res, open(f"{out_dir}/match_based_retrieval_results_book1.json", "w"),
             ensure_ascii=False, indent=1)
    print(f"[{tag}] method={method} theta={theta} -> {out_dir}/ ({len(res['retrieval_results'])} câu)")
