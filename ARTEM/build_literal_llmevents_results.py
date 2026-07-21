"""
build_literal_llmevents_results.py — method "literal" (đúng Eq.3.3 GỐC: tổng có trọng
số của indicator nhị phân, KHÔNG phải AND) trên events LLM-trích (B4, không gold),
full 668 câu. Dùng để đối chứng với Gated-AND, giống hệt cách đã làm cho LoCoMo.
"""
import os, json
from stem_with_novelty import StemEngine
from retrieval_from_cue import build_results

BASE668 = "data_root_ep200/book1"          # qa + cues (không đổi)
LLMEV = "data_root_ep200_llmevents/book1/extracted_features_book1.json"

qa668 = json.load(open(f"{BASE668}/qa_book1.json"))
events = json.load(open(LLMEV))
cues668 = json.load(open(f"{BASE668}/extracted_cues_calibrated.json"))
eng = StemEngine(events)

print(f"LLM-extracted events: {len(events)}, QA: {len(qa668)}")

out_dir = "data_root_extracted_literal_llmev668/book1"
os.makedirs(out_dir, exist_ok=True)
json.dump(qa668, open(f"{out_dir}/qa_book1.json", "w"), ensure_ascii=False, indent=1)
res = build_results(eng, qa668, cues668, "literal", 0.0)
json.dump(res, open(f"{out_dir}/match_based_retrieval_results_book1.json", "w"),
         ensure_ascii=False, indent=1)
print(f"[literal] -> {out_dir}/ ({len(res['retrieval_results'])} câu)")

# thống kê nhanh: literal có xu hướng giữ QUÁ NHIỀU event (union có trọng số, không AND)?
sizes = [len(r["results"]["all_vigilant_events_time_sorted"]) for r in res["retrieval_results"]]
import numpy as np
print(f"avg #events giữ lại/câu: {np.mean(sizes):.1f} (median {np.median(sizes):.0f}, max {max(sizes)})")
