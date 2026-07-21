"""
build_circuit_llmevents_results.py — method "circuit" (additive graded, relative-gate,
KHÔNG hard-threshold) trên events LLM-trích (B4), full 668. Đây là cơ chế CIRCUIT thống
nhất với LoCoMo (additive), kỳ vọng phá bỏ 59.6% zero-retrieval của gated/hard.
"""
import os, json
import numpy as np
from stem_with_novelty import StemEngine
from retrieval_from_cue import build_results

BASE668 = "data_root_ep200/book1"
LLMEV = "data_root_ep200_llmevents/book1/extracted_features_book1.json"
GATE = 8   # top-K: giữ 8 event điểm cao nhất theo additive graded score

qa668 = json.load(open(f"{BASE668}/qa_book1.json"))
events = json.load(open(LLMEV))
cues668 = json.load(open(f"{BASE668}/extracted_cues_calibrated.json"))
eng = StemEngine(events)
print(f"LLM-extracted events: {len(events)}, QA: {len(qa668)}")

out_dir = "data_root_extracted_circuit_llmev668/book1"
os.makedirs(out_dir, exist_ok=True)
json.dump(qa668, open(f"{out_dir}/qa_book1.json", "w"), ensure_ascii=False, indent=1)
res = build_results(eng, qa668, cues668, "circuit", GATE)
json.dump(res, open(f"{out_dir}/match_based_retrieval_results_book1.json", "w"),
         ensure_ascii=False, indent=1)

sizes = [len(r["results"]["all_vigilant_events_time_sorted"]) for r in res["retrieval_results"]]
zero = sum(1 for s in sizes if s == 0)
print(f"[circuit gate={GATE}] -> {out_dir}/ ({len(res['retrieval_results'])} câu)")
print(f"  avg #events/câu: {np.mean(sizes):.1f} (median {int(np.median(sizes))}, max {max(sizes)}) | zero-retrieval={zero}/{len(sizes)}={zero/len(sizes):.1%}")
