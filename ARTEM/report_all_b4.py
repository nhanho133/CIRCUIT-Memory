"""
report_all_b4.py — bảng EPBench cuối cùng: mọi method có kết quả (B4, LLM-extracted
events, DeepSeek, full 668). In F1 lenient/harsh tổng + theo retrieval_type + get-mode
+ answer-size bin, và số zero-retrieval (đo brittleness).
"""
import os, json
import pandas as pd

QA_PATH = "data_root_ep200/book1/qa_book1.json"
METHODS = [
    ("hard",    "Hard-AND (ARTEM baseline)"),
    ("literal", "Literal Eq.3.3 (indicator sum)"),
    ("gated",   "Gated-AND (drop low-conf axis)"),
    ("circuit", "CIRCUIT additive graded (top-K)"),
]


def bin_of(n):
    if n == 0: return "bin0"
    if n == 1: return "bin1"
    if n == 2: return "bin2"
    if 3 <= n <= 5: return "bin3-5"
    return "bin6+"


def load(tag):
    path = f"data_root_extracted_{tag}_llmev668/book1/artem_llm_results.json"
    if not os.path.exists(path):
        return None
    df = pd.DataFrame(json.load(open(path)))
    qa = {q["q_idx"]: q["get"] for q in json.load(open(QA_PATH))}
    df["get_mode"] = df["q_idx"].map(qa)
    df["ans_bin"] = df["n_items_correct_answer"].map(bin_of)
    return df


print("="*90)
print("EPBench (B4: LLM-extracted events, DeepSeek-R1-Distill-Qwen-14B, full 668) — tất cả method")
print("="*90)
print(f"{'method':<34} {'F1_len':>7} {'F1_harsh':>9} {'zero-ret':>9}")
print("-"*90)
loaded = {}
for tag, label in METHODS:
    df = load(tag)
    if df is None:
        print(f"{label:<34} {'(chưa chạy)':>7}")
        continue
    loaded[tag] = df
    zero = (df["retrieved_events_count"] == 0).sum()
    print(f"{label:<34} {df['f1_score_lenient'].mean():>7.3f} {df['f1_score_harsh'].mean():>9.3f} "
          f"{zero:>4d}/{len(df)} ({zero/len(df):.0%})")

for tag, label in METHODS:
    if tag not in loaded:
        continue
    df = loaded[tag]
    print(f"\n{'-'*90}\n{label}\n{'-'*90}")
    print("  by retrieval_type:")
    for t, g in df.groupby("retrieval_type"):
        print(f"    {t:<22} n={len(g):3d}  len={g['f1_score_lenient'].mean():.3f}  harsh={g['f1_score_harsh'].mean():.3f}")
    print("  by get-mode (all~SimpleRecall / chronological ~ Table-4):")
    for t, g in df.groupby("get_mode"):
        print(f"    {t:<15} n={len(g):3d}  len={g['f1_score_lenient'].mean():.3f}  harsh={g['f1_score_harsh'].mean():.3f}")
    print("  by answer-size bin:")
    for t in ["bin0", "bin1", "bin2", "bin3-5", "bin6+"]:
        g = df[df["ans_bin"] == t]
        if len(g):
            print(f"    {t:<15} n={len(g):3d}  len={g['f1_score_lenient'].mean():.3f}  harsh={g['f1_score_harsh'].mean():.3f}")

print("\n(Tham khảo, KHÔNG so trực tiếp - khác điều kiện đo: paper ARTEM gốc STEM=0.707,")
print(" trên book 'Synaptic Echoes' + DeepSeek của họ; ta dùng book Uscifi + judge tự host.)")
