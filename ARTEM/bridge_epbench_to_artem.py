"""
bridge_epbench_to_artem.py

Convert an EPBench book (groundtruth + QA) into the on-disk format that the
ARTEM/STEM pipeline (`eventRetriever.py` + `STEM_evaluation.py`) expects:

    <out_root>/book<ID>/extracted_features_book<ID>.json   # the events (t,s,e,c)
    <out_root>/book<ID>/qa_book<ID>.json                    # questions + ground truth

EPBench already stores cue / cue_completed / retrieval_type / get / correct_answer
in exactly the shape ARTEM uses, so this is mostly a pass-through + array->list
conversion. No LLM needed: EPBench ships the gold (t,s,e,c) per chapter, which we
feed straight in (skipping ARTEM's LLM `dataExtraction` step).

Usage:
    python bridge_epbench_to_artem.py --universe Udefault_Sdefault_seed0 \
        --chapters 20 --out data_root_epbench --book_id 1
"""
import os
import re
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v) for v in list(x)]
    return [str(x)]


def find_book(data_root, universe, chapters):
    pat = os.path.join(data_root, universe, "books", f"*nbchapters_{chapters}_*")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"No EPBench book matches {pat}")
    return hits[0]


def convert(book_dir, out_root, book_id):
    gt = pd.read_parquet(os.path.join(book_dir, "df_book_groundtruth.parquet"))
    qa = pd.read_parquet(os.path.join(book_dir, "df_qa.parquet")).reset_index(drop=True)

    out_dir = os.path.join(out_root, f"book{book_id}")
    os.makedirs(out_dir, exist_ok=True)

    # --- events: one per chapter, gold (t, s, e, c) ---
    events = []
    for _, r in gt.iterrows():
        primary = str(r["entity"])
        post = _to_list(r.get("post_entities"))
        events.append({
            "chapter": int(r["chapter"]),
            "time": [str(r["date"])],
            "spaces": str(r["location"]),
            "entities": [primary] + post,     # transform_blocks -> primary + post_entities
            "content": str(r["content"]),
        })
    feat_path = os.path.join(out_dir, f"extracted_features_book{book_id}.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    # --- questions + ground truth: pass EPBench fields straight through ---
    questions = []
    for i, r in qa.iterrows():
        cac = r.get("correct_answer_chapters")
        cac = [int(x) for x in cac] if cac is not None and len(cac) else []
        questions.append({
            "q_idx": int(r.get("q_idx", i)),
            "question": str(r["question"]),
            "cue": str(r["cue"]),
            "cue_completed": str(r["cue_completed"]),
            "retrieval_type": str(r["retrieval_type"]),
            "get": str(r["get"]),
            "correct_answer": _to_list(r["correct_answer"]),
            "bins_items_correct_answer": str(r.get("bins_items_correct_answer",
                                                   len(_to_list(r["correct_answer"])))),
            # extra fields required by LLM_as_a_Judge.load_questions_from_json
            "n_chapters_correct_answer": int(r.get("n_chapters_correct_answer", len(cac))),
            "n_items_correct_answer": int(r.get("n_items_correct_answer",
                                                len(_to_list(r["correct_answer"])))),
            "correct_answer_chapters": cac,
            "correct_answer_detailed": _to_list(r.get("correct_answer_detailed")),
        })
    qa_path = os.path.join(out_dir, f"qa_book{book_id}.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"[bridge] {book_dir.split('/')[-1]}")
    print(f"[bridge] events: {len(events)} -> {feat_path}")
    print(f"[bridge] questions: {len(questions)} -> {qa_path}")
    return out_dir, len(events), len(questions)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="../episodic-memory-benchmark/epbench/data",
                    help="EPBench data root (contains the U*_S*_seed* universes)")
    ap.add_argument("--universe", default="Udefault_Sdefault_seed0")
    ap.add_argument("--chapters", type=int, default=20)
    ap.add_argument("--out", default="data_root_epbench")
    ap.add_argument("--book_id", type=int, default=1)
    args = ap.parse_args()

    book_dir = find_book(args.data_root, args.universe, args.chapters)
    convert(book_dir, args.out, args.book_id)
