"""
cue_extractor.py — LLM trích cue (t,s,e,c) + confidence mỗi trục TỪ CÂU HỎI (Cách 4).

Khác với oracle cue (EPBench cho sẵn): ở đây LLM tự đọc câu hỏi văn xuôi và trích các
ràng buộc tìm kiếm + tự báo độ tự tin. LLM sẽ trích sai/lệch một số trục một cách tự
nhiên → đây là nguồn "trục sai" để novelty (adaptive conf-weighting) có đất diễn.

Giữ retrieval_type + get từ EPBench (spec task). CHỈ trích GIÁ TRỊ cue + confidence.

Output cache: <base>/book<k>/extracted_cues.json
  { "<q_idx_pos>": {"time": {value,confidence}|null, "spaces":..., "entities":..., "content":...} }
Idempotent: câu nào đã có trong cache thì bỏ qua.

Usage:
  export MODAL_LLM_URL=...
  python cue_extractor.py --base data_root_sample --book_id 1 [--limit 5]
"""
import os, re, json, argparse
from run_artem_llm import ModalLLMWrapper, MockLLM

AXES = ["time", "spaces", "entities", "content"]

# trục bị HỎI (answer target) theo retrieval_type → KHÔNG phải ràng buộc cue, phải bỏ
ASKED_AXIS = {"times": "time", "time": "time", "spaces": "spaces",
              "entities": "entities", "event contents": "content", "content": "content"}

PROMPT = """You extract SEARCH CONSTRAINTS from an episodic-memory question.

A question asks to find events filtered by some of these axes:
- time     : a date the events happened (e.g. "April 09, 2226")
- spaces   : a location (e.g. "Mars Valles Industrial Hub")
- entities : a person/protagonist name (e.g. "Aurora Chavez")
- content  : an event type/topic (e.g. "magnetic confinement rupture")

Extract ONLY the GIVEN constraints in the QUESTION, NOT the thing being asked for.
Example: "list all DATES of events about X" → the date is the ANSWER, set time=null;
only X (content) is the constraint. For each axis give the value and your confidence
0.0-1.0 that you extracted it correctly. If an axis is not a given constraint, output null.

Return ONLY a JSON object, no other text:
{{"time": {{"value":"...","confidence":0.0}} or null,
 "spaces": {{"value":"...","confidence":0.0}} or null,
 "entities": {{"value":"...","confidence":0.0}} or null,
 "content": {{"value":"...","confidence":0.0}} or null}}

QUESTION: "{question}"
"""


def parse_json(text):
    """Strip <think> and pull the first {...} object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return None
    return None


def norm_axis(v):
    """-> {'value':str,'confidence':float} or None"""
    if not isinstance(v, dict):
        return None
    val = v.get("value")
    if val in (None, "", "null"):
        return None
    try:
        c = float(v.get("confidence", 0.5))
    except Exception:
        c = 0.5
    return {"value": str(val).strip(), "confidence": max(0.0, min(1.0, c))}


def extract_cue(llm, question, retrieval_type=""):
    raw = llm.generate(user_prompt=PROMPT.format(question=question),
                       system_prompt="You are a precise information extractor. Output JSON only.",
                       max_new_tokens=2048)
    obj = parse_json(raw) or {}
    cue = {a: norm_axis(obj.get(a)) for a in AXES}
    # bỏ trục bị HỎI (answer target) — không phải ràng buộc
    drop = ASKED_AXIS.get(str(retrieval_type).lower())
    if drop:
        cue[drop] = None
    return cue, raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_root_sample")
    ap.add_argument("--book_id", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", choices=["modal", "mock"], default="modal")
    ap.add_argument("--show", action="store_true", help="in cue + oracle để đối chiếu")
    args = ap.parse_args()

    qa_path = f"{args.base}/book{args.book_id}/qa_book{args.book_id}.json"
    out_path = f"{args.base}/book{args.book_id}/extracted_cues.json"
    qa = json.load(open(qa_path))
    if args.limit:
        qa = qa[:args.limit]
    cache = json.load(open(out_path)) if os.path.exists(out_path) else {}

    llm = MockLLM() if args.backend == "mock" else ModalLLMWrapper()
    import time
    todo = [(i, q) for i, q in enumerate(qa) if str(i) not in cache]
    print(f"{len(qa)} câu | đã cache {len(cache)} | cần trích {len(todo)}")
    fails = 0
    for n, (i, q) in enumerate(todo):
        t0 = time.time()
        cue = None
        for attempt in range(4):
            try:
                cue, raw = extract_cue(llm, q["question"], q.get("retrieval_type", ""))
                break
            except Exception as e:
                print(f"  retry {attempt+1}/4 câu {i}: {type(e).__name__} {e}")
                time.sleep(8)
        if cue is None:
            fails += 1
            print(f"  BỎ QUA câu {i} sau 4 lần lỗi (sẽ thử lại lần chạy sau)")
            continue
        cache[str(i)] = cue
        if args.show:
            print(f"\n[{n+1}/{len(todo)}] Q: {q['question'][:90]}")
            print("  oracle :", q.get("cue_completed"))
            print("  trích  :", {a: (cue[a]['value'] if cue[a] else None) for a in AXES})
            print("  conf   :", {a: (cue[a]['confidence'] if cue[a] else None) for a in AXES})
            print(f"  ({time.time()-t0:.1f}s)")
        if (n + 1) % 10 == 0:
            json.dump(cache, open(out_path, "w"), ensure_ascii=False, indent=1)
            print(f"...saved {len(cache)} ({time.time()-t0:.1f}s/câu)")
    json.dump(cache, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"Xong. Cache {len(cache)} cue -> {out_path}")
