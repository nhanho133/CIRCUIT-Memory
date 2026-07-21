"""
llm_event_extraction.py — trích (time, spaces, entities, content) TỪ RAW CHAPTER TEXT
bằng LLM (Modal circuit-strong), thay cho gold events của EPBench.

Đây là bước còn thiếu để so trực tiếp với con số 0.707 trong paper ARTEM: paper của họ
KHÔNG dùng gold event, mà tự trích event bằng LLM (có lỗi trích) rồi mới retrieval.
Trước đó (data_root_ep200/extracted_features_book1.json) chúng ta đã dùng gold -> đó
là engine ceiling, không so được với 0.707.

Input:  raw book.json (EPBench Uscifi 200-chapter) -> tách "Chapter N ... " bằng regex
        (giống dataExtraction.py gốc của ARTEM).
Output: data_root_ep200_llmevents/book1/extracted_features_book1.json
        cùng schema với gold (chapter, time[list], spaces[str], entities[list], content[str])
        để StemEngine load thẳng, không cần sửa gì khác.
"""
import os, re, json, ast, argparse, urllib.request

BOOK_JSON = ("../episodic-memory-benchmark/epbench/data/Uscifi_Sscifi_seed2/books/"
             "model_claude-3-5-sonnet-20241022_itermax_10_Idefault_nbchapters_200_nbtokens_89736/book.json")
OUT_DIR = "data_root_ep200_llmevents/book1"

PROMPT_TMPL = """You are an information extraction agent.

Extract structured information from **only chapter {ch}** below. Return a **single dictionary** with the following keys:

{{
  "chapter": {ch},
  "date": ["..."],
  "location": "...",
  "entity": ["..."],
  "content": "..."
}}

The content should be a SINGLE SHORT sentence that summarises the book chapter. For example, content could be:
"plasma conduit rupture at the lunar station"
"containment failure in the reactor core"

Return ONLY the dictionary. Do not wrap in a code block. Do not output multiple dictionaries.

Chapter {ch}:
\"\"\"{text}\"\"\"
"""


def call_modal(url, prompt, max_new_tokens=200, timeout=300):
    body = json.dumps({"prompt": prompt, "system": "", "max_new_tokens": max_new_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()).get("text", "")


def parse_response(resp, ch):
    dict_candidates = re.findall(r"\{.*?\}", resp, re.DOTALL)
    for raw in dict_candidates:
        try:
            cleaned = raw.replace("'", '"')
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(raw)
            date = parsed.get("date", [])
            if isinstance(date, str):
                date = [date]
            entity = parsed.get("entity", [])
            if isinstance(entity, str):
                entity = [entity]
            loc = parsed.get("location", "")
            if isinstance(loc, list):
                loc = loc[0] if loc else ""
            content = parsed.get("content", "")
            if isinstance(content, list):
                content = content[0] if content else ""
            return {"chapter": ch, "time": date or ["unknown"], "spaces": str(loc),
                    "entities": entity or ["unknown"], "content": str(content)}
        except Exception:
            continue
    return {"chapter": ch, "time": ["unknown"], "spaces": "unknown",
            "entities": ["unknown"], "content": "extraction failed"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("MODAL_LLM_URL"))
    ap.add_argument("--book", default=BOOK_JSON)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    if not args.url:
        raise ValueError("Set MODAL_LLM_URL")

    raw_text = json.load(open(args.book)) if args.book.endswith(".json") else open(args.book).read()
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    split = re.findall(r"(Chapter\s+(\d+))(.*?)(?=Chapter\s+\d+|$)", raw_text, re.DOTALL)
    chapters = [{"chapter": int(n), "text": t.strip()} for _, n, t in split]
    print(f"Found {len(chapters)} chapters")

    os.makedirs(args.out, exist_ok=True)
    results = []
    for ch in chapters:
        prompt = PROMPT_TMPL.format(ch=ch["chapter"], text=ch["text"])
        resp = call_modal(args.url, prompt)
        ev = parse_response(resp, ch["chapter"])
        results.append(ev)
        print(f"  ch{ch['chapter']:3d}: time={ev['time']} spaces={ev['spaces'][:30]!r} "
              f"entities={ev['entities'][:2]} content={ev['content'][:50]!r}")

    out_path = os.path.join(args.out, "extracted_features_book1.json")
    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} LLM-extracted events -> {out_path}")
