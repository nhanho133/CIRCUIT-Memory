"""
run_artem_llm.py — standard ARTEM answer-generation + LLM-as-a-Judge on top of the
retrieval results, using a Modal-served LLM (or a mock for a no-GPU smoke test).

Prereqs already produced by the rest of the pipeline:
    <base>/book<id>/qa_book<id>.json                          (bridge_epbench_to_artem.py)
    <base>/book<id>/match_based_retrieval_results_book<id>.json (eventRetriever.py)

Backends:
    --backend mock   : no GPU; canned LLM responses, validates the wiring + F1 plumbing
    --backend modal  : calls your deployed Modal endpoint (set MODAL_LLM_URL)

Usage:
    export MODAL_LLM_URL="https://<workspace>--artem-llm-generate.modal.run"
    python run_artem_llm.py --base data_root_ep200 --book_id 1 --backend modal --limit 50
"""
import os, json, argparse, urllib.request
import pandas as pd

from LLM_as_a_Judge import (
    LLMWrapper, ARTMemorySystem, ARTMemoryEvaluator,
    load_questions_from_json, DeepseekR1Wrapper,
)


class ModalLLMWrapper(LLMWrapper):
    """Calls the Modal web endpoint deployed by modal_llm.py."""
    def __init__(self, url: str = None, max_new_tokens: int = 512, timeout: int = 1800):
        self.url = url or os.environ.get("MODAL_LLM_URL")
        if not self.url:
            raise ValueError("Set MODAL_LLM_URL or pass url= (your deployed Modal endpoint)")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def generate(self, user_prompt: str, system_prompt: str = "", max_new_tokens: int = None) -> str:
        body = json.dumps({
            "prompt": user_prompt,
            "system": system_prompt,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
        }).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode()).get("text", "")


class MockLLM(LLMWrapper):
    """No-GPU stand-in: validates the answer->judge->F1 plumbing end to end."""
    def generate(self, user_prompt: str, system_prompt: str = "", max_new_tokens: int = None) -> str:
        if "judge" in system_prompt.lower() or "expert" in system_prompt.lower():
            return ('{"identified_items_in_AI_answer": ["mock"], '
                    '"matching_score": [{"mock": 1.0}], "explanation": "mock"}')
        return "Mock answer based on the retrieved events."


def summarize(df: pd.DataFrame):
    f1cols = [c for c in df.columns if "f1" in c.lower()]
    print(f"\n#questions evaluated: {len(df)}")
    for c in f1cols:
        print(f"  mean {c}: {df[c].mean():.3f}")
    if f1cols and "retrieval_type" in df.columns:
        c = f1cols[0]
        print(f"\n  {c} by retrieval_type:")
        for t, g in df.groupby("retrieval_type"):
            print(f"    {t:<22} n={len(g):3d}  {g[c].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_root_ep200")
    ap.add_argument("--book_id", type=int, default=1)
    ap.add_argument("--backend", choices=["mock", "modal", "local"], default="mock")
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    args = ap.parse_args()

    if args.backend == "modal":
        llm = ModalLLMWrapper()
    elif args.backend == "local":
        llm = DeepseekR1Wrapper()              # needs a local GPU + valid model path
    else:
        llm = MockLLM()
    print(f"Backend: {args.backend}  ({type(llm).__name__})")

    art = ARTMemorySystem(base_path=args.base)
    art.load_retrieval_data_for_books([args.book_id])

    qa_path = f"{args.base}/book{args.book_id}/qa_book{args.book_id}.json"
    questions = load_questions_from_json(qa_path, args.book_id)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions for book {args.book_id}")

    evaluator = ARTMemoryEvaluator(
        answering_model=llm, judge_model=llm, art_system=art,
        output_dir=f"{args.base}/book{args.book_id}/art_evaluation_results",
    )
    ckpt = f"{args.base}/book{args.book_id}/artem_llm_results.checkpoint.jsonl"
    df = evaluator.evaluate_questions(questions, checkpoint_path=ckpt)
    out = f"{args.base}/book{args.book_id}/artem_llm_results.json"
    df.to_json(out, orient="records", indent=2)
    print(f"\nSaved {out}")
    summarize(df)
