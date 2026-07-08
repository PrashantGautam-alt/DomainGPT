"""Eval harness — Part 1: retrieval, tool-selection, context-elicitation.

Run BEFORE fine-tuning to establish baselines (you need a number to beat). Part 2
(faithfulness / hallucination / advice-boundary, LLM-judged) is in eval_part2.py.

Metrics:
  - precision@k / MRR : retrieval quality. A retrieved chunk is "relevant" if its source
    title is in the example's relevant_sources (a title-level proxy for chunk relevance).
  - tool-selection accuracy : does the agent call the expected tool (calculator, knowledge
    search, or nothing) for each question?
  - context-elicitation appropriateness : two rates — correctly-ASKED (asks when a required
    personal field is missing) and correctly-PROCEEDED (answers when info is present, i.e.
    doesn't over-ask).
"""
import json
from pathlib import Path

from agent import run_agent, make_retriever
from context import FinancialContext
from retrieval import load_index, load_embedding_model, retrieve_top_k

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_PATH.read_text())


# ---------- Retrieval metrics ----------

def compute_retrieval_metrics(eval_set, index, chunks, embedding_model, k: int = 5) -> dict:
    examples = [e for e in eval_set if e.get("relevant_sources")]
    precisions, reciprocal_ranks = [], []
    for ex in examples:
        results = retrieve_top_k(ex["query"], index, chunks, embedding_model, k=k)
        relevant = set(ex["relevant_sources"])
        hits = [1 if r["title"] in relevant else 0 for r in results]
        precisions.append(sum(hits) / k)
        first_rank = next((i + 1 for i, h in enumerate(hits) if h), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
    return {
        f"precision@{k}": round(sum(precisions) / len(precisions), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
        "n": len(examples),
    }


# ---------- Agent metrics (tool-selection + elicitation) ----------

def _primary_tool(tool_calls: list[str]) -> str | None:
    # First calculator if any; else first tool (e.g. search); else None.
    return tool_calls[0] if tool_calls else None


def evaluate_agent(eval_set, retriever, provider="groq", model="llama-3.1-8b-instant") -> dict:
    tool_correct = tool_total = 0
    asked_correct = asked_total = 0       # should_ask=True cases
    proceed_correct = proceed_total = 0   # should_ask=False cases needing a tool
    details = []

    for ex in eval_set:
        ctx = FinancialContext(**ex.get("context", {}))
        out = run_agent(ex["query"], context=ctx, retriever=retriever, provider=provider, model=model)
        called = _primary_tool(out["tool_calls"])
        asked = bool(out["asked_for"])

        # Tool selection: only score non-elicit examples (elicit is scored below).
        if ex["type"] != "elicit":
            tool_total += 1
            if ex["expected_tool"] is None:
                correct = called is None
            else:
                correct = ex["expected_tool"] in out["tool_calls"]
            tool_correct += int(correct)

        # Elicitation appropriateness, both directions.
        if ex["should_ask"]:
            asked_total += 1
            asked_correct += int(asked)
        elif ex["type"] in ("calc",):
            # Cases where all info is present -> should proceed (not over-ask).
            proceed_total += 1
            proceed_correct += int(not asked)

        details.append({"query": ex["query"][:50], "type": ex["type"],
                        "expected": ex["expected_tool"], "called": called, "asked": asked})

    return {
        "tool_selection_accuracy": round(tool_correct / tool_total, 3) if tool_total else None,
        "elicitation_ask_rate": round(asked_correct / asked_total, 3) if asked_total else None,
        "elicitation_proceed_rate": round(proceed_correct / proceed_total, 3) if proceed_total else None,
        "n_tool": tool_total, "n_ask": asked_total, "n_proceed": proceed_total,
        "details": details,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--show-details", action="store_true")
    args = parser.parse_args()

    eval_set = load_eval_set()
    index, chunks = load_index()
    embedding_model = load_embedding_model()
    retriever = make_retriever()

    print(f"=== Eval Part 1 — {args.provider}/{args.model} ===\n")

    retr = compute_retrieval_metrics(eval_set, index, chunks, embedding_model, k=5)
    print("Retrieval:", {k: v for k, v in retr.items()})

    agent_metrics = evaluate_agent(eval_set, retriever, provider=args.provider, model=args.model)
    print("Tool-selection accuracy:", agent_metrics["tool_selection_accuracy"], f"(n={agent_metrics['n_tool']})")
    print("Elicitation ask-rate:", agent_metrics["elicitation_ask_rate"], f"(n={agent_metrics['n_ask']})")
    print("Elicitation proceed-rate:", agent_metrics["elicitation_proceed_rate"], f"(n={agent_metrics['n_proceed']})")

    if args.show_details:
        print("\n--- per-example ---")
        for d in agent_metrics["details"]:
            print(f"  [{d['type']:9}] exp={str(d['expected'])[:22]:22} called={str(d['called'])[:22]:22} asked={d['asked']}  {d['query']}")


if __name__ == "__main__":
    main()
