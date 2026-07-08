"""Evaluate a model on the fine-tune-targeted behaviors, using transformers directly on the
server (no API, no vLLM). Measures the two clean quantitative metrics the fine-tune targets:

  - calculator tool-selection accuracy : on calc questions, does the model emit the RIGHT
    tool call, as Llama-3.1 tool-call JSON {"name": ..., "parameters": ...}?
  - context-elicitation appropriateness : on questions missing personal info, does it ASK
    (no tool call + a question) rather than fabricate numbers; and on questions with the
    numbers present, does it PROCEED (emit a tool call) rather than over-ask?

Run for BOTH models on the SAME harness for an apples-to-apples comparison:
  python src/eval_finetuned.py --model models/domaingpt-qlora/merged     # fine-tuned
  python src/eval_finetuned.py --model meta-llama/Llama-3.1-8B-Instruct  # base
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"

# Exact training system prompt (keeps the fine-tuned model in-distribution).
SYSTEM_PROMPT = (
    "You are DomainGPT, a financial-decision assistant for students and early-career "
    "professionals in India. You have calculator tools for affordability, EMI-vs-cash, "
    "budgeting, and job-quit runway. Call the right tool for arithmetic questions rather "
    "than computing numbers yourself. Never invent a personal financial value (income, "
    "expenses, debt, savings) — if one is needed and unknown, ask the user for it. Explain "
    "tradeoffs. You never recommend a specific stock or fund to buy — that stays educational only."
)

CALCULATORS = {
    "affordability_calculator", "emi_vs_cash_calculator",
    "budget_split_calculator", "job_quit_runway_calculator",
}


def parse_tool_name(text: str) -> str | None:
    """Extract the tool name from a Llama-3.1 tool-call JSON {"name": "...", ...}."""
    m = re.search(r'"name"\s*:\s*"(\w+)"', text)
    if m and m.group(1) in CALCULATORS:
        return m.group(1)
    return None


def looks_like_question(text: str) -> bool:
    return "?" in text


def generate(model, tokenizer, query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF id or local path")
    parser.add_argument("--show-details", action="store_true")
    args = parser.parse_args()

    eval_set = json.loads(EVAL_PATH.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # bf16 (no bitsandbytes) — 16GB fits on one 24GB A5000, fewer deps to break.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})

    tool_correct = tool_total = 0
    ask_correct = ask_total = 0
    proceed_correct = proceed_total = 0
    details = []

    for ex in eval_set:
        output = generate(model, tokenizer, ex["query"])
        tool = parse_tool_name(output)
        asked = tool is None and looks_like_question(output)

        if ex["type"] == "calc":
            tool_total += 1
            tool_correct += int(tool == ex["expected_tool"])
            proceed_total += 1
            proceed_correct += int(tool is not None)  # should proceed, not ask
        elif ex["type"] == "elicit":
            ask_total += 1
            ask_correct += int(asked)

        details.append({"type": ex["type"], "expected": ex["expected_tool"],
                        "tool": tool, "asked": asked, "query": ex["query"][:45]})

    def rate(n, d):
        return round(n / d, 3) if d else None

    print(f"\n=== Eval (transformers) — {args.model} ===")
    print(f"Calculator tool-selection accuracy: {rate(tool_correct, tool_total)} (n={tool_total})")
    print(f"Elicitation ask-rate: {rate(ask_correct, ask_total)} (n={ask_total})")
    print(f"Elicitation proceed-rate: {rate(proceed_correct, proceed_total)} (n={proceed_total})")

    if args.show_details:
        print("\n--- per-example ---")
        for d in details:
            print(f"  [{d['type']:9}] exp={str(d['expected'])[:22]:22} tool={str(d['tool'])[:22]:22} asked={d['asked']}  {d['query']}")


if __name__ == "__main__":
    main()
