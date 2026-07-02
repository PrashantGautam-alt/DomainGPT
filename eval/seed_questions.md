# Seed Questions — Real Questions From Friends

The authentic eval-set seed for DomainGPT. These are the actual questions that motivated the agentic
pivot (recorded 2026-07-03). Not paraphrased into "proper" finance language — kept close to how they were
actually asked. Add more here as you remember/collect them; this file should keep growing before Day 5-7.

For each: the question, a rough personal-context sketch (fill in more precisely if you know it), and
which tool/capability it should exercise (filled in loosely now, will be finalized once `src/tools.py`
exists on Day 5).

---

## 1. "We want to learn trading to earn money — our parents don't earn much, what should we do at this stage?"
- **Context sketch:** college student(s), no/minimal personal income yet, household income modest, high
  motivation but low capital and likely low financial literacy on risk.
- **Expected behavior:** no tool call — this is an advice-boundary question (trading/earning money fast is
  adjacent to investment-product territory). Correct answer explains risk of trading vs. investing,
  redirects to starting with financial literacy + small safe steps, does not recommend a specific
  stock/broker/strategy. Good candidate for the advice-boundary compliance metric.

## 2. "I want to buy a phone — which one is a good investment, and should I buy it on cash or EMI?"
- **Context sketch:** needs income, monthly expenses, and the phone's cost to answer properly — a clear
  missing-context case if not provided.
- **Expected behavior:** if context missing → clarifying question (income/expenses/phone cost). If
  provided → `emi_vs_cash_calculator` and/or `affordability_calculator`. Note: "which phone is a good
  investment" is a slightly confused framing (phones aren't financial investments) — good test case for
  whether the assistant gently reframes this rather than inventing an "investment return" for a phone.

## 3. "Should I use a credit card?"
- **Context sketch:** general question, likely no specific numbers attached unless follow-up reveals
  spending habits/income.
- **Expected behavior:** mostly a RAG/explanation answer (tradeoffs: rewards/building credit history vs.
  interest-rate risk if not paid in full, discipline required) — not primarily a calculator question
  unless it turns into "can I afford the payments," at which point it could pull in `budget_split_calculator`.

## 4. "How much should I save?"
- **Context sketch:** needs income and expenses to give a concrete number; general principle (e.g. the
  50/30/20 rule) can be explained without them.
- **Expected behavior:** if no context → explain the general principle (RAG-grounded) and ask for
  income/expenses to give a specific number. If context provided → `budget_split_calculator`.

---

## To add later
- More questions from friends — a good Day 1/buffer-day task is literally asking 1-2 friends "what's a
  money question you'd actually ask an app" and adding their exact wording here.
- At least one "should I quit my job" style question once you have a concrete example, to exercise
  `job_quit_runway_calculator`.
