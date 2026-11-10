# Module 2 — Copy-Paste Prompt Examples

Companion to `1_Prompt_Engineering.html` and `2_Advanced_Reasoning.html`.

**How to run these:** paste any block directly into this Claude Code terminal (or any Claude chat / `claude -p "..."`). Where a "Weak" and "Strong" version are shown, run both back-to-back so you can see the actual difference in output quality — that's the benefit the guide is describing.

---

## PART 1 — PROMPT ENGINEERING

### 1. Zero-Shot Prompting

```
Classify the sentiment of this review as Positive, Negative, or Neutral.
Review: "The battery life is incredible but the camera is disappointing."
```

No examples given — the model relies purely on instruction-following from pretraining.

---

### 2. Few-Shot Prompting

```
Classify sentiment as Positive, Negative, or Neutral.

Review: "Best purchase I've made all year." → Positive
Review: "Stopped working after two days." → Negative
Review: "It's fine, does what it says." → Neutral

Review: "Works okay I guess, nothing special." →
```

Compare this to the zero-shot version above on a trickier, more ambiguous review — few-shot examples pin down the exact label format and boundary cases the instruction alone leaves fuzzy.

---

### 3. Chain-of-Thought — Standard vs. CoT (the benefit, side by side)

**Weak — standard prompting (often gets multi-step math wrong):**
```
Q: A juggler can juggle 16 balls. Half are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: (answer only, one word or number, no explanation)
```

**Strong — few-shot Chain-of-Thought:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many does he have now?
A: Roger started with 5 balls. 2 cans × 3 balls = 6 more balls. 5 + 6 = 11. The answer is 11.

Q: A juggler can juggle 16 balls. Half are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A:
```

**Strong — zero-shot CoT (the "magic phrase", no examples needed):**
```
A juggler can juggle 16 balls. Half are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
Let's think step by step.
```

Run the "Weak" version a few times on a harder multi-step word problem and compare it against either CoT version — CoT spreads the computation across visible tokens instead of forcing it into a single guess.

---

### 4. Self-Consistency (sample multiple reasoning paths, majority vote)

Run this exact prompt 3–5 times in separate turns and tally the final answers yourself — that tally *is* self-consistency:

```
A store had 120 apples. They sold 35% on Monday and 20% of what remained on Tuesday. How many apples are left?
Let's think step by step, and end your answer with "Final answer: X".
```

Or have the model simulate the whole ensemble in one turn, which also demonstrates why voting beats trusting a single chain:

```
Solve the following problem 5 times independently, using a different reasoning approach each time. Do not let earlier attempts influence later ones — write each as if it's the first time you've seen the problem. Then report all 5 final answers and state the majority-vote answer.

Problem: A store had 120 apples. They sold 35% on Monday and 20% of what remained on Tuesday. How many apples are left?
```

---

### 5. Prompt Anatomy — unstructured vs. structured

**Weak — vague, no format spec:**
```
Look at this and tell me what's wrong with it.

def calc(a, b, op):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b
```

**Strong — role, task, context, format all made explicit:**
```
You are a senior Python code reviewer.

Task: Review the function below for bugs and edge cases.

Context: This function will be called with arbitrary user-supplied operator strings from a web form.

def calc(a, b, op):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b

Output format: A numbered list of issues, each as "Issue: ... | Fix: ...". No prose outside the list.
```

---

## PART 2 — ADVANCED REASONING

### 6. Tree-of-Thoughts (search over reasoning paths, not one chain)

```
Use the numbers 4, 9, 10, 13 exactly once each, with +, -, *, / and parentheses, to make 24.

Use a Tree-of-Thoughts approach:
1. Propose 3 different first moves (an operation on two of the numbers).
2. For each, evaluate whether it's "promising" or a "dead end" toward reaching 24.
3. Expand only the promising branch(es) with a second move.
4. Repeat until you reach 24 or exhaust promising branches.

Show the branches you explored and pruned, then give the final expression.
```

Compare against asking for the answer directly with no search structure — Tree-of-Thoughts catches dead ends before committing to them; a single chain-of-thought attempt can't backtrack once it's a few tokens in.

---

### 7. Graph-of-Thoughts (merge two independent branches)

```
Task: Sort this list using merge sort, showing your work: [38, 27, 43, 3, 9, 82, 10]

1. Split the list into two halves and solve each half independently (show each as its own sorted sub-list).
2. Then explicitly AGGREGATE (merge) the two sorted sub-lists into the final sorted list — show the merge step comparing elements pairwise.
```

The aggregation step is the operation a plain Tree-of-Thoughts can't do — two independent branches get combined into one better answer instead of just being pruned or kept.

---

### 8. Program-of-Thoughts / PAL (delegate arithmetic to code)

```
A store had 120 apples. They sold 35% on Monday and 20% of what remained on Tuesday. How many apples are left?

Do not compute the arithmetic yourself in words. Instead, write a short Python snippet that computes the answer, then state the result the code would print.
```

If you're running this inside an environment with code execution enabled (e.g. Claude with the code execution tool, or Claude Code itself), ask it to actually run the snippet — that's the full PAL/PoT loop: the LLM handles the logic, a real interpreter handles the arithmetic.

---

### 9. ReAct (Thought → Action → Observation, grounded in real tool calls)

```
Question: Who directed the highest-grossing film of 2023, and what was that director's next project after it?

Work through this using a Thought / Action / Observation loop:
- Thought: reason about what you need to find out next
- Action: state a search query you would run
- Observation: (if you have web search available, actually run it and use the real result; otherwise state your best-known answer and flag it as unverified)

Repeat until you have enough information, then give the Final Answer.
```

If you have web search enabled (Claude.ai, the API with the `web_search` tool, or Claude Code), this prompt visibly grounds each Thought in a real Observation instead of the model's static training knowledge — compare it to just asking the question directly with no loop and see whether it hedges or guesses on the second part.

---

### 10. Reflection (Reflexion — learn from a failed attempt, within one episode)

```
Attempt 1: Multiply 47 × 68 using only mental arithmetic, showing your work, and give the final number.

Now: Verify Attempt 1 by recomputing it a different way (e.g. distributive expansion: 47×68 = 47×70 - 47×2). If the two results disagree, write a short reflection on what likely went wrong in Attempt 1, then give Attempt 2, corrected.
```

The point isn't the arithmetic — it's the pattern: attempt → evaluate → verbal self-reflection on the failure → informed retry, all within one conversation, no fine-tuning.

---

### 11. Self-Critique (generate → critique → revise)

```
Generate: Write a 150-word explanation of how neural networks learn, for a beginner audience.

Critique: Now critique your own draft above as harshly and specifically as possible — find oversimplifications, inaccuracies, missing caveats, or anything a beginner would misunderstand.

Revise: Now produce a final version that fixes every issue you just raised in the critique.
```

Run just the "Generate" step alone first and compare it to the final revised version — that gap is what the extra critique/revise passes are buying you (and per the guide, self-critique won't catch errors that come from a genuine knowledge gap shared by generator and critic — only an external verifier catches those).

---

## Quick reference — which technique to reach for

| Symptom | Reach for |
|---|---|
| Task is common, format is simple | Zero-shot |
| Model keeps missing your exact output format | Few-shot |
| Multi-step math/logic, wrong answers | Chain-of-thought |
| CoT answer is inconsistent across runs | Self-consistency |
| Problem has real branching/backtracking (puzzles, planning) | Tree-of-Thoughts |
| Problem decomposes into parts that must be recombined | Graph-of-Thoughts |
| Errors are specifically arithmetic/counting | Program-of-Thoughts / PAL |
| Answer needs current/external info | ReAct (with real tools) |
| Task has a clear pass/fail signal (code, tests) | Reflection |
| Output quality/completeness needs polish | Self-critique |
